"""
SecureFace-RX 관제 화면
Flask + OpenCV MJPEG 스트리밍

실행:
    python app.py
    브라우저에서 http://localhost:5000 접속
"""
import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import config as c
from pipeline import SecureFaceRX

# ── 설정 ─────────────────────────────────────────────────────────────
CKPT = str(Path(__file__).parent /
           "checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth")
YOLO = "C:/Users/HOSEO/Desktop/face_blur_stream/yolov8n-face.pt"
PSF_PATH = str(Path(__file__).parent / "web_output" / "capture.psf")
Path(PSF_PATH).parent.mkdir(exist_ok=True)

app = Flask(__name__)

# ── 전역 상태 ─────────────────────────────────────────────────────────
pipeline = None
cap      = None
lock     = threading.Lock()

state = {
    "frame":        None,   # 웹캠 원본 프레임
    "display":      None,   # 블러 적용된 스트림 프레임
    "face_count":   0,
    "status":       "초기화 중...",
    "processing":   False,
    "restored_b64": None,   # 복원 결과 (새 창에서 표시)
    "protected_b64": None,  # 보호 캡처 결과
}


# ── 모델 초기화 ───────────────────────────────────────────────────────
def init_pipeline():
    global pipeline
    state["status"] = "모델 로드 중..."
    pipeline = SecureFaceRX(
        checkpoint_path=CKPT,
        obf_type="blur",
        detector_model=YOLO,
    )
    state["status"] = "준비 완료"


# ── 웹캠 캡처 스레드 ──────────────────────────────────────────────────
def capture_loop():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tick       = 0
    last_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        display = frame.copy()

        # YOLO 검출 (3프레임마다)
        if tick % 3 == 0 and pipeline is not None:
            try:
                dets = pipeline.detector.detect(frame)
                last_boxes = [d.bbox for d in dets]
            except Exception:
                last_boxes = []

        # 얼굴 영역 블러 + 박스
        for (x1, y1, x2, y2) in last_boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 > x1 and y2 > y1:
                roi = display[y1:y2, x1:x2]
                k   = max(21, ((x2 - x1) // 8) * 2 + 1)
                display[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 100), 2)

        tick += 1
        with lock:
            state["frame"]      = frame.copy()
            state["display"]    = display
            state["face_count"] = len(last_boxes)

        time.sleep(0.03)


# ── MJPEG 스트림 ──────────────────────────────────────────────────────
def gen_frames():
    while True:
        with lock:
            frame = state["display"]
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


def img_to_b64(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ── Flask 라우트 ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def get_status():
    with lock:
        return jsonify({
            "face_count": state["face_count"],
            "status":     state["status"],
            "processing": state["processing"],
            "has_psf":    Path(PSF_PATH).exists(),
        })


# ── 보호 캡처 ─────────────────────────────────────────────────────────
@app.route("/protect", methods=["POST"])
def protect():
    if state["processing"]:
        return jsonify({"ok": False, "msg": "처리 중입니다."})
    if pipeline is None:
        return jsonify({"ok": False, "msg": "모델 로드 중입니다."})

    password = request.json.get("password", 0)

    with lock:
        frame = state["frame"]
    if frame is None:
        return jsonify({"ok": False, "msg": "웹캠 프레임 없음"})

    def run():
        with lock:
            state["processing"] = True
            state["status"]     = "보호 처리 중..."
            state["restored_b64"]  = None   # 이전 복원 결과 초기화
            state["protected_b64"] = None
        try:
            protected_frame, _ = pipeline.protect_image(
                frame, password, out_psf=PSF_PATH
            )
            with lock:
                state["protected_b64"] = img_to_b64(protected_frame)
                state["status"]        = f"보호 완료 ✓  얼굴 {state['face_count']}개 처리됨"
        except Exception as e:
            with lock:
                state["status"] = f"오류: {e}"
        finally:
            with lock:
                state["processing"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


# ── 복원 ─────────────────────────────────────────────────────────────
@app.route("/restore", methods=["POST"])
def restore():
    if state["processing"]:
        return jsonify({"ok": False, "msg": "처리 중입니다."})
    if not Path(PSF_PATH).exists():
        return jsonify({"ok": False, "msg": "먼저 보호 캡처를 해주세요."})

    password = request.json.get("password", 0)

    def run():
        with lock:
            state["processing"] = True
            state["status"]     = "복원 처리 중..."
        try:
            restored = pipeline.restore_image(PSF_PATH, password)
            with lock:
                state["restored_b64"] = img_to_b64(restored)
                state["status"]       = "복원 완료 ✓"
        except Exception as e:
            with lock:
                state["status"] = f"오류: {e}"
        finally:
            with lock:
                state["processing"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


# ── 복원 결과 뷰 (새 창) ──────────────────────────────────────────────
@app.route("/view")
def view():
    """복원 결과를 보여주는 전용 페이지 (새 창으로 열림)"""
    return render_template("view.html")


@app.route("/result_data")
def result_data():
    with lock:
        return jsonify({
            "restored":  state["restored_b64"],
            "protected": state["protected_b64"],
            "status":    state["status"],
        })


# ── 실행 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=init_pipeline, daemon=True).start()
    threading.Thread(target=capture_loop, daemon=True).start()
    print("\n[SecureFace-RX] http://localhost:5000 에서 접속하세요\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
