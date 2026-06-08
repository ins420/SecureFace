"""
SecureFace-RX 시각적 데모
얼굴 영역 탐지 → 보호(blur+embed) → 복원 과정을 이미지로 보여줍니다.

사용법:
    python demo_visual.py --img Z:/캡스톤디자인/chacha.jpg
"""
import argparse, sys
from pathlib import Path
import numpy as np
import cv2
import torch

import config as c
from models.embedder import ModelDWT
from models.modules import DWT
from utils.key_gen import generate_key, make_key_rec
from utils.image_processing import Obfuscator, to_tensor, to_numpy


CKPT = "checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth"


def imread_safe(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image
    pil = Image.open(str(path)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def imwrite_safe(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


def detect_faces(img_bgr):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return []
    result = []
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw//2, y + fh//2
        # 이미지 상단 2/3 내에 있어야 실제 얼굴일 가능성 높음
        if cy < h * 0.75:
            result.append((x, y, fw, fh))
    # 없으면 전체 반환
    if not result:
        result = faces.tolist()
    # 면적 기준 내림차순 정렬
    return sorted(result, key=lambda r: r[2]*r[3], reverse=True)


def square_crop(img, x, y, w, h, margin=0.15):
    """마진 포함 정사각형으로 크롭"""
    m = int(margin * max(w, h))
    x1 = max(0, x - m); y1 = max(0, y - m)
    x2 = min(img.shape[1], x + w + m); y2 = min(img.shape[0], y + h + m)
    ch, cw = y2-y1, x2-x1
    side = max(ch, cw)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    sx1 = max(0, cx-side//2); sy1 = max(0, cy-side//2)
    sx2 = min(img.shape[1], sx1+side); sy2 = min(img.shape[0], sy1+side)
    return img[sy1:sy2, sx1:sx2], (sx1, sy1, sx2-sx1, sy2-sy1)


def run_pipeline(embedder, dwt_mod, face_crop, password=0, obf='blur'):
    """보호 -> 복원 -> 오복원 수행, 결과 dict 반환"""
    device = next(embedder.parameters()).device
    obfuscator = Obfuscator(obf_type=obf)

    img = cv2.resize(face_crop, (c.NORM_RESOLUTION, c.NORM_RESOLUTION),
                     interpolation=cv2.INTER_AREA)
    xa = to_tensor(img, device=device)
    xa_obfs = obfuscator(xa)

    skey = generate_key(password, bs=1,
                         w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
    skey_dwt = dwt_mod(skey.float())

    with torch.no_grad():
        xa_out_z, xa_proc = embedder(xa, xa_obfs, skey_dwt)
        del xa_out_z

        key_rec = make_key_rec(skey_dwt)
        xa_rev, _ = embedder(key_rec, xa_proc, skey_dwt, rev=True)

        skey_w = generate_key(str(password)+"_WRONG", bs=1,
                               w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
        skey_dwt_w = dwt_mod(skey_w.float())
        key_rec_w = make_key_rec(skey_dwt_w)
        xa_wrong, _ = embedder(key_rec_w, xa_proc, skey_dwt_w, rev=True)

    def norm01(t):
        return ((t.squeeze(0).permute(1,2,0).clamp(-1.,1.).cpu().float().numpy()
                 + 1.0) / 2.0).astype(np.float64)

    return {
        # uint8 BGR — 저장 및 시각화용
        "original":   to_numpy(xa.cpu()),
        "obfuscated": to_numpy(xa_obfs.cpu()),
        "protected":  to_numpy(xa_proc.cpu()),
        "restored":   to_numpy(xa_rev.cpu()),
        "wrong":      to_numpy(xa_wrong.cpu()),
        # float [0,1] RGB — 정밀 지표용
        "original_f":   norm01(xa),
        "obfuscated_f": norm01(xa_obfs),
        "protected_f":  norm01(xa_proc),
        "restored_f":   norm01(xa_rev),
        "wrong_f":      norm01(xa_wrong),
    }


def make_comparison_panel(res, face_loc=None):
    """
    5단계 비교 패널 생성
    [원본] [블러(y)] [보호본(y_hat)] | [복원(x_rec)] [오복원(x_wrong)]
    """
    size = 256
    labels = [
        ("1. Original", res["original"]),
        ("2. Blur (y)", res["obfuscated"]),
        ("3. Protected (y_hat)", res["protected"]),
        ("4. Restored [correct key]", res["restored"]),
        ("5. Wrong key (garbage)", res["wrong"]),
    ]
    panels = []
    for title, img in labels:
        tile = cv2.resize(img, (size, size))
        # 제목 배경
        header = np.zeros((36, size, 3), dtype=np.uint8)
        cv2.putText(header, title, (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        tile = np.vstack([header, tile])
        panels.append(tile)

    # 구분선
    div = np.zeros((size+36, 3, 3), dtype=np.uint8)
    div[:] = (60, 60, 200)

    row1 = np.hstack(panels[:3] + [div] + panels[3:])

    # 설명 바
    info = np.zeros((38, row1.shape[1], 3), dtype=np.uint8)
    msgs = [
        "< Protection: y_hat looks like y (blurred), but hides original >",
        "< Restoration: correct key -> original recovered >",
        "< Security: wrong key -> cannot recover original >",
    ]
    x_pos = [8, size*3+18, size*3+18+size+8]
    for msg, xp in zip(msgs, x_pos):
        cv2.putText(info, msg, (xp, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1)

    panel = np.vstack([row1, info])
    return panel


def main():
    parser = argparse.ArgumentParser(description="SecureFace-RX 시각적 데모")
    parser.add_argument("--img", required=True, help="입력 이미지 경로")
    parser.add_argument("--ckpt", default=CKPT, help=".pth 가중치 경로")
    parser.add_argument("--password", default=0, help="비밀번호")
    parser.add_argument("--obf", default="blur",
                        choices=["blur","pixelate","median","mask"])
    parser.add_argument("--out", default="demo_output", help="결과 저장 폴더")
    parser.add_argument("--all-faces", action="store_true",
                        help="탐지된 모든 얼굴 처리")
    parser.add_argument("--face-idx", type=int, default=0,
                        help="처리할 얼굴 인덱스 (0=최대, 1,2... 순서)")
    args = parser.parse_args()

    # ── 모델 로드 ──────────────────────────────────────────────────
    if not Path(args.ckpt).exists():
        sys.exit(f"[오류] 가중치 없음: {args.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = ModelDWT(n_blocks=c.INV_BLOCKS).to(device).eval()
    dwt_mod = DWT().to(device)
    state = torch.load(args.ckpt, map_location=device)
    embedder.load_state_dict(state, strict=False)
    print(f"[OK] 모델 로드 완료 (device={device})")

    # ── 이미지 로드 ────────────────────────────────────────────────
    img = imread_safe(args.img)
    if img is None:
        sys.exit(f"[오류] 이미지 로드 실패: {args.img}")
    print(f"[OK] 이미지 로드: {args.img}  {img.shape}")

    out_dir = Path(args.out) if Path(args.out).is_absolute() else Path(__file__).parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 얼굴 탐지 ─────────────────────────────────────────────────
    faces = detect_faces(img)
    print(f"[OK] 얼굴 탐지: {len(faces)}개")

    if not faces:
        print("  얼굴 탐지 실패 -> 이미지 중앙 크롭 사용")
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h-side)//2, (w-side)//2
        faces = [(x0, y0, side, side)]

    from skimage.metrics import structural_similarity as sk_ssim

    def psnr(a, b):
        mse = np.mean((a - b)**2)
        return 100.0 if mse<1e-20 else 20*np.log10(1.0/(np.sqrt(mse)+1e-12))

    def ssim(a, b):
        return sk_ssim(a, b, channel_axis=-1, data_range=1.0)

    def ck(v, op, t, n):
        ok = (v>=t if op=='>=' else v>t if op=='>' else v<t)
        print(f"  {'[OK]' if ok else '[--]'}  {n} = {v:.2f}  (목표 {op} {t})")

    # 처리할 얼굴 목록 결정
    if args.all_faces:
        targets = list(range(len(faces)))
    else:
        idx = min(args.face_idx, len(faces)-1)
        targets = [idx]

    # 원본 이미지에 모든 탐지 박스 그리기
    vis = img.copy()
    for fi, (x, y, fw, fh) in enumerate(faces):
        color = (0, 255, 0) if fi in targets else (100, 100, 100)
        cv2.rectangle(vis, (x, y), (x+fw, y+fh), color, 2)
        cv2.putText(vis, f"#{fi}", (x, max(y-6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    imwrite_safe(str(out_dir/"original_with_detection.png"), vis)

    for idx in targets:
        x, y, fw, fh = faces[idx]
        crop, (cx, cy, cw, ch) = square_crop(img, x, y, fw, fh)
        tag = f"face{idx}"
        print(f"\n[처리 중] 얼굴 #{idx}  ({cx},{cy}) {cw}x{ch}px"
              f"  obf={args.obf}, password={args.password}")
        res = run_pipeline(embedder, dwt_mod, crop, args.password, args.obf)

        rps_p = psnr(res["protected_f"],  res["obfuscated_f"])
        rps_s = ssim(res["protected_f"],  res["obfuscated_f"])
        rec_p = psnr(res["restored_f"],   res["original_f"])
        rec_s = ssim(res["restored_f"],   res["original_f"])
        wrg_p = psnr(res["wrong_f"],      res["original_f"])
        wrg_s = ssim(res["wrong_f"],      res["original_f"])

        print("  ===== KPI =====")
        ck(rps_p, '>=', 49.0, "RPS  PSNR")
        ck(rps_s, '>=',  0.99, "RPS  SSIM")
        ck(rec_p, '>=', 40.0, "Rec  PSNR")
        ck(rec_s, '>=',  0.97, "Rec  SSIM")
        ck(wrg_p, '<',  11.0, "Wrg  PSNR")
        ck(wrg_s, '<',   0.20, "Wrg  SSIM")

        # 저장 (uint8 이미지 — float 키는 제외)
        save_keys = ["original","obfuscated","protected","restored","wrong"]
        for name in save_keys:
            imwrite_safe(str(out_dir/f"{tag}_{name}.png"), res[name])

        panel = make_comparison_panel(res)
        imwrite_safe(str(out_dir/f"{tag}_comparison.png"), panel)
        print(f"  저장: {out_dir}/{tag}_comparison.png")

    print(f"\n[저장 완료] {out_dir.resolve()}")
    print("  face#_comparison.png        - 5단계 비교 이미지 (얼굴별)")
    print("  original_with_detection.png - 탐지된 얼굴 위치 표시")


if __name__ == "__main__":
    main()
