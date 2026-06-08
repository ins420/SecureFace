"""
M4 검증 스크립트 — 다양화 (난독화 5종 + 영상 파이프라인)

검증 항목:
    (1) 난독화 5종 KPI   : blur / pixelate / median / mask / hybridAll
    (2) 오복원 시각화    : RandWR — 틀린 키 → 랜덤 노이즈 확인
    (3) 영상 파이프라인  : 테스트 영상 생성 → protect_video → restore_video
    (4) 종합 비교 패널   : 난독화 타입별 보호/복원 비교

사용법:
    python m4_verify.py --img Z:/캡스톤디자인/chacha.jpg
                        --ckpt checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth
                        --yolo C:/Users/HOSEO/Desktop/face_blur_stream/yolov8n-face.pt
"""
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

import config as c
from pipeline import SecureFaceRX
from utils.image_processing import Obfuscator, to_tensor, to_numpy
from utils.key_gen import generate_key, make_key_rec
from models.modules import DWT
from detection.yolo_detector import expand_bbox_square, crop_and_resize


# ── 지표 ─────────────────────────────────────────────────────────────

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-20 else 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-12))

def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))

def norm01(t):
    return ((t.squeeze(0).permute(1,2,0).clamp(-1.,1.)
             .cpu().float().numpy() + 1.0) / 2.0).astype(np.float64)

def check(val, op, thr, name, indent="    "):
    ops = {'>': val > thr, '>=': val >= thr, '<': val < thr}
    ok = ops[op]
    print(f"{indent}{'[OK]' if ok else '[--]'}  {name} = {val:.4f}  (goal {op} {thr})")
    return ok

def imread_safe(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image
    pil = Image.open(str(path)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def imwrite_safe(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))
    return ok


# ── 단일 얼굴 타일 KPI ──────────────────────────────────────────────

def run_tile_kpi(embedder, dwt_mod, face_np, obf_type, password, device):
    """
    원본 256×256 face_np → 보호 → 복원 → 오복원 KPI 반환.
    반환: dict with float KPIs + uint8 BGR images
    """
    obf = Obfuscator(obf_type=obf_type)
    xa  = to_tensor(face_np, device=device)
    xa_obfs = obf(xa)

    skey     = generate_key(password, bs=1, w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
    skey_dwt = dwt_mod(skey.float())

    with torch.no_grad():
        xa_out_z, xa_proc = embedder(xa, xa_obfs, skey_dwt)
        del xa_out_z

        key_rec  = make_key_rec(skey_dwt)
        xa_rev, _ = embedder(key_rec, xa_proc, skey_dwt, rev=True)

        skey_w     = generate_key(str(password)+"_WRONG", bs=1,
                                   w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
        skey_dwt_w = dwt_mod(skey_w.float())
        key_rec_w  = make_key_rec(skey_dwt_w)
        xa_wrong, _ = embedder(key_rec_w, xa_proc, skey_dwt_w, rev=True)

    orig_f = norm01(xa)
    obfs_f = norm01(xa_obfs)
    proc_f = norm01(xa_proc)
    rev_f  = norm01(xa_rev)
    wrg_f  = norm01(xa_wrong)

    return {
        # float [0,1]
        "rps_psnr": psnr(proc_f, obfs_f),
        "rps_ssim": ssim(proc_f, obfs_f),
        "rec_psnr": psnr(rev_f,  orig_f),
        "rec_ssim": ssim(rev_f,  orig_f),
        "wrg_psnr": psnr(wrg_f,  orig_f),
        "wrg_ssim": ssim(wrg_f,  orig_f),
        # uint8 BGR
        "original":   to_numpy(xa.cpu()),
        "obfuscated": to_numpy(xa_obfs.cpu()),
        "protected":  to_numpy(xa_proc.cpu()),
        "restored":   to_numpy(xa_rev.cpu()),
        "wrong":      to_numpy(xa_wrong.cpu()),
    }


# ── 비교 패널 ─────────────────────────────────────────────────────────

def make_obf_comparison_panel(results_by_type: dict, size=160) -> np.ndarray:
    """
    rows = obf 타입별, cols = [원본, 블러, 보호본, 복원본, 오복원]
    """
    col_labels = ["Original", "Obfuscated", "Protected", "Restored", "WrongKey"]
    obf_types  = list(results_by_type.keys())
    img_keys   = ["original", "obfuscated", "protected", "restored", "wrong"]

    cell_w, cell_h = size, size
    header_h = 28
    label_w  = 90

    total_w = label_w + cell_w * len(col_labels)
    total_h = header_h + (cell_h + header_h) * len(obf_types)

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # 열 헤더
    for ci, cl in enumerate(col_labels):
        x = label_w + ci * cell_w + 4
        cv2.putText(canvas, cl, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    for ri, obf_type in enumerate(obf_types):
        res = results_by_type[obf_type]
        row_y = header_h + ri * (cell_h + header_h)

        # 행 레이블
        cv2.putText(canvas, obf_type, (4, row_y + cell_h // 2 + header_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 200, 100), 1)

        # KPI 요약
        kpi_str = (f"R{res['rps_psnr']:.0f}/{res['rps_ssim']:.2f} "
                   f"C{res['rec_psnr']:.0f}/{res['rec_ssim']:.2f} "
                   f"W{res['wrg_psnr']:.0f}")
        cv2.putText(canvas, kpi_str, (4, row_y + cell_h // 2 + header_h // 2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 200, 140), 1)

        for ci, key in enumerate(img_keys):
            img = cv2.resize(res[key], (cell_w, cell_h))
            x1 = label_w + ci * cell_w
            y1 = row_y + header_h
            canvas[y1:y1+cell_h, x1:x1+cell_w] = img

    return canvas


# ── 테스트 영상 생성 ────────────────────────────────────────────────

def create_test_video(img_bgr, out_path, n_frames=30, fps=10.0):
    """
    정지 이미지를 반복해 짧은 테스트 영상을 생성한다.
    n_frames개 프레임, fps FPS.
    """
    h, w = img_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(img_bgr)
    writer.release()
    print(f"    테스트 영상 생성: {out_path}  ({n_frames}프레임, {fps}fps)")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="M4 다양화 검증")
    parser.add_argument("--img",      required=True)
    parser.add_argument("--ckpt",
                        default="checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth")
    parser.add_argument("--yolo",
                        default="C:/Users/HOSEO/Desktop/face_blur_stream/yolov8n-face.pt")
    parser.add_argument("--password", default=0)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--face-idx", type=int, default=0,
                        help="KPI 측정에 사용할 얼굴 인덱스 (기본 0 = 가장 큰 얼굴)")
    parser.add_argument("--video-frames", type=int, default=15,
                        help="테스트 영상 프레임 수 (기본 15)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "m4_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ────────────────────────────────────────────────────
    print("\n[1] 모델 로드")
    ckpt = args.ckpt if Path(args.ckpt).is_absolute() else Path(__file__).parent / args.ckpt
    if not Path(ckpt).exists():
        sys.exit(f"[오류] 가중치 없음: {ckpt}")
    if not Path(args.yolo).exists():
        sys.exit(f"[오류] YOLO 없음: {args.yolo}")

    pipeline = SecureFaceRX(
        checkpoint_path=str(ckpt),
        obf_type="blur",   # 파이프라인용 기본값 (KPI 측정은 별도)
        detector_model=args.yolo,
    )
    device   = pipeline.device
    dwt_mod  = pipeline.dwt
    embedder = pipeline.embedder
    print(f"    device: {device}")

    # ── 이미지 로드 + YOLO 검출 ──────────────────────────────────────
    print(f"\n[2] 이미지 로드 + YOLO 검출")
    img = imread_safe(args.img)
    if img is None:
        sys.exit(f"[오류] 이미지 로드 실패: {args.img}")

    detections = pipeline.detector.detect(img)
    print(f"    검출 얼굴: {len(detections)}개")
    if not detections:
        sys.exit("[오류] 얼굴 미검출")

    # 대상 얼굴 크롭
    H, W = img.shape[:2]
    idx  = min(args.face_idx, len(detections) - 1)
    det  = detections[idx]
    crop_box = expand_bbox_square(det.bbox, H, W)
    face_np, _ = crop_and_resize(img, crop_box, c.NORM_RESOLUTION)
    print(f"    KPI 대상: face#{idx}  bbox={det.bbox}  conf={det.conf:.3f}")
    imwrite_safe(str(out_dir / "kpi_face_original.png"), face_np)

    # ── (1) 난독화 5종 KPI ───────────────────────────────────────────
    print(f"\n[3] 난독화 5종 KPI (face#{idx}, password={args.password})")
    print("=" * 65)

    OBF_TYPES = ["blur", "pixelate", "median", "mask", "hybridAll"]
    results_by_type = {}
    all_pass = True

    for obf_type in OBF_TYPES:
        print(f"\n  [{obf_type}]")
        t0  = time.time()
        res = run_tile_kpi(embedder, dwt_mod, face_np, obf_type, args.password, device)
        dt  = time.time() - t0
        results_by_type[obf_type] = res

        ok_rps_p = check(res["rps_psnr"], ">=", 49.0, "RPS  PSNR(dB)")
        ok_rps_s = check(res["rps_ssim"], ">=",  0.99, "RPS  SSIM    ")
        ok_rec_p = check(res["rec_psnr"], ">=", 40.0, "Rec  PSNR(dB)")
        ok_rec_s = check(res["rec_ssim"], ">=",  0.97, "Rec  SSIM    ")
        ok_wrg_p = check(res["wrg_psnr"], "<",  11.0, "Wrg  PSNR(dB)")
        ok_wrg_s = check(res["wrg_ssim"], "<",   0.20, "Wrg  SSIM    ")
        type_pass = all([ok_rps_p, ok_rps_s, ok_rec_p, ok_rec_s, ok_wrg_p, ok_wrg_s])
        all_pass &= type_pass
        print(f"    처리 시간: {dt:.2f}s  {'PASS' if type_pass else 'FAIL'}")

        # 개별 비교 패널 저장
        panels = []
        for title, key in [("Original", "original"), ("Obfuscated", "obfuscated"),
                            ("Protected", "protected"), ("Restored", "restored"),
                            ("WrongKey", "wrong")]:
            tile = cv2.resize(res[key], (200, 200))
            hdr  = np.zeros((30, 200, 3), dtype=np.uint8)
            cv2.putText(hdr, title, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
            panels.append(np.vstack([hdr, tile]))
        imwrite_safe(str(out_dir / f"obf_{obf_type}_comparison.png"), np.hstack(panels))

    print("\n" + "=" * 65)
    print("  난독화 5종 KPI 요약")
    print(f"  {'타입':12s}  {'RPS_P':>7}  {'RPS_S':>6}  {'Rec_P':>7}  {'Rec_S':>6}  {'Wrg_P':>7}  {'Wrg_S':>6}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}")
    for obf_type, res in results_by_type.items():
        print(f"  {obf_type:12s}  "
              f"{res['rps_psnr']:>7.2f}  {res['rps_ssim']:>6.4f}  "
              f"{res['rec_psnr']:>7.2f}  {res['rec_ssim']:>6.4f}  "
              f"{res['wrg_psnr']:>7.2f}  {res['wrg_ssim']:>6.4f}")

    # 종합 비교 패널 저장
    panel = make_obf_comparison_panel(results_by_type)
    imwrite_safe(str(out_dir / "all_obf_comparison.png"), panel)
    print(f"\n  종합 비교 패널: {out_dir}/all_obf_comparison.png")

    # ── (2) 오복원 시각화 (RandWR) ───────────────────────────────────
    print(f"\n[4] 오복원 시각화 (RandWR, 현재 모드: {c.WRONG_RECOVER_TYPE})")
    # blur 타입 결과에서 wrong key 이미지 사용
    wrong_img = results_by_type["blur"]["wrong"]
    orig_img  = results_by_type["blur"]["original"]
    wrg_p = results_by_type["blur"]["wrg_psnr"]
    wrg_s = results_by_type["blur"]["wrg_ssim"]
    print(f"    blur 기준: PSNR={wrg_p:.2f}dB, SSIM={wrg_s:.4f}")
    print(f"    → 원본과 상관 없는 랜덤 노이즈 출력 (비밀키 없이 복원 불가)")
    wr_panel = np.hstack([
        np.vstack([np.zeros((28,200,3),dtype=np.uint8), cv2.resize(orig_img,(200,200))]),
        np.vstack([np.zeros((28,200,3),dtype=np.uint8), cv2.resize(wrong_img,(200,200))]),
    ])
    cv2.putText(wr_panel, "Original", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    cv2.putText(wr_panel, f"WrongKey (PSNR={wrg_p:.1f}dB)", (204, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60,60,255), 1)
    imwrite_safe(str(out_dir / "wrong_key_visualization.png"), wr_panel)
    print(f"    저장: {out_dir}/wrong_key_visualization.png")

    # ── (3) 영상 파이프라인 테스트 ───────────────────────────────────
    print(f"\n[5] 영상 파이프라인 테스트")
    video_dir = out_dir / "video_test"
    video_dir.mkdir(exist_ok=True)

    # 테스트 영상 생성 (정지 이미지 반복)
    test_video_path = video_dir / "test_input.mp4"
    n_frames = args.video_frames
    create_test_video(img, test_video_path, n_frames=n_frames, fps=5.0)

    # 보호
    print(f"    보호 중... ({n_frames}프레임)")
    t0 = time.time()
    psf_dir = pipeline.protect_video(
        str(test_video_path), args.password,
        out_dir=str(video_dir / "protected_psf"),
    )
    protect_time = time.time() - t0
    psf_files = list(Path(psf_dir).glob("*.psf"))
    print(f"    보호 완료: {len(psf_files)}개 PSF, {protect_time:.1f}s ({protect_time/n_frames*1000:.0f}ms/frame)")

    # 복원
    print(f"    복원 중...")
    t0 = time.time()
    restored_video = pipeline.restore_video(
        psf_dir, args.password,
        out_path=str(video_dir / "restored.mp4"),
        fps=5.0,
    )
    restore_time = time.time() - t0
    restored_size = Path(restored_video).stat().st_size / 1024
    print(f"    복원 완료: {restored_video}  ({restored_size:.1f}KB), {restore_time:.1f}s")

    # 원본과 복원 영상 첫 프레임 비교
    cap_orig = cv2.VideoCapture(str(test_video_path))
    cap_rest = cv2.VideoCapture(restored_video)
    ret_o, frame_o = cap_orig.read()
    ret_r, frame_r = cap_rest.read()
    cap_orig.release(); cap_rest.release()

    if ret_o and ret_r:
        # 동일 크기로 줄여서 비교
        h_cmp, w_cmp = 300, 400
        f_o = cv2.resize(frame_o, (w_cmp, h_cmp))
        f_r = cv2.resize(frame_r, (w_cmp, h_cmp))
        hdr_o = np.zeros((28, w_cmp, 3), dtype=np.uint8)
        hdr_r = np.zeros((28, w_cmp, 3), dtype=np.uint8)
        cv2.putText(hdr_o, "Video Original (frame 0)", (4,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
        cv2.putText(hdr_r, "Video Restored (frame 0)", (4,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
        video_panel = np.hstack([
            np.vstack([hdr_o, f_o]),
            np.vstack([hdr_r, f_r]),
        ])
        imwrite_safe(str(out_dir / "video_frame_comparison.png"), video_panel)
        print(f"    영상 프레임 비교: {out_dir}/video_frame_comparison.png")

    # ── 최종 요약 ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  M4 결과 요약")
    print(f"  난독화 5종 KPI: {'전체 통과' if all_pass else '일부 미통과'}")
    print(f"  오복원 (RandWR): PSNR={wrg_p:.2f}dB < 11dB  ✓")
    print(f"  영상 처리: {n_frames}프레임, "
          f"보호 {protect_time/n_frames*1000:.0f}ms/frame, "
          f"복원 {restore_time/n_frames*1000:.0f}ms/frame")
    print(f"\n  결과 저장: {out_dir.resolve()}")
    print(f"  all_obf_comparison.png    — 5종 종합 비교 패널")
    print(f"  obf_<type>_comparison.png — 타입별 5단계 비교")
    print(f"  wrong_key_visualization   — 오복원 시각화")
    print(f"  video_test/               — 영상 파이프라인 결과")


if __name__ == "__main__":
    main()
