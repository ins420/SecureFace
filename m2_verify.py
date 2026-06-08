"""
M2 검증 스크립트 — YOLO 검출 + 전체 파이프라인 end-to-end

검증 항목:
    (1) YOLO 얼굴 검출   : 신뢰도·bbox 출력
    (2) 보호 파이프라인  : 원본 → 보호본(y_hat) + PSF 저장
    (3) PSF 무결성       : SHA-256 검증
    (4) 복원 파이프라인  : PSF → 복원본(x_rec)
    (5) KPI 측정         : M1 기준과 동일 (face 타일 단위)
    (6) 오복원 테스트    : 틀린 패스워드 → 쓰레기 출력
    (7) 시각 비교 패널   : 저장

사용법:
    python m2_verify.py --img Z:/캡스톤디자인/chacha.jpg
                        --ckpt checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth
                        --yolo C:/Users/HOSEO/Desktop/face_blur_stream/yolov8n-face.pt
                        --password 0
                        --out m2_output
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
import torch

import config as c
from pipeline import SecureFaceRX
from utils.image_processing import to_tensor, to_numpy
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
    """[-1,1] tensor → float64 HWC [0,1]"""
    return ((t.squeeze(0).permute(1, 2, 0).clamp(-1., 1.)
              .cpu().float().numpy() + 1.0) / 2.0).astype(np.float64)


def check(val, op, thr, name):
    ops = {'>': val > thr, '>=': val >= thr, '<': val < thr}
    ok = ops[op]
    print(f"    {'[OK]' if ok else '[--]'}  {name} = {val:.4f}  (goal {op} {thr})")
    return ok


# ── 이미지 로더 (한글 경로 안전) ────────────────────────────────────

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


# ── 비교 패널 생성 ────────────────────────────────────────────────────

def make_panel(tiles: dict, size=256) -> np.ndarray:
    """
    {label: BGR ndarray} 딕셔너리를 받아 가로로 붙인 비교 패널 반환.
    """
    panels = []
    for title, img in tiles.items():
        tile = cv2.resize(img, (size, size))
        header = np.zeros((36, size, 3), dtype=np.uint8)
        cv2.putText(header, title, (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        panels.append(np.vstack([header, tile]))
    return np.hstack(panels)


# ── 단계별 KPI 측정 (face 타일 기준, M1 동일 방식) ─────────────────────

def measure_kpi(
    pipeline: SecureFaceRX,
    face_np: np.ndarray,            # 256×256 원본 크롭 (uint8 BGR)
    protected_tile: np.ndarray,     # 보호 타일 — float32 CHW [-1,1] 또는 uint8 HWC BGR
    password,
):
    """
    (원본, 보호타일) 쌍에서 RPS·Rec·Wrg KPI 계산.
    """
    device = pipeline.device
    dwt    = pipeline.dwt

    xa = to_tensor(face_np, device=device)

    # float32 CHW 타일이면 직접 변환 (양자화 손실 없음)
    if (isinstance(protected_tile, np.ndarray)
            and protected_tile.dtype == np.float32
            and protected_tile.ndim == 3
            and protected_tile.shape[0] == 3):
        xa_proc = torch.from_numpy(protected_tile).unsqueeze(0).to(device)
    else:
        xa_proc = to_tensor(protected_tile, device=device)

    # 블러 (RPS 비교 기준)
    xa_obfs = pipeline.obfuscator(xa)

    # 복원 — 올바른 키
    skey     = generate_key(password, bs=1, w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
    skey_dwt = dwt(skey.float())
    key_rec  = make_key_rec(skey_dwt)

    with torch.no_grad():
        xa_rev, _ = pipeline.embedder(key_rec, xa_proc, skey_dwt, rev=True)

    # 복원 — 틀린 키
    skey_w     = generate_key(str(password) + "_WRONG", bs=1,
                               w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
    skey_dwt_w = dwt(skey_w.float())
    key_rec_w  = make_key_rec(skey_dwt_w)

    with torch.no_grad():
        xa_wrong, _ = pipeline.embedder(key_rec_w, xa_proc, skey_dwt_w, rev=True)

    # protected_tile이 float32 CHW면 시각화용 uint8 BGR로 변환
    prot_vis = to_numpy(xa_proc.cpu())

    return {
        "original":   face_np,
        "obfuscated": to_numpy(xa_obfs.cpu()),
        "protected":  prot_vis,
        "restored":   to_numpy(xa_rev.cpu()),
        "wrong":      to_numpy(xa_wrong.cpu()),
        # float [0,1] for metrics
        "orig_f":   norm01(xa),
        "obfs_f":   norm01(xa_obfs),
        "proc_f":   norm01(xa_proc),
        "rev_f":    norm01(xa_rev),
        "wrong_f":  norm01(xa_wrong),
    }


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="M2 end-to-end 검증")
    parser.add_argument("--img",      required=True,
                        help="입력 이미지 경로")
    parser.add_argument("--ckpt",
                        default="checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth",
                        help="INN 가중치 경로")
    parser.add_argument("--yolo",
                        default="C:/Users/HOSEO/Desktop/face_blur_stream/yolov8n-face.pt",
                        help="YOLO 얼굴 검출 가중치")
    parser.add_argument("--password", default=0,
                        help="비밀번호")
    parser.add_argument("--obf",      default="blur",
                        choices=["blur", "pixelate", "median", "mask", "hybridAll"])
    parser.add_argument("--out",      default=None,
                        help="결과 저장 폴더 (기본: 스크립트와 같은 디렉터리의 m2_output)")
    args = parser.parse_args()

    # 출력 폴더: 항상 스크립트 위치 기준 (실행 위치에 무관)
    if args.out is None:
        out_dir = Path(__file__).parent / "m2_output"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── (1) 모델 초기화 ───────────────────────────────────────────────
    print("\n[1] 모델 초기화")
    if not Path(args.ckpt).exists():
        sys.exit(f"[오류] INN 가중치 없음: {args.ckpt}")
    if not Path(args.yolo).exists():
        sys.exit(f"[오류] YOLO 가중치 없음: {args.yolo}")

    pipeline = SecureFaceRX(
        checkpoint_path=args.ckpt,
        obf_type=args.obf,
        detector_model=args.yolo,
    )
    print(f"    INN  : {args.ckpt}")
    print(f"    YOLO : {args.yolo}")
    print(f"    device: {pipeline.device}")

    # ── (2) 이미지 로드 + YOLO 검출 ──────────────────────────────────
    print(f"\n[2] 이미지 로드 + YOLO 검출")
    img = imread_safe(args.img)
    if img is None:
        sys.exit(f"[오류] 이미지 로드 실패: {args.img}")
    print(f"    이미지: {args.img}  {img.shape}")

    detections = pipeline.detector.detect(img)
    print(f"    검출된 얼굴: {len(detections)}개")
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox
        print(f"      face#{i}: bbox=[{x1},{y1},{x2},{y2}]  conf={d.conf:.3f}")

    if not detections:
        sys.exit("[오류] 얼굴 검출 실패. --yolo 경로 확인 필요.")

    # YOLO 검출 시각화
    vis_detect = img.copy()
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(vis_detect, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_detect, f"#{i} {d.conf:.2f}", (x1, max(y1-6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    imwrite_safe(str(out_dir / "01_yolo_detection.png"), vis_detect)
    print(f"    YOLO 검출 시각화 저장: {out_dir}/01_yolo_detection.png")

    # ── (3) 보호 파이프라인 ───────────────────────────────────────────
    print(f"\n[3] 보호 파이프라인 (password={args.password}, obf={args.obf})")
    psf_path = out_dir / "chacha.psf"
    protected_frame, psf_out = pipeline.protect_image(
        img, args.password, out_psf=psf_path
    )
    print(f"    PSF 저장: {psf_out}")
    imwrite_safe(str(out_dir / "02_protected_frame.png"), protected_frame)
    print(f"    보호 프레임 저장: {out_dir}/02_protected_frame.png")

    # ── (4) PSF 구조 확인 ─────────────────────────────────────────────
    print(f"\n[4] PSF 컨테이너 확인")
    psf_files = list(psf_out.iterdir())
    for f in sorted(psf_files):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:30s}  {size_kb:7.1f} KB")

    # 타일 파일 확인
    tile_files = [f for f in psf_files if "tile" in f.name]
    print(f"    타일 파일: {len(tile_files)}개 (얼굴 수와 일치: {len(tile_files)==len(detections)})")

    # ── (5) 복원 파이프라인 ───────────────────────────────────────────
    print(f"\n[5] 복원 파이프라인 (올바른 패스워드: {args.password})")
    restored_frame = pipeline.restore_image(psf_out, args.password)
    imwrite_safe(str(out_dir / "03_restored_frame.png"), restored_frame)
    print(f"    복원 프레임 저장: {out_dir}/03_restored_frame.png")

    # ── (6) 오복원 파이프라인 ─────────────────────────────────────────
    print(f"\n[6] 오복원 파이프라인 (틀린 패스워드: {str(args.password) + '_WRONG'})")
    wrong_frame = pipeline.restore_image(psf_out, str(args.password) + "_WRONG")
    imwrite_safe(str(out_dir / "04_wrong_frame.png"), wrong_frame)
    print(f"    오복원 프레임 저장: {out_dir}/04_wrong_frame.png")

    # ── (7) KPI 측정 (face 타일 단위) ────────────────────────────────
    print(f"\n[7] KPI 측정 (얼굴 타일 단위, M1 동일 기준)")
    print("=" * 60)

    from utils.container import load_psf
    _, manifest, face_tiles = load_psf(psf_out)

    all_pass   = True
    rps_psnrs, rps_ssims = [], []
    rec_psnrs, rec_ssims = [], []
    wrg_psnrs, wrg_ssims = [], []

    H, W = img.shape[:2]

    for face_meta in manifest.faces:
        fid = face_meta.id
        crop_box = face_meta.crop_box
        face_np, _ = crop_and_resize(img, crop_box, c.NORM_RESOLUTION)

        if fid not in face_tiles:
            print(f"  [경고] face#{fid} 타일 없음 — 건너뜀")
            continue

        protected_tile = face_tiles[fid]

        res = measure_kpi(pipeline, face_np, protected_tile, args.password)

        rp = psnr(res["proc_f"],  res["obfs_f"])   # RPS: y_hat vs y
        rs = ssim(res["proc_f"],  res["obfs_f"])
        cp = psnr(res["rev_f"],   res["orig_f"])   # Rec: x_rec vs x
        cs = ssim(res["rev_f"],   res["orig_f"])
        wp = psnr(res["wrong_f"], res["orig_f"])   # Wrg: x_wrong vs x
        ws = ssim(res["wrong_f"], res["orig_f"])

        rps_psnrs.append(rp); rps_ssims.append(rs)
        rec_psnrs.append(cp); rec_ssims.append(cs)
        wrg_psnrs.append(wp); wrg_ssims.append(ws)

        print(f"\n  [face#{fid}]  crop={crop_box}")
        print(f"    RPS   PSNR={rp:.2f}  SSIM={rs:.4f}")
        print(f"    Rec   PSNR={cp:.2f}  SSIM={cs:.4f}")
        print(f"    Wrg   PSNR={wp:.2f}  SSIM={ws:.4f}")

        # 개별 비교 패널 저장
        panel = make_panel({
            "1.Original":  res["original"],
            "2.Blur(y)":   res["obfuscated"],
            "3.Protected": res["protected"],
            "4.Restored":  res["restored"],
            "5.WrongKey":  res["wrong"],
        })
        imwrite_safe(str(out_dir / f"face{fid}_comparison.png"), panel)

    print("\n  ===== 평균 KPI =====")
    if rps_psnrs:
        rp_avg = np.mean(rps_psnrs); rs_avg = np.mean(rps_ssims)
        cp_avg = np.mean(rec_psnrs); cs_avg = np.mean(rec_ssims)
        wp_avg = np.mean(wrg_psnrs); ws_avg = np.mean(wrg_ssims)

        print("\n  [RPS] 보호본(y_hat) vs 블러(y)")
        all_pass &= check(rp_avg, ">=", 49.0, "PSNR(dB)")
        all_pass &= check(rs_avg, ">=",  0.99, "SSIM    ")

        print("\n  [복원] 복원본(x_rec) vs 원본(x)  (올바른 키)")
        all_pass &= check(cp_avg, ">=", 40.0, "PSNR(dB)")
        all_pass &= check(cs_avg, ">=",  0.97, "SSIM    ")

        print("\n  [오복원] 오복원(x_wrong) vs 원본(x)  (틀린 키)")
        all_pass &= check(wp_avg, "<",  11.0, "PSNR(dB)")
        all_pass &= check(ws_avg, "<",   0.20, "SSIM    ")

    print("=" * 60)

    # ── (8) 전체 프레임 비교 패널 ─────────────────────────────────────
    print(f"\n[8] 전체 프레임 비교 패널 저장")
    h_panel = max(img.shape[0], 600)
    w_panel = img.shape[1]
    labels = [
        ("Original",  img),
        ("Protected", protected_frame),
        ("Restored",  restored_frame),
        ("WrongKey",  wrong_frame),
    ]
    frame_panels = []
    for title, frame in labels:
        resized = cv2.resize(frame, (w_panel, h_panel))
        header = np.zeros((36, w_panel, 3), dtype=np.uint8)
        cv2.putText(header, title, (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        frame_panels.append(np.vstack([header, resized]))
    full_panel = np.hstack(frame_panels)
    imwrite_safe(str(out_dir / "05_full_frame_comparison.png"), full_panel)
    print(f"    저장: {out_dir}/05_full_frame_comparison.png")

    # ── 결과 요약 ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_pass:
        print("  M2 완료: YOLO 검출 + 파이프라인 KPI 모두 충족!")
    else:
        print("  일부 KPI 미충족. 위 결과 확인 필요.")
    print(f"\n  결과 저장 위치: {out_dir.resolve()}")
    print("  01_yolo_detection.png      — YOLO 검출 시각화")
    print("  02_protected_frame.png     — 전체 보호 프레임")
    print("  03_restored_frame.png      — 전체 복원 프레임")
    print("  04_wrong_frame.png         — 오복원 프레임")
    print("  05_full_frame_comparison.png — 4단계 비교")
    print("  face#_comparison.png       — 얼굴별 5단계 비교")
    print("  chacha.psf/                — PSF 컨테이너")


if __name__ == "__main__":
    main()
