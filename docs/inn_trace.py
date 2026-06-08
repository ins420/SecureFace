"""
INN 픽셀 수준 추적 스크립트
실제 얼굴 이미지에서 2×2 패치를 여러 개 골라
DWT → Affine Coupling → IWT 각 단계의 실제 수치를 출력한다.

사용법:
    python docs/inn_trace.py --img Z:/캡스톤디자인/chacha.jpg
"""
import argparse, sys
from pathlib import Path
import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as c
from models.embedder import ModelDWT
from models.modules import DWT, IWT
from utils.image_processing import Obfuscator, to_tensor, to_numpy
from utils.key_gen import generate_key
from detection.yolo_detector import expand_bbox_square, crop_and_resize

CKPT = str(Path(__file__).parent.parent /
           "checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth")
YOLO = "C:/Users/HOSEO/Desktop/face_blur_stream/yolov8n-face.pt"


def haar_dwt_2x2(patch):
    """2×2 패치 1채널 → LL, LH, HL, HH (하르 웨이블릿)"""
    a, b, c_, d = float(patch[0,0]), float(patch[0,1]), float(patch[1,0]), float(patch[1,1])
    LL = (a + b + c_ + d) / 2
    LH = (a + b - c_ - d) / 2
    HL = (a - b + c_ - d) / 2
    HH = (a - b - c_ + d) / 2
    return LL, LH, HL, HH


def haar_iwt_2x2(LL, LH, HL, HH):
    """LL, LH, HL, HH → 2×2 패치 (역변환)"""
    a = (LL + LH + HL + HH) / 2
    b = (LL + LH - HL - HH) / 2
    c_ = (LL - LH + HL - HH) / 2
    d = (LL - LH - HL + HH) / 2
    return a, b, c_, d


def print_section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def show_patch_example(label, patch_orig, patch_blur, patch_prot, patch_rest):
    """2×2 패치(1채널) 하나에 대해 전체 흐름 출력"""
    print(f"\n  ── {label} ──")

    # 픽셀값 출력
    print(f"  원본 픽셀:  {int(patch_orig[0,0]):3d} {int(patch_orig[0,1]):3d}  │  "
          f"블러 픽셀:  {int(patch_blur[0,0]):3d} {int(patch_blur[0,1]):3d}  │  "
          f"보호 픽셀:  {int(patch_prot[0,0]):3d} {int(patch_prot[0,1]):3d}  │  "
          f"복원 픽셀:  {int(patch_rest[0,0]):3d} {int(patch_rest[0,1]):3d}")
    print(f"             {int(patch_orig[1,0]):3d} {int(patch_orig[1,1]):3d}  │            "
          f"{int(patch_blur[1,0]):3d} {int(patch_blur[1,1]):3d}  │            "
          f"{int(patch_prot[1,0]):3d} {int(patch_prot[1,1]):3d}  │            "
          f"{int(patch_rest[1,0]):3d} {int(patch_rest[1,1]):3d}")

    # DWT 계수
    LL_x, LH_x, HL_x, HH_x = haar_dwt_2x2(patch_orig)
    LL_y, LH_y, HL_y, HH_y = haar_dwt_2x2(patch_blur)
    LL_p, LH_p, HL_p, HH_p = haar_dwt_2x2(patch_prot)
    LL_r, LH_r, HL_r, HH_r = haar_dwt_2x2(patch_rest)

    print(f"\n  DWT 계수:       원본 x1       블러 x2      보호본 y1      복원 x1_rec")
    print(f"  LL(평균)     {LL_x:9.2f}    {LL_y:9.2f}    {LL_p:9.2f}    {LL_r:9.2f}")
    print(f"  LH(수평엣지) {LH_x:9.2f}    {LH_y:9.2f}    {LH_p:9.2f}    {LH_r:9.2f}")
    print(f"  HL(수직엣지) {HL_x:9.2f}    {HL_y:9.2f}    {HL_p:9.2f}    {HL_r:9.2f}")
    print(f"  HH(대각엣지) {HH_x:9.2f}    {HH_y:9.2f}    {HH_p:9.2f}    {HH_r:9.2f}")

    # 보호 차이 (y1 - x2): 숨겨진 신호
    print(f"\n  보호본 - 블러 (숨겨진 미세 신호):")
    print(f"  ΔLL={LL_p-LL_y:+.3f}  ΔLH={LH_p-LH_y:+.3f}  "
          f"ΔHL={HL_p-HL_y:+.3f}  ΔHH={HH_p-HH_y:+.3f}")

    # t 추정값 (y1 ≈ x2 이므로 t ≈ x2 - x1)
    print(f"\n  역산으로 추정되는 이동량 t = y1 - x1 (exp(s)≈1 가정):")
    print(f"  t_LL={LL_p-LL_x:+.2f}  t_LH={LH_p-LH_x:+.2f}  "
          f"t_HL={HL_p-HL_x:+.2f}  t_HH={HH_p-HH_x:+.2f}")

    # 복원 정확도
    err_orig = abs(patch_rest.astype(float) - patch_orig.astype(float)).mean()
    err_blur = abs(patch_prot.astype(float) - patch_blur.astype(float)).mean()
    print(f"\n  복원 오차 |x1_rec - x1| 평균: {err_orig:.3f} px")
    print(f"  보호 위장 |y1   - x2 | 평균: {err_blur:.3f} px  (사람 눈에 안 보임)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--yolo", default=YOLO)
    parser.add_argument("--password", default=0)
    args = parser.parse_args()

    # 모델 로드
    device = torch.device("cpu")
    embedder = ModelDWT(n_blocks=c.INV_BLOCKS).to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    embedder.load_state_dict(state, strict=False)
    dwt_mod = DWT().to(device)
    obf = Obfuscator(obf_type="blur")

    # 이미지 로드 + 얼굴 크롭
    buf = np.fromfile(args.img, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    from ultralytics import YOLO as UltraYOLO
    yolo = UltraYOLO(args.yolo)
    results = yolo(img, verbose=False, conf=0.25)[0]
    box = results.boxes[0].xyxy.cpu().numpy()[0].astype(int)
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    crop_box = expand_bbox_square([x1,y1,x2,y2], H, W)
    face_np, _ = crop_and_resize(img, crop_box, 256)

    # INN 실행
    xa     = to_tensor(face_np, device=device)
    xa_obf = obf(xa)
    skey   = generate_key(args.password, bs=1, w=256, h=256).to(device)
    skey_dwt = dwt_mod(skey.float())

    with torch.no_grad():
        _, xa_proc = embedder(xa, xa_obf, skey_dwt)
        from utils.key_gen import make_key_rec
        key_rec = make_key_rec(skey_dwt)
        xa_rev, _ = embedder(key_rec, xa_proc, skey_dwt, rev=True)

    # uint8 변환
    orig = to_numpy(xa.cpu())       # 256×256 BGR
    blur = to_numpy(xa_obf.cpu())
    prot = to_numpy(xa_proc.cpu())
    rest = to_numpy(xa_rev.cpu())

    # 채널 평균 (그레이스케일로 단순화)
    orig_g = orig.mean(axis=2).astype(np.uint8)
    blur_g = blur.mean(axis=2).astype(np.uint8)
    prot_g = prot.mean(axis=2).astype(np.uint8)
    rest_g = rest.mean(axis=2).astype(np.uint8)

    print_section("INN 픽셀 수준 추적 — 실제 얼굴 이미지")
    print(f"  이미지: {args.img}")
    print(f"  얼굴 크롭: {crop_box}  →  256×256 정규화")

    # ── 예시 1: 밝은 영역 (이마/뺨) ──
    # 픽셀 밝기 기준으로 패치 선택
    regions = []
    for name, row_range, col_range in [
        ("밝은 영역 (이마)",     ( 20, 80),  ( 60,140)),
        ("중간 영역 (눈 주변)", (100,140),  ( 80,160)),
        ("어두운 영역 (눈동자)",(110,130),  (115,145)),
        ("엣지 영역 (윤곽선)", ( 60,100),  ( 20, 60)),
        ("배경과 경계",         ( 10, 50),  (200,240)),
    ]:
        r0,r1 = row_range
        c0,c1 = col_range
        patch_o = orig_g[r0:r0+2, c0:c0+2]
        patch_b = blur_g[r0:r0+2, c0:c0+2]
        patch_p = prot_g[r0:r0+2, c0:c0+2]
        patch_r = rest_g[r0:r0+2, c0:c0+2]
        regions.append((name, patch_o, patch_b, patch_p, patch_r))

    print_section("각 영역별 픽셀 추적 결과")
    for name, po, pb, pp, pr in regions:
        show_patch_example(name, po, pb, pp, pr)

    # ── 통계 요약 ──
    print_section("전체 256×256 이미지 통계")
    diff_prot_blur = np.abs(prot.astype(float) - blur.astype(float))
    diff_rest_orig = np.abs(rest.astype(float) - orig.astype(float))
    diff_blur_orig = np.abs(blur.astype(float) - orig.astype(float))

    print(f"\n  블러 - 원본     차이: 평균 {diff_blur_orig.mean():.2f} / max {diff_blur_orig.max():.0f}  (블러가 얼마나 바꿨나)")
    print(f"  보호본 - 블러  차이: 평균 {diff_prot_blur.mean():.2f} / max {diff_prot_blur.max():.0f}  (숨겨진 신호 강도)")
    print(f"  복원 - 원본     차이: 평균 {diff_rest_orig.mean():.2f} / max {diff_rest_orig.max():.0f}  (복원 오차)")

    print(f"\n  → 보호본은 블러와 평균 {diff_prot_blur.mean():.2f}px 차이")
    print(f"     사람 눈 감지 한계 ≈ 3-5px 차이")
    print(f"     {diff_prot_blur.mean():.2f}px 는 육안으로 절대 구분 불가")


if __name__ == "__main__":
    main()
