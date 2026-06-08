"""
M1 검증 스크립트
가중치 파일을 받은 후 이 파일 하나만 실행하면 됩니다.

사용법:
    python m1_verify.py --ckpt checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth

성공 기준 (SRS KPI):
    RPS   : PSNR(ŷ, y) >= 49dB     SSIM >= 0.99
    복원  : PSNR(x̌, x) >= 40dB     SSIM >= 0.97
    오복원: PSNR(x⃛, x) <  11dB     SSIM <  0.20
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch
import cv2

import config as c


def imread_safe(path: str) -> np.ndarray | None:
    """
    한글 경로 + webp/jpg/png 모두 지원하는 안전한 이미지 로더.
    cv2.imread는 Windows 한글 경로에서 실패하므로 np.fromfile 우회 사용.
    webp는 OpenCV가 못 읽을 경우 Pillow로 폴백.
    """
    p = Path(path)
    if not p.exists():
        return None

    # 1차 시도: np.fromfile + imdecode (한글 경로 우회)
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # 2차 시도: Pillow (webp 완전 지원)
    try:
        from PIL import Image
        pil = Image.open(str(p)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[imread_safe] 실패: {e}")
        return None
from models.embedder import ModelDWT
from models.modules import DWT
from utils.key_gen import generate_key, make_key_rec
from utils.image_processing import Obfuscator, to_tensor, to_numpy


# ── 지표 계산 ──────────────────────────────────────────────────────

def psnr(a, b):
    """float [0,1] 배열 입력 (원본: data_range=1.0)"""
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-20 else 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-12))

def ssim(a, b):
    """float [0,1] 배열 입력 (원본: StructuralSimilarityIndexMeasure(data_range=1.0))"""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))

def check(val, op, thr, name):
    ops = {'>': val > thr, '>=': val >= thr, '<': val < thr}
    ok  = ops[op]
    mark = "[OK]" if ok else "[--]"
    print(f"    {mark}  {name} = {val:.2f}  (goal {op} {thr})")
    return ok


# ── 체크포인트 구조 분석 ───────────────────────────────────────────

def inspect_checkpoint(path):
    print(f"\n[1] 체크포인트 분석: {path}")
    raw = torch.load(path, map_location='cpu')
    if isinstance(raw, dict):
        # state_dict 직접 저장인지 wrapper 인지 확인
        sample_keys = list(raw.keys())[:5]
        print(f"    키 수: {len(raw)}")
        print(f"    키 예시: {sample_keys}")
        has_module = any(k.startswith("module.") for k in raw)
        has_wrapper = any(k in ("state_dict", "model", "epoch") for k in raw)
        print(f"    DataParallel 래퍼: {has_module}")
        print(f"    dict 래퍼: {has_wrapper}")
    return raw


def load_weights(embedder, path):
    """공식 저장 방식: torch.save(embedder.state_dict(), path)"""
    raw = torch.load(path, map_location='cpu')

    # wrapper 제거
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]

    # DataParallel module. prefix 제거
    if isinstance(raw, dict) and all(k.startswith("module.") for k in raw):
        raw = {k[len("module."):]: v for k, v in raw.items()}

    missing, unexpected = embedder.load_state_dict(raw, strict=False)
    if missing:
        print(f"    [경고] missing  ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"    [경고] unexpected ({len(unexpected)}): {unexpected[:3]}")
    if not missing and not unexpected:
        print("    [OK] 가중치 완전 일치")
    return missing, unexpected


# ── 단일 이미지 KPI 측정 ──────────────────────────────────────────

def run_kpi(embedder, dwt, img_bgr, password, obf_type='blur'):
    obf   = Obfuscator(obf_type=obf_type)
    device = next(embedder.parameters()).device

    # 256x256 리사이즈 (INTER_AREA: 다운샘플링 최적, 원본은 PIL BICUBIC)
    img = cv2.resize(img_bgr, (c.NORM_RESOLUTION, c.NORM_RESOLUTION),
                     interpolation=cv2.INTER_AREA)
    xa  = to_tensor(img, device=device)          # (1,3,256,256)

    # 사전 난독화
    xa_obfs = obf(xa)

    # 키 생성 (원본: password=0)
    skey     = generate_key(password, bs=1,
                             w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
    skey_dwt = dwt(skey.float())

    with torch.no_grad():
        # 보호
        xa_out_z, xa_proc = embedder(xa, xa_obfs, skey_dwt)
        del xa_out_z

        # 정상 복원 (올바른 키)
        key_rec = make_key_rec(skey_dwt)
        xa_rev, _ = embedder(key_rec, xa_proc, skey_dwt, rev=True)

        # 오복원 (틀린 키)
        skey_w     = generate_key(str(password) + "_WRONG", bs=1,
                                   w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION).to(device)
        skey_dwt_w = dwt(skey_w.float())
        key_rec_w  = make_key_rec(skey_dwt_w)
        xa_wrong, _ = embedder(key_rec_w, xa_proc, skey_dwt_w, rev=True)

    # ── float [0,1] 변환 (원본과 동일: normalize(x) = (x+1)/2) ───
    def norm01(t):
        """[-1,1] tensor → float64 HWC RGB [0,1]"""
        return ((t.squeeze(0).permute(1,2,0).clamp(-1.,1.).cpu().float().numpy()
                 + 1.0) / 2.0).astype(np.float64)

    # ── uint8 BGR 변환 (이미지 저장용) ─────────────────────────────
    x_np      = to_numpy(xa.cpu())
    y_np      = to_numpy(xa_obfs.cpu())
    yhat_np   = to_numpy(xa_proc.cpu())
    xrec_np   = to_numpy(xa_rev.cpu())
    xwrong_np = to_numpy(xa_wrong.cpu())

    return {
        # uint8 BGR — 저장 전용
        "x": x_np, "y": y_np, "yhat": yhat_np,
        "xrec": xrec_np, "xwrong": xwrong_np,
        # float [0,1] RGB — 지표 계산 (원본 방식, 양자화 오차 없음)
        "x_f":     norm01(xa),
        "y_f":     norm01(xa_obfs),
        "yhat_f":  norm01(xa_proc),
        "xrec_f":  norm01(xa_rev),
        "xwrong_f":norm01(xa_wrong),
    }


# ── 메인 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="M1 KPI 검증")
    parser.add_argument("--ckpt",     required=True, help=".pth 파일 경로")
    parser.add_argument("--img",      default=None,  help="테스트 이미지 (없으면 랜덤)")
    parser.add_argument("--password", default=0,     help="비밀번호 (원본: 0)")
    parser.add_argument("--obf",      default="blur",
                        choices=["blur","pixelate","median","mask"])
    parser.add_argument("--out-dir",  default="m1_output", help="결과 이미지 저장 폴더")
    parser.add_argument("--n-images", type=int, default=1, help="여러 이미지 평균 (폴더 입력 시)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"[오류] 가중치 파일 없음: {ckpt_path}\n"
                 "  BaiduDisk https://pan.baidu.com/s/1q-s1G4aqSzcXEofDOEfeHg\n"
                 "  비밀번호: 3cvh  에서 다운로드 후 checkpoints/ 에 저장하세요.")

    # ── 모델 로드 ──────────────────────────────────────────────────
    inspect_checkpoint(args.ckpt)

    print(f"\n[2] 모델 로드 (n_blocks={c.INV_BLOCKS})")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = ModelDWT(n_blocks=c.INV_BLOCKS).to(device).eval()
    dwt_mod  = DWT().to(device)
    load_weights(embedder, args.ckpt)

    # ── 테스트 이미지 준비 ─────────────────────────────────────────
    if args.img and Path(args.img).is_file():
        img = imread_safe(args.img)
        imgs = [img] if img is not None else []
        if not imgs:
            sys.exit(f"[오류] 이미지 로드 실패: {args.img}")
        print(f"\n[3] 테스트 이미지: {args.img}  {imgs[0].shape}")
    elif args.img and Path(args.img).is_dir():
        exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
        paths = [p for p in Path(args.img).rglob("*") if p.suffix.lower() in exts]
        paths = paths[:args.n_images]
        imgs  = [imread_safe(str(p)) for p in paths]
        imgs  = [i for i in imgs if i is not None]
        print(f"\n[3] 테스트 이미지 {len(imgs)}장: {args.img}")
    else:
        print("\n[3] 테스트 이미지 없음 -> 랜덤 256x256 사용 (KPI 참고용)")
        imgs = [np.random.randint(0, 256, (256,256,3), dtype=np.uint8)]

    # ── KPI 측정 ──────────────────────────────────────────────────
    print(f"\n[4] KPI 측정 (obf={args.obf}, password={args.password})")
    print("=" * 55)

    rps_psnrs, rps_ssims = [], []
    rec_psnrs, rec_ssims = [], []
    wrg_psnrs, wrg_ssims = [], []

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else Path(__file__).parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        if img is None:
            continue
        res = run_kpi(embedder, dwt_mod, img, args.password, args.obf)

        # 지표: float [0,1] 사용 (원본 코드와 동일한 방식)
        rp = psnr(res["yhat_f"],  res["y_f"])
        rs = ssim(res["yhat_f"],  res["y_f"])
        cp = psnr(res["xrec_f"],  res["x_f"])
        cs = ssim(res["xrec_f"],  res["x_f"])
        wp = psnr(res["xwrong_f"],res["x_f"])
        ws = ssim(res["xwrong_f"],res["x_f"])

        rps_psnrs.append(rp); rps_ssims.append(rs)
        rec_psnrs.append(cp); rec_ssims.append(cs)
        wrg_psnrs.append(wp); wrg_ssims.append(ws)

        # 첫 번째 이미지만 저장
        if i == 0:
            cv2.imwrite(str(out_dir/"01_original.png"),  res["x"])
            cv2.imwrite(str(out_dir/"02_obfuscated.png"),res["y"])
            cv2.imwrite(str(out_dir/"03_protected.png"), res["yhat"])
            cv2.imwrite(str(out_dir/"04_restored.png"),  res["xrec"])
            cv2.imwrite(str(out_dir/"05_wrong.png"),     res["xwrong"])

    # ── 결과 출력 ─────────────────────────────────────────────────
    rp_avg = np.mean(rps_psnrs); rs_avg = np.mean(rps_ssims)
    cp_avg = np.mean(rec_psnrs); cs_avg = np.mean(rec_ssims)
    wp_avg = np.mean(wrg_psnrs); ws_avg = np.mean(wrg_ssims)

    all_pass = True
    print("\n  [RPS] 보호본(y_hat) vs 난독화(y)  (시각적 유사도)")
    all_pass &= check(rp_avg, ">=", 49.0, "PSNR(dB)")
    all_pass &= check(rs_avg, ">=",  0.99, "SSIM    ")

    print("\n  [복원] 복원본(x_rec) vs 원본(x)   (올바른 키)")
    all_pass &= check(cp_avg, ">=", 40.0, "PSNR(dB)")
    all_pass &= check(cs_avg, ">=",  0.97, "SSIM    ")

    print("\n  [오복원] 오복원(x_wrong) vs 원본(x)  (틀린 키)")
    all_pass &= check(wp_avg, "<",  11.0, "PSNR(dB)")
    all_pass &= check(ws_avg, "<",   0.20, "SSIM    ")

    print("=" * 55)
    if all_pass:
        print("  M1 완료: 모든 KPI 충족!")
    else:
        print("  일부 KPI 미충족. (이미지 해상도/난독화 강도 확인 필요)")

    print(f"\n  결과 이미지 저장: {out_dir.resolve()}")
    print(f"  이미지 수: {len(imgs)}장 평균")


if __name__ == "__main__":
    main()
