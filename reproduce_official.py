"""
M1 재현 스크립트 — 원본 test_tcsvt.py 의 보호/복원 흐름을 재현
원본 호출 패턴을 그대로 사용.

사용:
    python reproduce_official.py --ckpt checkpoints/securefacerx.pth --img sample.jpg

공식 KPI (SRS §8):
    RPS   : PSNR(ŷ, y) ≳ 49dB, SSIM ≳ 0.99
    복원  : PSNR(x̌, x) > 40dB, SSIM > 0.97
    오복원: PSNR(x⃛, x) < 11dB
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import config as c
from models.embedder import ModelDWT, init_model
from models.modules import DWT
from utils.key_gen import generate_key, make_key_rec
from utils.image_processing import Obfuscator, to_tensor, to_numpy


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8))


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(a, b, channel_axis=-1, data_range=255))
    except ImportError:
        return -1.0


def gauss_noise(shape, device) -> torch.Tensor:
    """원본 코드의 gauss_noise — SECRET_KEY_AS_NOISE=False 일 때 사용."""
    return torch.zeros(shape, device=device).normal_(mean=0, std=0.1)


def reproduce(args):
    device = c.DEVICE
    print(f"device: {device}")

    # ── 모델 로드 ───────────────────────────────────────────────
    embedder = ModelDWT(n_blocks=c.INV_BLOCKS).to(device)
    dwt      = DWT().to(device)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        if all(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        embedder.load_state_dict(state, strict=False)
        print(f"가중치 로드: {args.ckpt}")
    else:
        init_model(embedder, device)
        print("경고: 가중치 없음 - 랜덤 초기화 (KPI 미충족 예상)")

    embedder.eval()
    obf = Obfuscator(obf_type=args.obf)

    # ── 입력 이미지 ─────────────────────────────────────────────
    img = cv2.imread(args.img) if args.img else None
    if img is None:
        print("테스트용 랜덤 이미지 사용 (256×256)")
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    img = cv2.resize(img, (c.NORM_RESOLUTION, c.NORM_RESOLUTION))
    xa  = to_tensor(img, device=device)           # (1,3,256,256)

    # ── 사전 난독화 ─────────────────────────────────────────────
    xa_obfs = obf(xa)
    print(f"사전 난독화: {args.obf}")

    # ── KeyGen (원본 패턴: password=0) ──────────────────────────
    _bs, _c, _w, _h = 1, c.channels_in, c.NORM_RESOLUTION, c.NORM_RESOLUTION
    password1 = args.password  # 올바른 키
    skey1     = generate_key(password1, _bs, _w, _h).to(device)
    skey1_dwt = dwt(skey1.float())

    # ── 보호 ────────────────────────────────────────────────────
    with torch.no_grad():
        xa_out_z, xa_proc = embedder(xa, xa_obfs, skey1_dwt)
        del xa_out_z       # 부산물 즉시 폐기

    ya_np = to_numpy(xa_proc.cpu())
    y_np  = to_numpy(xa_obfs.cpu())
    x_np  = to_numpy(xa.cpu())

    rps_psnr = compute_psnr(ya_np, y_np)
    rps_ssim = compute_ssim(ya_np, y_np)
    print(f"\n[RPS] ŷ ↔ y    PSNR={rps_psnr:.2f}dB  SSIM={rps_ssim:.4f}")
    print(f"  목표: PSNR≳49dB, SSIM≳0.99  {'✓' if rps_psnr>=49 else '✗'}")

    # ── 정상 복원 (올바른 키) ────────────────────────────────────
    with torch.no_grad():
        if c.SECRET_KEY_AS_NOISE:
            key_rec = make_key_rec(skey1_dwt)     # skey1_dwt.repeat(1,3,1,1)
        else:
            key_rec = gauss_noise((_bs, _c*4, _w//2, _h//2), device)
        xa_rev, _ = embedder(key_rec, xa_proc, skey1_dwt, rev=True)

    x_rec_np = to_numpy(xa_rev.cpu())
    rec_psnr = compute_psnr(x_rec_np, x_np)
    rec_ssim = compute_ssim(x_rec_np, x_np)
    print(f"\n[정상 복원] x̌ ↔ x  PSNR={rec_psnr:.2f}dB  SSIM={rec_ssim:.4f}")
    print(f"  목표: PSNR>40dB, SSIM>0.97  {'✓' if rec_psnr>40 else '✗'}")

    # ── 오복원 (틀린 키) ─────────────────────────────────────────
    password2 = str(password1) + "_WRONG"
    skey2     = generate_key(password2, _bs, _w, _h).to(device)
    skey2_dwt = dwt(skey2.float())
    with torch.no_grad():
        if c.SECRET_KEY_AS_NOISE:
            key_rec_w = make_key_rec(skey2_dwt)
        else:
            key_rec_w = gauss_noise((_bs, _c*4, _w//2, _h//2), device)
        xa_rev_wrong, _ = embedder(key_rec_w, xa_proc, skey2_dwt, rev=True)

    x_wrong_np = to_numpy(xa_rev_wrong.cpu())
    wrong_psnr = compute_psnr(x_wrong_np, x_np)
    wrong_ssim = compute_ssim(x_wrong_np, x_np)
    print(f"\n[오복원]   x⃛ ↔ x  PSNR={wrong_psnr:.2f}dB  SSIM={wrong_ssim:.4f}")
    print(f"  목표: PSNR<11dB, SSIM<0.2    {'✓' if wrong_psnr<11 else '✗'}")

    # ── 결과 저장 ─────────────────────────────────────────────────
    if args.out_dir:
        od = Path(args.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(od/"01_original_x.png"),       x_np)
        cv2.imwrite(str(od/"02_obfuscated_y.png"),     y_np)
        cv2.imwrite(str(od/"03_protected_y_hat.png"),  ya_np)
        cv2.imwrite(str(od/"04_restored_correct.png"), x_rec_np)
        cv2.imwrite(str(od/"05_wrong_recovery.png"),   x_wrong_np)
        print(f"\n결과 저장: {od}")

    return {
        "rps_psnr": rps_psnr, "rps_ssim": rps_ssim,
        "rec_psnr": rec_psnr, "rec_ssim": rec_ssim,
        "wrong_psnr": wrong_psnr, "wrong_ssim": wrong_ssim,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=None)
    parser.add_argument("--img",      default=None)
    parser.add_argument("--password", default=0)         # 원본: int 0
    parser.add_argument("--obf",      default="blur",
                        choices=["blur","pixelate","median","mask"])
    parser.add_argument("--out-dir",  default="reproduce_output")
    args = parser.parse_args()
    reproduce(args)
