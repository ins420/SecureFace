"""
SecureFace-RX 단위/통합 테스트
가중치 없이 랜덤 초기화로 shape·흐름·수식을 검증.

실행: python test_pipeline.py
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import cv2


# ── 공통 헬퍼 ─────────────────────────────────────────────────────

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8))


def _face_frame():
    """검출기 없이 테스트용 더미 프레임."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


# ── Unit: DWT/IWT 원본 수식 역연산 ────────────────────────────────

def test_dwt_iwt():
    from models.modules import DWT, IWT
    dwt, iwt = DWT(), IWT()
    x = torch.rand(2, 3, 256, 256)
    # DWT → IWT 는 완전한 역연산
    y = iwt(dwt(x))
    err = (x - y).abs().max().item()
    assert err < 1e-5, f"DWT·IWT 역연산 오차: {err:.2e}"
    print(f"PASS  DWT·IWT 역연산 (max_err={err:.2e})")


# ── Unit: DWT 서브밴드 수 확인 ─────────────────────────────────────

def test_dwt_shape():
    from models.modules import DWT
    dwt = DWT()
    x = torch.rand(1, 3, 256, 256)
    y = dwt(x)
    assert y.shape == (1, 12, 128, 128), f"DWT shape 오류: {y.shape}"
    print(f"PASS  DWT shape (1,3,256,256) → (1,12,128,128)")


# ── Unit: KeyGen 결정론적 / avalanche ─────────────────────────────

def test_keygen_deterministic():
    from utils.key_gen import generate_key
    k1 = generate_key("pw123", 1)
    k2 = generate_key("pw123", 1)
    assert torch.equal(k1, k2), "동일 비밀번호 → 다른 키"
    print("PASS  KeyGen 결정론적")


def test_keygen_shape():
    from utils.key_gen import generate_key
    k = generate_key(0, bs=2, w=256, h=256)   # 원본처럼 int password
    assert k.shape == (2, 1, 256, 256), f"KeyGen shape 오류: {k.shape}"
    print(f"PASS  KeyGen shape {k.shape}")


def test_keygen_avalanche():
    from utils.key_gen import generate_key
    k1 = generate_key("pw123",    1).float()
    k2 = generate_key("pw123X",   1).float()
    ratio = (k1 != k2).float().mean().item()
    assert ratio > 0.4, f"avalanche 미충족: {ratio:.1%}"
    print(f"PASS  KeyGen avalanche ({ratio:.1%} bits differ)")


# ── Unit: e(s) 범위 ────────────────────────────────────────────────

def test_affine_scale_range():
    from models.invblock import INV_block_affine
    import config as c
    blk = INV_block_affine()
    s   = torch.linspace(-10, 10, 1000)
    es  = blk.e(s)
    lo, hi = es.min().item(), es.max().item()
    assert lo > 0.1,     f"e(s) 하한 너무 낮음: {lo:.4f}"
    assert hi < 10.0,    f"e(s) 상한 너무 높음: {hi:.4f}"
    print(f"PASS  e(s) 범위 [{lo:.4f}, {hi:.4f}] (clamp={c.clamp})")


# ── Unit: INV_block_affine forward·inverse 일관성 ─────────────────

def test_invblock_consistency():
    from models.invblock import INV_block_affine
    blk = INV_block_affine().eval()
    x   = torch.rand(1, 24, 128, 128)
    key = torch.rand(1, 4,  128, 128)
    with torch.no_grad():
        y  = blk(x,   key, rev=False)
        x2 = blk(y,   key, rev=True)
    err = (x - x2).abs().max().item()
    # 랜덤 초기화라도 역연산 구조는 성립해야 함
    assert err < 0.5, f"INV_block forward·inverse 오차 큼: {err:.4f}"
    print(f"PASS  INV_block_affine 역연산 (max_err={err:.4f})")


# ── Unit: ModelDWT 채널 순서 / shape 확인 ─────────────────────────

def test_embedder_shapes():
    from models.embedder import ModelDWT
    from models.modules import DWT
    from utils.key_gen import generate_key, make_key_rec
    import config as c

    device = torch.device('cpu')
    model  = ModelDWT(n_blocks=c.INV_BLOCKS).eval()
    dwt    = DWT()

    xa      = torch.rand(1, 3, 256, 256)
    xa_obfs = torch.rand(1, 3, 256, 256)
    skey    = generate_key(0, bs=1).float()
    skey_dwt = dwt(skey)               # (1,4,128,128)

    with torch.no_grad():
        # 보호: (xa, xa_obfs, skey_dwt) → (z, steg_img)
        z, steg = model(xa, xa_obfs, skey_dwt, rev=False)
        assert z.shape    == (1, 12, 128, 128), f"z shape 오류: {z.shape}"
        assert steg.shape == (1,  3, 256, 256), f"steg shape 오류: {steg.shape}"

        # 복원
        key_rec = make_key_rec(skey_dwt)          # (1,12,128,128)
        rec, _ = model(key_rec, steg, skey_dwt, rev=True)
        assert rec.shape == (1, 3, 256, 256), f"rec shape 오류: {rec.shape}"

    print("PASS  ModelDWT shape 검증 (z:12ch, steg:3ch, rec:3ch)")


# ── Integration: PSF 저장/로드 무결성 ─────────────────────────────

def test_psf_integrity():
    from utils.container import save_psf, load_psf, FaceMeta
    frame = _face_frame()
    faces = [FaceMeta(id=0, bbox=[10,10,100,100],
                      crop_box=[5,5,110,110], scale=2.5)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.psf"
        save_psf(frame, faces, p)
        loaded, mf = load_psf(p)
    assert np.array_equal(frame, loaded), "PSF 이미지 불일치"
    assert mf.secrets_stored is False
    print("PASS  PSF 저장·로드 무결성")


def test_psf_tamper_detection():
    from utils.container import save_psf, load_psf, FaceMeta
    frame = _face_frame()
    faces = [FaceMeta(id=0, bbox=[10,10,100,100],
                      crop_box=[5,5,110,110], scale=2.0)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.psf"
        save_psf(frame, faces, p)
        img = (p / "protected.png")
        img.write_bytes(img.read_bytes()[:-50] + b'\x00'*50)
        try:
            load_psf(p)
            assert False, "변조 감지 실패"
        except RuntimeError as e:
            print(f"PASS  변조 감지 RuntimeError ('{str(e)[:40]}...')")


# ── 실행 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== SecureFace-RX 테스트 ===\n")
    tests = [
        test_dwt_iwt,
        test_dwt_shape,
        test_keygen_deterministic,
        test_keygen_shape,
        test_keygen_avalanche,
        test_affine_scale_range,
        test_invblock_consistency,
        test_embedder_shapes,
        test_psf_integrity,
        test_psf_tamper_detection,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n결과: {passed}/{len(tests)} 통과")
