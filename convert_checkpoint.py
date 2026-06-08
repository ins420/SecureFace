"""
공식 ProFace 체크포인트를 SecureFace-RX 형식으로 변환하는 유틸 (M1 지원)

공식 가중치: hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth
다운로드:   https://github.com/lixionga/ProFace → FacePrivacy/PRO-Face S 리포지터리

사용:
    python convert_checkpoint.py --src official.pth --dst checkpoints/securefacerx.pth

공식 구현의 키 이름 규칙:
    inv_blocks.0.r.conv1.weight   (DataParallel 없는 경우)
    module.inv_blocks.0.r.conv1.weight (DataParallel 래핑 시)
SecureFace-RX 구조와 직접 호환되도록 prefix만 조정.
"""

import argparse
import torch
from collections import OrderedDict


OFFICIAL_KEY_MAP = {
    # 공식 ProFace S 키 → SecureFace-RX 키 (구조가 같으면 이름도 같음)
    # 만약 이름이 다르다면 여기에 매핑 추가
    # "official.key": "our.key",
}


def convert(src: str, dst: str, verbose: bool = True):
    state = torch.load(src, map_location="cpu")

    # state_dict 추출
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    # DataParallel module. prefix 제거
    new_state = OrderedDict()
    for k, v in state.items():
        nk = k.lstrip("module.")
        # 명시적 이름 매핑 적용
        nk = OFFICIAL_KEY_MAP.get(nk, nk)
        new_state[nk] = v
        if verbose:
            print(f"  {k} → {nk}  {tuple(v.shape)}")

    torch.save(new_state, dst)
    print(f"\n변환 완료: {dst}  (총 {len(new_state)}개 파라미터 텐서)")

    # 호환성 확인
    _verify(new_state)


def _verify(state: dict):
    from models.embedder import ModelDWT
    import config
    model = ModelDWT(n_blocks=config.INV_BLOCKS)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[경고] missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[경고] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    if not missing and not unexpected:
        print("[호환성] 완전 일치. 가중치를 그대로 사용할 수 있습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="공식 .pth 파일")
    parser.add_argument("--dst", required=True, help="변환된 출력 .pth")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    convert(args.src, args.dst, args.verbose)
