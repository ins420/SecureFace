"""
SecureFace-RX CLI
사용법:
    python cli.py protect  <input>  --password <pw> [--out output.psf] [--obf blur]
    python cli.py restore  <psf>    --password <pw> [--out restored.png]
    python cli.py protect-video <video> --password <pw> [--out-dir frames/]
    python cli.py restore-video <dir>   --password <pw> [--out restored.mp4]
"""

import argparse
import sys
from pathlib import Path

import cv2


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="SecureFace-RX",
        description="YOLO 검출 + PRO-Face S 가역 익명화",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── protect ──────────────────────────────────────────────────
    p_prot = sub.add_parser("protect", help="이미지 보호")
    p_prot.add_argument("input",  type=str, help="입력 이미지 경로")
    p_prot.add_argument("--password", "-p", required=True, help="비밀번호")
    p_prot.add_argument("--out",   "-o", default=None,   help="출력 PSF 경로 (기본: <input>.psf)")
    p_prot.add_argument("--ckpt",        default=None,   help="모델 가중치 파일")
    p_prot.add_argument("--obf",         default="blur", choices=["blur","pixelate","median","mask","hybridAll"])
    p_prot.add_argument("--device",      default=None,   help="cuda / cpu")
    p_prot.add_argument("--zip",  action="store_true",   help="PSF를 zip으로 압축")

    # ── restore ──────────────────────────────────────────────────
    p_rest = sub.add_parser("restore", help="PSF 복원")
    p_rest.add_argument("psf",    type=str, help="PSF 경로")
    p_rest.add_argument("--password", "-p", required=True, help="비밀번호")
    p_rest.add_argument("--out",   "-o", default=None,   help="출력 이미지 경로")
    p_rest.add_argument("--ckpt",        default=None,   help="모델 가중치 파일")
    p_rest.add_argument("--device",      default=None)

    # ── protect-video ─────────────────────────────────────────────
    p_pv = sub.add_parser("protect-video", help="영상 보호")
    p_pv.add_argument("input",    type=str)
    p_pv.add_argument("--password", "-p", required=True)
    p_pv.add_argument("--out-dir",       default="protected_frames")
    p_pv.add_argument("--ckpt",          default=None)
    p_pv.add_argument("--obf",           default="blur")
    p_pv.add_argument("--device",        default=None)

    # ── restore-video ─────────────────────────────────────────────
    p_rv = sub.add_parser("restore-video", help="영상 복원")
    p_rv.add_argument("psf_dir",  type=str)
    p_rv.add_argument("--password", "-p", required=True)
    p_rv.add_argument("--out",           default="restored.mp4")
    p_rv.add_argument("--fps",           default=30.0, type=float)
    p_rv.add_argument("--ckpt",          default=None)
    p_rv.add_argument("--device",        default=None)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    from pipeline import SecureFaceRX

    if args.command == "protect":
        frame = cv2.imread(args.input)
        if frame is None:
            sys.exit(f"이미지 로드 실패: {args.input}")

        out_psf = args.out or (Path(args.input).stem + ".psf")
        pipeline = SecureFaceRX(
            checkpoint_path=args.ckpt,
            device=args.device,
            obf_type=args.obf,
        )
        protected, psf_path = pipeline.protect_image(
            frame, args.password, out_psf=out_psf, as_zip=args.zip
        )
        # 보호 결과 미리보기 저장
        preview_path = str(Path(out_psf).with_suffix(".preview.png"))
        cv2.imwrite(preview_path, protected)
        print(f"[보호 완료]")
        print(f"  PSF      : {psf_path}")
        print(f"  미리보기 : {preview_path}")

    elif args.command == "restore":
        out_img = args.out or (Path(args.psf).stem + "_restored.png")
        pipeline = SecureFaceRX(
            checkpoint_path=args.ckpt,
            device=args.device,
        )
        restored = pipeline.restore_image(args.psf, args.password)
        cv2.imwrite(out_img, restored)
        print(f"[복원 완료] → {out_img}")

    elif args.command == "protect-video":
        pipeline = SecureFaceRX(
            checkpoint_path=args.ckpt,
            device=args.device,
            obf_type=args.obf,
        )
        pipeline.protect_video(args.input, args.password, out_dir=args.out_dir)

    elif args.command == "restore-video":
        pipeline = SecureFaceRX(
            checkpoint_path=args.ckpt,
            device=args.device,
        )
        pipeline.restore_video(
            args.psf_dir, args.password,
            out_path=args.out, fps=args.fps
        )


if __name__ == "__main__":
    main()
