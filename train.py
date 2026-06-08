"""
SecureFace-RX 학습 스크립트 (M1 재현 + 자체 학습용)
SRS §7 하이퍼파라미터 기준

데이터 구조 (CelebA align_crop 형식):
    data_root/
        train/  (얼굴 크롭 이미지 256×256)
        val/

사용:
    python train.py --data data_root --epochs 20 --batch 6 --obf blur
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import cv2
import numpy as np

import config
from models.embedder import ModelDWT
from models.modules import DWT
from utils.key_gen import generate_key, make_key_rec
from utils.image_processing import Obfuscator
from loss_functions import TotalLoss


# ── 데이터셋 ──────────────────────────────────────────────────────

class FaceDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str, size: int = config.NORM_RESOLUTION):
        self.paths = [
            p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXTS
        ]
        self.tf = T.Compose([
            T.ToTensor(),
            T.Resize((size, size), antialias=True),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img)


# ── 학습 루프 ─────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Train] device={device}, obf={args.obf}, wr={args.wr}")

    # 모델
    embedder = ModelDWT(n_blocks=config.INV_BLOCKS).to(device)
    dwt      = DWT().to(device)
    obf      = Obfuscator(obf_type=args.obf)
    criterion = TotalLoss(wrong_recover_type=args.wr).to(device)

    # init_scale: 가중치 초기화 스케일 (SRS §7)
    for m in embedder.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain=config.INIT_SCALE)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    optimizer = optim.Adam(
        embedder.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    # 체크포인트 이어하기
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        embedder.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"[Train] 체크포인트 재개: epoch {start_epoch}")

    # 데이터
    train_ds = FaceDataset(os.path.join(args.data, "train"))
    val_ds   = FaceDataset(os.path.join(args.data, "val"))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    print(f"[Train] train={len(train_ds)}, val={len(val_ds)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        embedder.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, x in enumerate(train_dl):
            x = x.to(device)
            bs = x.shape[0]
            w  = h = config.NORM_RESOLUTION

            # 사전 난독화
            y = obf(x)

            # 키
            skey     = generate_key("train_dummy", bs, w, h).to(device)
            skey_dwt = dwt(skey.float())

            # 보호
            z, ya_hat = embedder(x, y, skey_dwt, rev=False)
            z.zero_(); del z

            # 정상 복원
            key_rec = make_key_rec(skey_dwt)
            x_rec, _ = embedder(key_rec, ya_hat, skey_dwt, rev=True)

            # 오복원 (틀린 키)
            wrong_pw = "wrong_key_train"
            skey_w   = generate_key(wrong_pw, bs, w, h).to(device)
            skey_dwt_w = dwt(skey_w.float())
            key_rec_w  = make_key_rec(skey_dwt_w)
            x_wrong, _ = embedder(key_rec_w, ya_hat.detach(), skey_dwt_w, rev=True)

            loss, breakdown = criterion(ya_hat, y, x_rec, x, x_wrong)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += breakdown["L_total"]

            if step % 200 == 0:
                print(
                    f"  ep{epoch} step{step}/{len(train_dl)}"
                    f" | L_total={breakdown['L_total']:.4f}"
                    f" | L_guide={breakdown['L_guide']:.4f}"
                    f" | L_recon={breakdown['L_recon']:.4f}"
                    f" | L_wr={breakdown['L_wr']:.4f}"
                    f" | {time.time()-t0:.1f}s"
                )

        avg = epoch_loss / len(train_dl)
        print(f"[Epoch {epoch}] avg_loss={avg:.4f}, {time.time()-t0:.1f}s")

        # 체크포인트 저장
        ck_path = out_dir / f"ep{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "model": embedder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg,
        }, ck_path)
        print(f"  저장: {ck_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True)
    parser.add_argument("--epochs",  type=int,   default=15)
    parser.add_argument("--batch",   type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--obf",     default="blur",
                        choices=["blur","pixelate","median","mask","hybridAll"])
    parser.add_argument("--wr",      default=config.WRONG_RECOVER_TYPE,
                        choices=["Random","Obfs"])
    parser.add_argument("--device",  default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--resume",  default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
