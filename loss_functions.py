"""
손실 함수 (SRS §7 / config.py 기준)
L_total = λ1·L_guide + λ2·L_recon + λ3·L_wr

L_guide  : ŷ ↔ y  지각 유사도 (LPIPS·L1 혼합)
L_recon  : x̌ ↔ x  L1 픽셀 복원
L_wr     : 오복원 강화 손실
    - RandWR: Triplet(x̌_wrong, x, random_noise) — 논문 Table III PSNR<11dB
    - ObfsWR: L1(x̌_wrong, y) + LPIPS triplet   — 오복원이 난독화처럼 보이도록
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class LPIPSLoss(nn.Module):
    """lpips 패키지 래퍼 (net='vgg')."""
    def __init__(self):
        super().__init__()
        try:
            import lpips
            self._fn = lpips.LPIPS(net='vgg')
        except ImportError:
            self._fn = None
            import warnings
            warnings.warn("lpips 패키지 없음. LPIPS 손실 비활성화.")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._fn is None:
            return torch.tensor(0.0, device=x.device)
        return self._fn(x * 2 - 1, y * 2 - 1).mean()


class TripletLoss(nn.Module):
    """
    L_anchor와 L_negative 사이의 거리를 최대화, L_positive는 최소화.
    RandWR: anchor=x̌_correct, positive=x, negative=noise
    """
    def __init__(self, margin: float = config.TRIPLET_MARGIN):
        super().__init__()
        self._fn = nn.TripletMarginLoss(margin=margin, p=1, reduction='mean')

    def forward(
        self,
        anchor: torch.Tensor,     # 정상복원 x̌
        positive: torch.Tensor,   # 원본 x
        negative: torch.Tensor,   # 오복원 x⃛
    ) -> torch.Tensor:
        return self._fn(anchor, positive, negative)


class PerceptualTripletLoss(nn.Module):
    """LPIPS 기반 Triplet (논문의 percep_triplet_loss)."""
    def __init__(self, margin: float = config.TRIPLET_MARGIN):
        super().__init__()
        self.margin  = margin
        self.lpips   = LPIPSLoss()

    def forward(self, anchor, positive, negative):
        d_pos = self.lpips(anchor, positive)
        d_neg = self.lpips(anchor, negative)
        return F.relu(d_pos - d_neg + self.margin).mean()


class TotalLoss(nn.Module):
    """
    L_total = λ1·L_guide + λ2·L_recon + λ3·L_wr
    wrong_recover_type: 'Random' | 'Obfs'
    """

    def __init__(self, wrong_recover_type: str = config.WRONG_RECOVER_TYPE):
        super().__init__()
        self.wr_type    = wrong_recover_type
        self.lpips_loss = LPIPSLoss()
        self.triplet    = TripletLoss(margin=config.TRIPLET_MARGIN)
        self.perc_trip  = PerceptualTripletLoss(margin=config.TRIPLET_MARGIN)

        self.lam_guide  = config.LAMBDA_GUIDE
        self.lam_recon  = config.LAMBDA_RECONSTRUCTION
        self.lam_wr     = config.LAMBDA_LOW_FREQUENCY

    def forward(
        self,
        ya_hat: torch.Tensor,     # 보호본 ŷ
        y:      torch.Tensor,     # 사전난독 y
        x_rec:  torch.Tensor,     # 정상복원 x̌
        x:      torch.Tensor,     # 원본 x
        x_wrong:torch.Tensor,     # 오복원 x⃛
    ) -> tuple[torch.Tensor, dict]:

        # L_guide: ŷ ≈ y (보호본이 시각적으로 난독화처럼 보여야)
        l_guide = F.l1_loss(ya_hat, y) + 0.1 * self.lpips_loss(ya_hat, y)

        # L_recon: 정상복원 ↔ 원본
        l_recon = F.l1_loss(x_rec, x)

        # L_wr: 오복원 강화
        if self.wr_type == 'Random':
            noise = torch.rand_like(x)
            l_wr = self.triplet(x_rec, x, x_wrong) + self.perc_trip(x_rec, x, x_wrong)
        else:  # Obfs
            l_wr = F.l1_loss(x_wrong, y) + self.perc_trip(x_wrong, y, x_rec)

        loss = self.lam_guide * l_guide + self.lam_recon * l_recon + self.lam_wr * l_wr

        breakdown = {
            "L_guide": l_guide.item(),
            "L_recon": l_recon.item(),
            "L_wr":    l_wr.item(),
            "L_total": loss.item(),
        }
        return loss, breakdown
