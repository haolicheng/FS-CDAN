import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grl(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class STBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_blocks: int):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        layers += [ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeterogeneousFeatureFusion(nn.Module):
    def __init__(self, feature_channels: int, aux_dim: int):
        super().__init__()
        self.aux_vector_proj = nn.Linear(aux_dim, feature_channels)
        self.aux_grid_proj = nn.Conv2d(aux_dim, feature_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 4, kernel_size=1),
        )

    def _auxiliary_map(
        self,
        aux: Optional[torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if aux is None:
            return torch.zeros(batch_size, self.aux_vector_proj.out_features, height, width, device=device)

        if aux.dim() == 2:
            aux_map = self.aux_vector_proj(aux).unsqueeze(-1).unsqueeze(-1)
            return aux_map.expand(-1, -1, height, width)

        if aux.dim() == 4:
            if aux.shape[2:] != (height, width):
                aux = F.interpolate(aux, size=(height, width), mode="bilinear", align_corners=False)
            return self.aux_grid_proj(aux)

        raise ValueError(f"aux must be [B, D] or [B, D, H, W], got {list(aux.shape)}")

    def forward(
        self,
        close_features: torch.Tensor,
        period_features: torch.Tensor,
        trend_features: torch.Tensor,
        aux: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, height, width = close_features.shape
        aux_features = self._auxiliary_map(
            aux,
            batch_size=batch_size,
            height=height,
            width=width,
            device=close_features.device,
        )

        stacked = torch.cat([close_features, period_features, trend_features, aux_features], dim=1)
        weights = torch.softmax(self.attention(stacked), dim=1)
        return (
            weights[:, 0:1] * close_features
            + weights[:, 1:2] * period_features
            + weights[:, 2:3] * trend_features
            + weights[:, 3:4] * aux_features
        )


class SpatioTemporalFeatureExtractor(nn.Module):
    def __init__(
        self,
        len_closeness: int,
        len_period: int,
        len_trend: int,
        hidden_channels: int,
        num_blocks: int,
    ):
        super().__init__()
        self.close_branch = STBranch(len_closeness, hidden_channels, num_blocks)
        self.period_branch = STBranch(len_period, hidden_channels, num_blocks)
        self.trend_branch = STBranch(len_trend, hidden_channels, num_blocks)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, xc: torch.Tensor, xp: torch.Tensor, xt: torch.Tensor):
        return self.close_branch(xc), self.period_branch(xp), self.trend_branch(xt)

    def finalize(self, features: torch.Tensor) -> torch.Tensor:
        return self.fuse(features)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FSCDAN(nn.Module):
    def __init__(
        self,
        len_closeness: int,
        len_period: int,
        len_trend: int,
        hidden_channels: int,
        output_channels: int,
        aux_dim: int,
        use_auxiliary: bool,
        dropout: float,
        num_blocks: int,
    ):
        super().__init__()
        self.use_auxiliary = use_auxiliary and aux_dim > 0
        self.feature_extractor = SpatioTemporalFeatureExtractor(
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
        )
        self.hffm = HeterogeneousFeatureFusion(hidden_channels, aux_dim) if self.use_auxiliary else None
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
        )
        self.discriminator = DomainDiscriminator(hidden_channels)

    def encode(
        self,
        xc: torch.Tensor,
        xp: torch.Tensor,
        xt: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        close_features, period_features, trend_features = self.feature_extractor(xc, xp, xt)
        if self.hffm is not None:
            features = self.hffm(close_features, period_features, trend_features, aux)
        else:
            features = (close_features + period_features + trend_features) / 3.0
        return self.feature_extractor.finalize(features)

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.predictor(features)

    def forward(
        self,
        src_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        tgt_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None,
        alpha: float = 1.0,
    ):
        src_xc, src_xp, src_xt, src_aux = src_batch
        src_features = self.encode(src_xc, src_xp, src_xt, src_aux)
        src_pred = self.predict_from_features(src_features)

        if tgt_batch is None:
            return src_pred

        tgt_xc, tgt_xp, tgt_xt, tgt_aux = tgt_batch
        tgt_features = self.encode(tgt_xc, tgt_xp, tgt_xt, tgt_aux)
        tgt_pred = self.predict_from_features(tgt_features)
        src_domain_logits = self.discriminator(grl(src_features, alpha))
        tgt_domain_logits = self.discriminator(grl(tgt_features, alpha))
        return src_pred, tgt_pred, src_domain_logits, tgt_domain_logits


def dann_alpha(epoch: int, total_epochs: int) -> float:
    progress = epoch / max(total_epochs - 1, 1)
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
