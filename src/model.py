import torch
from torch import nn


class MLPICON(nn.Module):
    """Simplified MLP-based ICON baseline."""

    def __init__(self, input_dim: int = 1100, output_dim: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [B, 1100] -> y_pred shape [B, 100].
        # x 形状 [B, 1100] -> y_pred 形状 [B, 100]。
        return self.net(x)
