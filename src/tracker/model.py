"""Small MLP that scores whether a detection continues a track."""

from __future__ import annotations

import torch
from torch import nn

from tracker.constants import FEATURE_NAMES


class AssociationNet(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
