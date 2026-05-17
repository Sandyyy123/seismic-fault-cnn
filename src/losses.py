"""
Loss functions for seismic fault segmentation.

Seismic fault labels are highly imbalanced (fault pixels are rare),
so pure cross-entropy performs poorly. Dice loss handles class imbalance better.
The combined loss (BCE + Dice) is the recommended default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary or multi-class segmentation.
    Handles class imbalance without explicit weighting.
    """

    def __init__(self, smooth: float = 1.0, num_classes: int = 2):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal loss - down-weights easy negatives, focuses on hard fault pixels.
    Good when fault pixels are <5% of the section.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of cross-entropy (or focal) and Dice loss.
    Default weights (0.5 / 0.5) work well for moderate class imbalance.
    Use more Dice weight for severe imbalance (<2% fault pixels).
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        num_classes: int = 2,
        use_focal: bool = False,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = FocalLoss() if use_focal else nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
