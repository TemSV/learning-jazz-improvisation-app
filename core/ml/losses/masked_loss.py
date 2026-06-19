import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:

        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = padding_mask.view(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none',
            ignore_index=-100
        )

        prob = torch.exp(-ce_loss)

        focal_term = (1 - prob) ** self.gamma

        alpha_t = torch.where(targets_flat == 1, self.alpha, 1.0 - self.alpha)

        loss = alpha_t * focal_term * ce_loss

        loss = loss * mask_flat.float()

        valid_tokens = mask_flat.sum().float().clamp(min=1.0)

        return loss.sum() / valid_tokens