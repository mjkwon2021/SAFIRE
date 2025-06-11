"""
Myung-Joon Kwon
2024-02-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class R2R_ContrastiveLoss_TwoSource(nn.Module):
    """
    Region to Region Contrastive Loss for two sources used in SAFIRE pre-training.

    Args
    ----
    temperature (float): Softmax temperature (default 0.1)
    reduction  (str)   : 'mean' | 'sum' | 'none' – passed to F.cross_entropy
    """

    def __init__(self, temperature: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    @torch.no_grad()
    def _validate_inputs(pos: torch.Tensor, neg: torch.Tensor) -> None:
        if pos.dim() != 2 or neg.dim() != 2:
            raise ValueError("Both <positive_keys> and <negative_keys> must be 2-D tensors (N, D).")
        if pos.size(1) != neg.size(1):
            raise ValueError("Positive and negative keys must share the same feature dimension.")

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        positive_keys: torch.Tensor,
        negative_keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        positive_keys : Tensor [N_pos, D]
        negative_keys : Tensor [N_neg, D]

        Returns
        -------
        Scalar loss (float tensor)
        """
        self._validate_inputs(positive_keys, negative_keys)

        # 1) L2-normalize --------------------------------------------------
        positive_keys = F.normalize(positive_keys, dim=-1)
        negative_keys = F.normalize(negative_keys, dim=-1)

        # 2) Within-set cosine similarities --------------------------------
        pos_sim = positive_keys @ positive_keys.T  # [N_pos, N_pos]
        neg_sim = negative_keys @ negative_keys.T  # [N_neg, N_neg]

        pos_mask = (torch.ones_like(pos_sim) -
                    torch.eye(pos_sim.size(0), device=pos_sim.device)).bool()
        neg_mask = (torch.ones_like(neg_sim) -
                    torch.eye(neg_sim.size(0), device=neg_sim.device)).bool()

        pos_sim = pos_sim.masked_select(pos_mask).view(pos_sim.size(0), -1)  # [N_pos, N_pos-1]
        neg_sim = neg_sim.masked_select(neg_mask).view(neg_sim.size(0), -1)  # [N_neg, N_neg-1]

        # Mean-pooled anchor logits (first column in final logit table)
        pos_anchor_logit = pos_sim.mean(dim=1, keepdim=True)  # [N_pos, 1]
        neg_anchor_logit = neg_sim.mean(dim=1, keepdim=True)  # [N_neg, 1]

        # 3) Cross-set similarities ---------------------------------------
        cross_sim = positive_keys @ negative_keys.T          # [N_pos, N_neg]

        # 4) Assemble final logits ----------------------------------------
        pos_logits = torch.cat([pos_anchor_logit, cross_sim], dim=1)      # [N_pos, 1+N_neg]
        neg_logits = torch.cat([neg_anchor_logit, cross_sim.T], dim=1)    # [N_neg, 1+N_pos]

        # The first column (index 0) is always the positive target
        labels_pos = torch.zeros(pos_logits.size(0), dtype=torch.long, device=pos_logits.device)
        labels_neg = torch.zeros(neg_logits.size(0), dtype=torch.long, device=neg_logits.device)

        # 5) InfoNCE loss --------------------------------------------------
        loss_pos = F.cross_entropy(pos_logits / self.temperature, labels_pos, reduction=self.reduction)
        loss_neg = F.cross_entropy(neg_logits / self.temperature, labels_neg, reduction=self.reduction)

        return loss_pos + loss_neg