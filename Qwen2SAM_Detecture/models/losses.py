"""
V5 Loss functions for Qwen2SAM-Detecture.

L_total = λ_mask × (CE + Dice) + λ_lm × LM([SEG] only) + λ_orth × Orthogonal

The mask loss is the PRIMARY training signal. It backpropagates through
SAM → Projector → [SEG] hidden states → Qwen LoRA. The LM loss is
negligible (only predicting <|seg|> tokens via -100 masking). The
orthogonal regularisation keeps SAM LoRA well-conditioned.

No distillation loss — [SEG] token representations are visually-grounded,
not linguistic, so distilling to SAM's CLIP text encoder is counterproductive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qwen2sam_detecture import NUM_QUERY_SLOTS


# ===================================================================== #
#  Mask Losses                                                            #
# ===================================================================== #

def mask_pad_logits(logits, pad_mask):
    """Set PAD channel logits to -inf."""
    masked = logits.clone()
    inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
    masked[inf_mask] = float("-inf")
    return masked


def cross_entropy_loss(logits, targets, pad_mask):
    masked_logits = mask_pad_logits(logits, pad_mask)
    return F.cross_entropy(masked_logits, targets, reduction="mean")


def dice_loss(logits, targets, pad_mask, smooth=1.0):
    masked_logits = mask_pad_logits(logits, pad_mask)
    probs = F.softmax(masked_logits, dim=1)
    gt_onehot = F.one_hot(targets.long(), num_classes=NUM_QUERY_SLOTS)
    gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()

    B = logits.shape[0]
    active_mask = ~pad_mask
    dice_sum = torch.tensor(0.0, device=logits.device)
    n_active = 0

    for b in range(B):
        for c in range(NUM_QUERY_SLOTS):
            if not active_mask[b, c]:
                continue
            pred_c = probs[b, c].reshape(-1)
            gt_c = gt_onehot[b, c].reshape(-1)
            inter = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice_score = (2.0 * inter + smooth) / (union + smooth)
            dice_sum = dice_sum + (1.0 - dice_score)
            n_active += 1

    return dice_sum / max(n_active, 1)


def mask_loss(logits, targets, pad_mask, ce_weight=1.0, dice_weight=1.0):
    ce = cross_entropy_loss(logits, targets, pad_mask)
    dc = dice_loss(logits, targets, pad_mask)
    total = ce_weight * ce + dice_weight * dc
    return {"mask_total": total, "mask_ce": ce, "mask_dice": dc}


# ===================================================================== #
#  Orthogonal Regularisation                                              #
# ===================================================================== #

def orthogonal_regularization(model):
    penalty = None
    for module in model.modules():
        if hasattr(module, "orthogonal_penalty") and callable(module.orthogonal_penalty):
            p = module.orthogonal_penalty()
            if penalty is None:
                penalty = p
            else:
                penalty = penalty + p.to(penalty.device)
    return penalty if penalty is not None else torch.tensor(0.0)


# ===================================================================== #
#  V6 Proximity-Decayed LM Loss                                           #
# ===================================================================== #

def weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lm_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Per-token weighted language model loss with cosine-decayed weights.

    Instead of binary -100 masking (V5), each token gets a continuous
    weight from the Proximity Decay schedule: high at the start of each
    TEXTURE block (linguistic anchor) and low near <|seg|> (geometric
    freedom).

    Args:
        logits:     (B, L, V) raw Qwen logits.
        labels:     (B, L) token IDs (-100 for ignored positions).
        lm_weights: (B, L) per-token cosine-decayed weights.

    Returns:
        Scalar weighted LM loss.
    """
    # Standard causal LM shift: predict token i+1 from position i
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = lm_weights[..., 1:].contiguous()

    B, L, V = shift_logits.shape

    # Per-token CE (no reduction)
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, L)

    # Apply cosine-decayed weights (0 where labels == -100)
    valid_mask = (shift_labels != -100).float()
    weighted = per_token_loss * shift_weights * valid_mask

    # Normalise by total weight (not count — heavier tokens count more)
    total_weight = (shift_weights * valid_mask).sum().clamp(min=1e-6)
    return weighted.sum() / total_weight


# ===================================================================== #
#  V6 Combined Loss                                                       #
# ===================================================================== #

def combined_loss(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    pad_mask: torch.Tensor,
    k_gts: torch.Tensor,
    lm_loss: torch.Tensor,
    model: nn.Module,
    cfg: dict,
    qwen_logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    lm_weights: torch.Tensor = None,
) -> dict:
    """
    V6 combined loss: Mask (CE + Dice) + Proximity-Decayed LM + Orthogonal.

    If qwen_logits + labels + lm_weights are provided, uses the V6
    proximity-decayed weighted LM loss. Otherwise falls back to Qwen's
    internal loss (V5 compat / Oracle mode where LoRA is frozen).
    """
    w = cfg.get("loss", {})
    lam_mask = w.get("mask_weight", 1.0)
    lam_lm = w.get("lm_weight", 0.1)
    # V7: SAM3 frozen permanently, no LoRA → orthogonal_regularization()
    # returns 0 regardless of this weight. Kept for interface compatibility.
    lam_orth = w.get("orthogonal_weight", 0.0)
    ce_w = w.get("ce_weight", 1.0)
    dice_w = w.get("dice_weight", 3.0)

    # Upsample logits to GT resolution
    H_target, W_target = gt_masks.shape[1], gt_masks.shape[2]
    H_logit, W_logit = mask_logits.shape[2], mask_logits.shape[3]
    if H_logit != H_target or W_logit != W_target:
        logits_up = F.interpolate(
            mask_logits.float(), size=(H_target, W_target),
            mode="bilinear", align_corners=False,
        )
    else:
        logits_up = mask_logits.float()

    m_losses = mask_loss(logits_up, gt_masks, pad_mask,
                         ce_weight=ce_w, dice_weight=dice_w)

    # V6 Proximity-Decayed LM loss (if available)
    if qwen_logits is not None and labels is not None and lm_weights is not None:
        l_lm = weighted_lm_loss(qwen_logits, labels, lm_weights)
    else:
        # Fallback: use Qwen's internal loss (Oracle/frozen LoRA mode)
        l_lm = lm_loss if lm_loss is not None else torch.tensor(0.0)

    l_orth = orthogonal_regularization(model)

    total = (lam_mask * m_losses["mask_total"]
             + lam_lm * l_lm
             + lam_orth * l_orth)

    return {
        "total": total,
        "mask_total": m_losses["mask_total"],
        "mask_ce": m_losses["mask_ce"],
        "mask_dice": m_losses["mask_dice"],
        "lm_loss": l_lm,
        "orthogonal_reg": l_orth,
    }
