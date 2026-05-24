"""
Training utilities for Qwen2SAM-Detecture.

Standalone implementations — no cross-project imports.
"""

import math
import random
import yaml
import numpy as np
import torch
from pathlib import Path


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Running average tracker for loss values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmupCosineScheduler:
    """
    Learning rate scheduler: linear warmup followed by cosine decay.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of epochs.
        min_lr: Minimum learning rate after decay.
        steps_per_epoch: Number of optimizer steps per epoch.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, steps_per_epoch: int = 1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self._compute_lr(base_lr)

    def _compute_lr(self, base_lr: float) -> float:
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            return base_lr * self.step_count / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            return self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

    def get_last_lr(self) -> list:
        return [pg["lr"] for pg in self.optimizer.param_groups]


def get_lr(optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(model, optimizer, epoch, path, extra=None):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
    }

    # Save only trainable parameters
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data
    state["model_trainable"] = trainable_state

    # Save DUSTBIN embedding explicitly
    if hasattr(model, "dustbin_embed"):
        state["dustbin_embed"] = model.dustbin_embed.data

    # V7: SAM3 frozen permanently (no LoRA). Nothing extra to save.

    if extra:
        state.update(extra)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device="cuda"):
    """Load training checkpoint."""
    state = torch.load(path, map_location=device)

    # Restore trainable parameters
    trainable_state = state.get("model_trainable", {})
    model_state = model.state_dict()
    for name, data in trainable_state.items():
        if name in model_state:
            model_state[name].copy_(data)

    # Restore optimizer — tolerate param-group mismatches that occur when
    # the checkpoint was saved with a smaller set of trainable groups
    # (e.g. resuming a Stage-1 checkpoint after new groups have been added
    # in Stage 2, such as the masked-row SEG output embeddings). In that
    # case we skip the optimizer-state reload and let AdamW restart its
    # moment estimates from zero — correct behaviour for newly-added groups.
    if optimizer is not None and "optimizer" in state:
        saved = state["optimizer"]
        n_saved = len(saved.get("param_groups", []))
        n_current = len(optimizer.param_groups)
        if n_saved != n_current:
            print(f"  WARNING: optimizer param_group count mismatch "
                  f"(saved={n_saved}, current={n_current}). "
                  f"Skipping optimizer-state restore — AdamW moments will "
                  f"reinitialise. Trainable weights still loaded from "
                  f"'model_trainable'.")
        else:
            optimizer.load_state_dict(saved)

    return state.get("epoch", 0)
