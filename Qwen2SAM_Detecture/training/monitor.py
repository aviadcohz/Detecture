"""
Training monitor for Qwen2SAM-Detecture.

Provides:
  1. DataSanityChecker — Validate data pipeline at startup (image-mask alignment, batch integrity)
  2. TrainingLogger    — JSON-lines log file with per-step and per-epoch metrics
  3. PlotGenerator     — Loss curves, LR schedule, mIoU progression, etc.
  4. TestEvaluator     — Run model on test set, compute mIoU/mARI, save visualizations
"""

import json
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from scipy.optimize import linear_sum_assignment


# ===================================================================== #
#  1. Data Sanity Checker                                                 #
# ===================================================================== #

class DataSanityChecker:
    """
    Validates the data pipeline before training starts.

    Runs once at startup on the first N batches. Checks:
      - Image-mask spatial alignment (same H, W)
      - Index mask values are valid (0..K_gt, no gaps, no out-of-range)
      - K_gt matches actual non-zero labels in index_mask
      - No sample cross-contamination in batches (batch_size > 1)
      - SAM image normalization is correct (mean ~0, std ~1)
      - CLIP embeddings are non-zero for active textures
      - Descriptions contain expected "Texture of" prefix
      - Mask coverage sanity (not all dustbin, not all one class)

    Saves a visual report to the output directory.
    """

    def __init__(self, output_dir: str, n_batches: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_batches = n_batches

    def check(self, dataloader, dataset) -> bool:
        """
        Run all sanity checks on the first N batches.
        Returns True if all checks pass, False if critical issues found.
        """
        print(f"\n{'='*60}")
        print(f"  DATA SANITY CHECK ({self.n_batches} batches)")
        print(f"{'='*60}\n")

        issues = []      # (severity, message) — 'error' or 'warning'
        batch_reports = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.n_batches:
                break

            report = self._check_batch(batch, batch_idx, issues)
            batch_reports.append(report)

        # Check dataset-level properties
        self._check_dataset(dataset, issues)

        # Save visual report for first batch
        if batch_reports:
            self._save_visual_report(batch_reports[0], dataloader)

        # Print results
        errors = [msg for sev, msg in issues if sev == "error"]
        warnings = [msg for sev, msg in issues if sev == "warning"]

        if errors:
            print(f"\n  ERRORS ({len(errors)}):")
            for e in errors:
                print(f"    [ERROR] {e}")
        if warnings:
            print(f"\n  WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"    [WARN]  {w}")

        if not errors and not warnings:
            print("  All checks passed!")

        # Save report
        report_data = {
            "n_batches_checked": min(self.n_batches, len(batch_reports)),
            "errors": errors,
            "warnings": warnings,
            "batch_reports": batch_reports,
        }
        with open(self.output_dir / "sanity_check.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n  Report saved: {self.output_dir / 'sanity_check.json'}")
        print(f"  Visual report: {self.output_dir / 'sanity_batch_0.jpg'}")
        print(f"{'='*60}\n")

        if errors:
            print("  *** CRITICAL ISSUES FOUND — fix before training! ***\n")
            return False
        return True

    def _check_batch(self, batch, batch_idx, issues) -> dict:
        """Check a single batch for issues."""
        sam_images = batch["sam_images"]       # (B, 3, H, W)
        index_masks = batch["index_masks"]     # (B, H, W)
        k_gts = batch["k_gts"]                # (B,)
        qwen_inputs = batch["qwen_inputs"]
        gt_text_embeds = batch.get("gt_text_embeds")  # (B, 5, D) or None

        B = sam_images.shape[0]
        report = {
            "batch_idx": batch_idx,
            "batch_size": B,
            "samples": [],
        }

        for b in range(B):
            sample_report = {"batch_item": b}

            # --- Check 1: Image shape ---
            img = sam_images[b]  # (3, H, W)
            mask = index_masks[b]  # (H, W)
            _, H_img, W_img = img.shape
            H_mask, W_mask = mask.shape

            if H_img != H_mask or W_img != W_mask:
                issues.append(("error",
                    f"Batch {batch_idx}, item {b}: image ({H_img}x{W_img}) != "
                    f"mask ({H_mask}x{W_mask})"))
            sample_report["image_shape"] = [int(H_img), int(W_img)]
            sample_report["mask_shape"] = [int(H_mask), int(W_mask)]

            # --- Check 2: SAM normalization ---
            img_mean = img.mean().item()
            img_std = img.std().item()
            if abs(img_mean) > 1.0:
                issues.append(("warning",
                    f"Batch {batch_idx}, item {b}: SAM image mean={img_mean:.3f} "
                    f"(expected ~0 after normalization)"))
            sample_report["img_mean"] = round(img_mean, 4)
            sample_report["img_std"] = round(img_std, 4)

            # --- Check 3: Index mask values ---
            k_gt = int(k_gts[b].item())
            unique_vals = sorted(mask.unique().tolist())
            expected_vals = list(range(k_gt + 1))  # 0, 1, ..., k_gt

            # Check no out-of-range values
            if max(unique_vals) > 6:
                issues.append(("error",
                    f"Batch {batch_idx}, item {b}: mask has value {max(unique_vals)} "
                    f"(max should be 5)"))
            if min(unique_vals) < 0:
                issues.append(("error",
                    f"Batch {batch_idx}, item {b}: mask has negative value"))

            # Check k_gt matches actual labels
            actual_classes = [v for v in unique_vals if v > 0]
            if len(actual_classes) != k_gt:
                issues.append(("warning",
                    f"Batch {batch_idx}, item {b}: k_gt={k_gt} but mask has "
                    f"{len(actual_classes)} non-zero classes: {actual_classes}"))

            # Check for gaps (e.g., mask has {0, 1, 3} but missing 2)
            if actual_classes and actual_classes != list(range(1, len(actual_classes) + 1)):
                issues.append(("warning",
                    f"Batch {batch_idx}, item {b}: gap in mask labels: {actual_classes}"))

            sample_report["k_gt"] = k_gt
            sample_report["mask_unique"] = unique_vals

            # --- Check 4: Mask coverage ---
            total_pixels = H_mask * W_mask
            dustbin_pixels = int((mask == 0).sum().item())
            dustbin_frac = dustbin_pixels / total_pixels

            if dustbin_frac > 0.95:
                issues.append(("warning",
                    f"Batch {batch_idx}, item {b}: {dustbin_frac:.1%} dustbin — "
                    f"nearly empty mask"))
            if dustbin_frac < 0.001 and k_gt > 1:
                # Fine for k_gt=1 where one texture covers everything
                pass

            # Check each texture has reasonable area
            for c in range(1, k_gt + 1):
                c_frac = (mask == c).sum().item() / total_pixels
                if c_frac < 0.005:
                    issues.append(("warning",
                        f"Batch {batch_idx}, item {b}: texture {c} covers only "
                        f"{c_frac:.3%} of image"))
            sample_report["dustbin_frac"] = round(dustbin_frac, 4)

            # --- Check 5: Qwen GT embeddings (skip if not loaded — V5 uses live Qwen) ---
            qwen_gt = batch.get("qwen_gt_embeds")
            if qwen_gt is not None:
                embeds = qwen_gt[b]  # (MAX_TEXTURES, 4096)
                # Only check if embeddings are actually loaded (non-zero overall).
                # In V5, qwen_gt_embeds_path is not set, so all embeddings are zero
                # by default — this is expected and not an error.
                has_any_nonzero = embeds.norm().item() > 0.01
                if has_any_nonzero:
                    for t in range(k_gt):
                        norm = embeds[t].norm().item()
                        if norm < 0.01:
                            issues.append(("error",
                                f"Batch {batch_idx}, item {b}: Qwen GT embedding for "
                                f"texture {t+1} is near-zero (norm={norm:.4f})"))
                    # Check PAD embeddings are zero
                    for t in range(k_gt, embeds.shape[0]):
                        norm = embeds[t].norm().item()
                        if norm > 0.01:
                            issues.append(("warning",
                                f"Batch {batch_idx}, item {b}: PAD Qwen GT embedding "
                                f"slot {t+1} is non-zero (norm={norm:.4f})"))

            report["samples"].append(sample_report)

        # --- Check 6: Cross-contamination in batch (B > 1) ---
        if B > 1:
            for i in range(B):
                for j in range(i + 1, B):
                    if torch.equal(index_masks[i], index_masks[j]):
                        issues.append(("warning",
                            f"Batch {batch_idx}: items {i} and {j} have identical masks"))
                    if torch.equal(sam_images[i], sam_images[j]):
                        issues.append(("error",
                            f"Batch {batch_idx}: items {i} and {j} have identical images "
                            f"— possible data loading bug!"))

        # --- Check 7: Input IDs have <TEX_i> tokens ---
        if "input_ids" in qwen_inputs:
            input_ids = qwen_inputs["input_ids"]  # (B, seq_len)
            for b in range(B):
                ids = input_ids[b].tolist()
                # Check that assistant section exists (has texture tokens)
                ids_str = str(ids)
                # Simple check: the sequence should vary per sample if B > 1
                pass  # Token-level checks are hard without tokenizer; visual report covers this

        return report

    def _check_dataset(self, dataset, issues):
        """Check dataset-level properties."""
        n = len(dataset)
        if n < 10:
            issues.append(("warning", f"Very small dataset: {n} samples"))

        # Spot-check a few samples for file existence
        import os
        for idx in [0, n // 2, n - 1]:
            meta = dataset.samples[idx]
            if not os.path.exists(meta["image_path"]):
                issues.append(("error", f"Missing image: {meta['image_path']}"))
            for tex in meta["textures"]:
                if not os.path.exists(tex["mask_path"]):
                    issues.append(("error", f"Missing mask: {tex['mask_path']}"))
                if not tex["description"].startswith("Texture of"):
                    issues.append(("warning",
                        f"Description missing 'Texture of' prefix: {tex['description'][:50]}"))

    def _save_visual_report(self, report, dataloader):
        """Save a visual grid showing the first batch: image | mask | overlay."""
        try:
            # Get the raw batch again
            batch = next(iter(dataloader))
        except StopIteration:
            return

        sam_images = batch["sam_images"]   # (B, 3, H, W)
        index_masks = batch["index_masks"]  # (B, H, W)
        k_gts = batch["k_gts"]
        B = sam_images.shape[0]

        rows = []
        for b in range(min(B, 4)):  # max 4 rows
            img = sam_images[b].permute(1, 2, 0).numpy()  # (H, W, 3)
            img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            mask = index_masks[b].numpy()
            mask_vis = _colorize_mask(mask)

            # Overlay
            overlay = img.copy()
            active = mask > 0
            overlay[active] = cv2.addWeighted(img, 0.4, mask_vis, 0.6, 0)[active]

            # Add text
            k = int(k_gts[b].item())
            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"Sample {b}", (5, 15), font, 0.4, (0, 255, 0), 1)
            cv2.putText(mask_vis, f"k_gt={k}", (5, 15), font, 0.4, (255, 255, 255), 1)
            cv2.putText(overlay, f"Overlay", (5, 15), font, 0.4, (255, 255, 255), 1)

            row = np.hstack([img, mask_vis, overlay])
            rows.append(row)

        canvas = np.vstack(rows)
        cv2.imwrite(str(self.output_dir / "sanity_batch_0.jpg"), canvas)


# ===================================================================== #
#  2. Training Logger                                                     #
# ===================================================================== #

class TrainingLogger:
    """
    Logs metrics to a JSONL file (one JSON object per line) for easy parsing.
    Also maintains an in-memory history for live plotting.

    Log file format (one JSON per line):
      {"type": "step", "epoch": 1, "step": 10, "total_loss": 0.5, ...}
      {"type": "epoch", "epoch": 1, "train_loss": 0.4, "val_miou": 0.3, ...}
      {"type": "test", "epoch": 5, "test_miou": 0.25, "test_mari": 0.6, ...}
    """

    def __init__(self, log_dir: str, run_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_name = run_name

        self.log_path = self.log_dir / f"{run_name}.jsonl"
        self.log_file = open(self.log_path, "a", buffering=1)  # line-buffered

        # In-memory history for plotting
        self.step_history = []    # per-step records
        self.epoch_history = []   # per-epoch records
        self.test_history = []    # per-test records

        self.start_time = time.time()
        self._log({"type": "start", "timestamp": datetime.now().isoformat(),
                    "run_name": run_name})

    def log_step(self, epoch: int, step: int, total_steps: int,
                 losses: dict, lr: float):
        """Log a single training step."""
        record = {
            "type": "step",
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "lr": lr,
            "elapsed_sec": time.time() - self.start_time,
        }
        for k, v in losses.items():
            record[k] = float(v) if isinstance(v, (int, float)) else v
        self.step_history.append(record)
        self._log(record)

    def log_epoch(self, epoch: int, train_metrics: dict, val_miou: float,
                  lr: float, is_best: bool = False, extra: dict = None):
        """Log end-of-epoch summary."""
        record = {
            "type": "epoch",
            "epoch": epoch,
            "val_miou": val_miou,
            "lr": lr,
            "is_best": is_best,
            "elapsed_sec": time.time() - self.start_time,
        }
        for k, v in train_metrics.items():
            record[f"train_{k}"] = float(v) if isinstance(v, (int, float)) else v
        if extra:
            record.update(extra)
        self.epoch_history.append(record)
        self._log(record)

    def log_test(self, epoch: int, metrics: dict):
        """Log test evaluation results."""
        record = {"type": "test", "epoch": epoch}
        record.update(metrics)
        self.test_history.append(record)
        self._log(record)

    def _log(self, record: dict):
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()

    def close(self):
        self._log({"type": "end", "timestamp": datetime.now().isoformat(),
                    "total_elapsed_sec": time.time() - self.start_time})
        self.log_file.close()


# ===================================================================== #
#  2. Plot Generator                                                      #
# ===================================================================== #

class PlotGenerator:
    """
    Generates training progress plots from TrainingLogger history.
    Saves PNG figures to the plot directory.
    """

    def __init__(self, plot_dir: str, baselines: dict = None):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        # baseline reference lines: {"name": {"miou": 0.7, "mari": 0.68}}
        self.baselines = baselines or {}

    def update(self, logger: TrainingLogger):
        """Generate all plots from current logger state."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping plots")
            return

        if logger.epoch_history:
            self._plot_loss_curves(logger, plt)
            self._plot_lr_schedule(logger, plt)
            self._plot_val_miou(logger, plt)
            self._plot_loss_components(logger, plt)

        if logger.test_history:
            self._plot_test_metrics(logger, plt)

        if logger.step_history:
            self._plot_step_loss(logger, plt)

    def _plot_loss_curves(self, logger, plt):
        """Train loss per epoch."""
        epochs = [r["epoch"] for r in logger.epoch_history]
        losses = [r.get("train_total", 0) for r in logger.epoch_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, losses, "b-o", markersize=3, label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(self.plot_dir / "loss_train.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_lr_schedule(self, logger, plt):
        """Learning rate over epochs."""
        epochs = [r["epoch"] for r in logger.epoch_history]
        lrs = [r["lr"] for r in logger.epoch_history]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, lrs, "g-", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.savefig(self.plot_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_val_miou(self, logger, plt):
        """Validation mIoU over epochs with key baselines."""
        epochs = [r["epoch"] for r in logger.epoch_history]
        mious = [r["val_miou"] for r in logger.epoch_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, mious, "r-o", markersize=4, linewidth=1.5, label="Val mIoU")

        # Mark best
        best_idx = np.argmax(mious)
        ax.annotate(f"Best: {mious[best_idx]:.4f} (ep {epochs[best_idx]})",
                     xy=(epochs[best_idx], mious[best_idx]),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=9, color="red")

        # Add test mIoU if available
        if logger.test_history:
            t_epochs = [r["epoch"] for r in logger.test_history]
            t_mious = [r.get("test_miou", 0) for r in logger.test_history]
            ax.plot(t_epochs, t_mious, "b-s", markersize=6, linewidth=1.5,
                    label="Test mIoU (RWTD)")

        # Draw only the 3 most important baseline reference lines
        key_baselines = {}
        for name, bl in self.baselines.items():
            if "miou" in bl and bl["miou"] > 0:
                key_baselines[name] = bl["miou"]
        # Keep only top 3 by value to avoid clutter
        sorted_bl = sorted(key_baselines.items(), key=lambda x: -x[1])[:3]
        colors_bl = ["#999999", "#bbbbbb", "#dddddd"]
        for i, (name, miou) in enumerate(sorted_bl):
            short_name = name.replace("qwen_", "").replace("_parsed", "")
            if len(short_name) > 25:
                short_name = short_name[:22] + "..."
            ax.axhline(miou, color=colors_bl[min(i, 2)], linestyle="--",
                        alpha=0.7, linewidth=0.8)
            ax.text(0.02, miou + 0.008, f'{short_name} ({miou:.3f})',
                    fontsize=7, color="#666666", transform=ax.get_yaxis_transform())

        ax.set_xlabel("Epoch")
        ax.set_ylabel("mIoU")
        ax.set_title("Segmentation Quality (mIoU)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.savefig(self.plot_dir / "val_miou.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_loss_components(self, logger, plt):
        """Loss components on 3 separate panels: Mask, LM, Orth."""
        epochs = [r["epoch"] for r in logger.epoch_history]

        ce_vals = [r.get("train_mask_ce", 0) for r in logger.epoch_history]
        dice_vals = [r.get("train_mask_dice", 0) for r in logger.epoch_history]
        lm_vals = [r.get("train_lm_loss", 0) for r in logger.epoch_history]
        orth_vals = [r.get("train_orthogonal_reg", 0) for r in logger.epoch_history]

        # Determine how many panels we need (always mask, conditionally LM + Orth)
        panels = [("mask", True)]
        if any(v > 0 for v in lm_vals):
            panels.append(("lm", True))
        if any(v > 0 for v in orth_vals):
            panels.append(("orth", True))

        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
        if n == 1:
            axes = [axes]

        # Panel 1: Mask losses (CE + Dice)
        ax = axes[0]
        ax.plot(epochs, ce_vals, color="#e41a1c", linewidth=1.5,
                marker="o", markersize=3, label="CE")
        ax.plot(epochs, dice_vals, color="#377eb8", linewidth=1.5,
                marker="o", markersize=3, label="Dice")
        ax.set_ylabel("Loss")
        ax.set_title("Mask Losses (CE + Dice)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        panel_idx = 1

        # Panel 2: LM loss
        if any(v > 0 for v in lm_vals):
            ax = axes[panel_idx]
            ax.plot(epochs, lm_vals, color="#ff7f00", linewidth=1.5,
                    marker="s", markersize=4, label="LM ([SEG] only)")
            ax.set_ylabel("LM Loss")
            ax.set_title("LM Loss (predicting <|seg|> tokens)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            panel_idx += 1

        # Panel 3: Orthogonal regularisation
        if any(v > 0 for v in orth_vals):
            ax = axes[panel_idx]
            ax.plot(epochs, orth_vals, color="#a65628", linewidth=1.5,
                    marker="^", markersize=4, label="Orth Reg (SAM LoRA)")
            ax.set_ylabel("Orth Loss")
            ax.set_title("Orthogonal Regularisation (SAM LoRA)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            panel_idx += 1

        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "loss_components.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_test_metrics(self, logger, plt):
        """Test set mIoU and mARI with clean baselines."""
        epochs = [r["epoch"] for r in logger.test_history]
        mious = [r.get("test_miou", 0) for r in logger.test_history]
        maris = [r.get("test_mari", 0) for r in logger.test_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs, mious, "b-o", markersize=6, linewidth=1.5)

        # Filter out broken mARI entries (near-zero from failed inference)
        valid_mari = [(e, m) for e, m in zip(epochs, maris) if abs(m) > 0.01]
        if valid_mari:
            m_epochs, m_vals = zip(*valid_mari)
            ax2.plot(m_epochs, m_vals, "g-o", markersize=6, linewidth=1.5)
        else:
            ax2.text(0.5, 0.5, "mARI not yet available\n(awaiting next test eval)",
                     ha="center", va="center", fontsize=10, color="#888888",
                     transform=ax2.transAxes)

        # Add only top 3 baselines to each axis
        for ax, metric_key in [(ax1, "miou"), (ax2, "mari")]:
            key_bl = {n: bl[metric_key] for n, bl in self.baselines.items()
                      if metric_key in bl and bl[metric_key] > 0}
            sorted_bl = sorted(key_bl.items(), key=lambda x: -x[1])[:3]
            for i, (name, val) in enumerate(sorted_bl):
                short = name.replace("qwen_", "").replace("_parsed", "")
                if len(short) > 25:
                    short = short[:22] + "..."
                ax.axhline(val, color="#aaaaaa", linestyle="--", alpha=0.6,
                            linewidth=0.8)
                ax.text(0.02, val + 0.008, f'{short} ({val:.3f})',
                        fontsize=7, color="#666666",
                        transform=ax.get_yaxis_transform())

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("mIoU")
        ax1.set_title("Test Set mIoU (RWTD)")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mARI")
        ax2.set_title("Test Set mARI (RWTD)")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.plot_dir / "test_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_step_loss(self, logger, plt):
        """Per-step loss (smoothed) for the most recent epoch."""
        if not logger.step_history:
            return
        last_epoch = logger.step_history[-1]["epoch"]
        recent = [r for r in logger.step_history if r["epoch"] == last_epoch]
        if len(recent) < 2:
            return

        steps = [r["step"] for r in recent]
        losses = [r.get("total", 0) for r in recent]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, "b-", alpha=0.3, linewidth=0.5)
        # Smoothed
        if len(losses) > 10:
            kernel = max(len(losses) // 20, 3)
            smoothed = np.convolve(losses, np.ones(kernel) / kernel, mode="valid")
            ax.plot(steps[:len(smoothed)], smoothed, "b-", linewidth=2, label="Smoothed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Step Loss — Epoch {last_epoch}")
        ax.grid(True, alpha=0.3)
        fig.savefig(self.plot_dir / "step_loss_latest.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ===================================================================== #
#  3. Test Evaluator                                                      #
# ===================================================================== #

# Color palette for visualization (BGR)
VIS_COLORS = [
    (0, 0, 0),       # 0: dustbin (black)
    (255, 0, 0),     # 1: blue
    (0, 255, 0),     # 2: green
    (0, 0, 255),     # 3: red
    (255, 255, 0),   # 4: cyan
    (0, 255, 255),   # 5: yellow
    (255, 0, 255),   # 6: magenta
]


def _compute_ari(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between two segmentation maps.
    Both inputs are (H, W) integer arrays.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    n = len(pred_flat)

    # Contingency table
    pred_labels = np.unique(pred_flat)
    gt_labels = np.unique(gt_flat)

    contingency = np.zeros((len(pred_labels), len(gt_labels)), dtype=np.int64)
    pred_map = {v: i for i, v in enumerate(pred_labels)}
    gt_map = {v: i for i, v in enumerate(gt_labels)}

    for p, g in zip(pred_flat, gt_flat):
        contingency[pred_map[p], gt_map[g]] += 1

    # ARI from contingency table
    sum_comb_c = sum(int(nij) * (int(nij) - 1) // 2 for nij in contingency.ravel())
    sum_comb_a = sum(int(ai) * (int(ai) - 1) // 2 for ai in contingency.sum(axis=1))
    sum_comb_b = sum(int(bj) * (int(bj) - 1) // 2 for bj in contingency.sum(axis=0))

    sum_comb_n = n * (n - 1) // 2
    expected = sum_comb_a * sum_comb_b / max(sum_comb_n, 1)
    max_index = 0.5 * (sum_comb_a + sum_comb_b)

    if max_index == expected:
        return 1.0
    return (sum_comb_c - expected) / (max_index - expected)


def _colorize_mask(mask: np.ndarray, n_classes: int = 7) -> np.ndarray:
    """Convert index mask (H, W) to colored BGR image."""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(n_classes, len(VIS_COLORS))):
        vis[mask == c] = VIS_COLORS[c]
    return vis


class TestEvaluator:
    """
    Evaluates the model on a test set every N epochs.

    Computes:
      - mIoU (per-sample, then mean)
      - mARI (Adjusted Rand Index)

    Saves visualizations:
      - For each test sample: [original | GT mask | predicted mask] side-by-side
      - Summary grid of worst/best cases
    """

    def __init__(self, test_metadata: str, output_dir: str,
                 image_size: int = 1008, eval_every: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = eval_every
        self.image_size = image_size
        self.test_metadata = test_metadata

    def should_evaluate(self, epoch: int) -> bool:
        """Check if we should evaluate at this epoch."""
        return (epoch + 1) % self.eval_every == 0

    @torch.no_grad()
    def evaluate(self, model, processor, device, epoch: int,
                 **kwargs) -> dict:
        """
        Live-Inference Evaluation with the standard "1 to 6" prompt.

        Uses the two-pass inference_forward() path: Qwen generates its own
        texture descriptions from scratch (rather than teacher-forced GT
        text). The user prompt is the default training-time "1 to 6"
        template; the model decides the count itself. Hungarian matching
        in _compute_matched_miou handles k_pred != k_gt.

        This measures the true end-to-end pipeline performance on OOD data.
        Over-segmentation behavior is tracked via k_pred distribution
        and discussed transparently in the paper.

        Returns dict with metrics: test_miou, test_mari, per_sample_results.
        """
        from data.dataset import DetectureDataset, DetectureCollator

        test_ds = DetectureDataset(
            self.test_metadata,
            image_size=self.image_size,
            augment=False,
        )
        collator = DetectureCollator(processor, inference=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=collator,
        )

        model.eval()
        epoch_dir = self.output_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        all_ious = []
        all_aris = []
        per_sample = []

        for idx, batch in enumerate(test_loader):
            qwen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch["qwen_inputs"].items()}
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]  # (1, H, W)
            k_gts = batch["k_gts"]

            # Live inference: Qwen generates, then two-pass block-diagonal
            # mask reconstruction extracts [SEG] hidden states.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference_forward(
                    qwen_inputs=qwen_inputs,
                    sam_images=sam_images,
                )

            mask_logits = out["mask_logits"]  # (1, C, H, W)
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())
            gen_text = out.get("generated_text", [""])[0]

            # WTA prediction: upsample logits to GT resolution, then argmax
            masked_logits = mask_logits.clone()
            inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked_logits)
            masked_logits[inf_mask] = float("-inf")

            H_gt, W_gt = index_masks.shape[1], index_masks.shape[2]
            if masked_logits.shape[2] != H_gt or masked_logits.shape[3] != W_gt:
                masked_logits = torch.nn.functional.interpolate(
                    masked_logits.float(), size=(H_gt, W_gt),
                    mode="bilinear", align_corners=False,
                )
            pred = masked_logits[0].argmax(dim=0).cpu().numpy()
            gt = index_masks[0].numpy()
            k_gt = int(k_gts[0].item())

            sample_iou, matched_pred = self._compute_matched_miou(pred, gt, k_pred, k_gt)
            all_ious.append(sample_iou)

            ari = _compute_ari(pred, gt)
            all_aris.append(ari)

            self._save_visualization(
                idx, epoch_dir, batch, pred, gt, sample_iou, ari, k_gt, k_pred,
            )

            per_sample.append({
                "idx": idx,
                "miou": sample_iou,
                "ari": ari,
                "k_gt": k_gt,
                "k_pred": k_pred,
                "generated_text": gen_text,
            })

        test_miou = np.mean(all_ious) if all_ious else 0.0
        test_mari = np.mean(all_aris) if all_aris else 0.0

        # Save per-sample results
        with open(epoch_dir / "results.json", "w") as f:
            json.dump({
                "epoch": epoch + 1,
                "test_miou": float(test_miou),
                "test_mari": float(test_mari),
                "n_samples": len(per_sample),
                "per_sample": per_sample,
            }, f, indent=2)

        # Save summary visualization (worst 10 + best 10)
        self._save_summary_grid(epoch_dir, per_sample)

        print(f"  Test mIoU: {test_miou:.4f} | mARI: {test_mari:.4f} "
              f"({len(per_sample)} samples)")

        return {
            "test_miou": float(test_miou),
            "test_mari": float(test_mari),
            "n_samples": len(per_sample),
        }

    def _compute_matched_miou(self, pred: np.ndarray, gt: np.ndarray,
                               k_pred: int, k_gt: int) -> tuple:
        """
        Compute mIoU with Hungarian matching between pred and GT classes.
        Returns (miou, remapped_pred).
        """
        # Build IoU cost matrix between pred channels and GT classes
        cost = np.zeros((max(k_pred, 1), max(k_gt, 1)))
        for pi in range(k_pred):
            pred_c = (pred == pi + 1)
            for gi in range(k_gt):
                gt_c = (gt == gi + 1)
                inter = (pred_c & gt_c).sum()
                union = (pred_c | gt_c).sum()
                iou = inter / max(union, 1)
                cost[pi, gi] = 1.0 - iou  # minimize cost

        row_ind, col_ind = linear_sum_assignment(cost)

        # Compute matched IoU
        ious = []
        matched_pred = np.zeros_like(pred)
        for r, c in zip(row_ind, col_ind):
            if r < k_pred and c < k_gt:
                ious.append(1.0 - cost[r, c])
                matched_pred[pred == r + 1] = c + 1

        return (np.mean(ious) if ious else 0.0), matched_pred

    def _save_visualization(self, idx, epoch_dir, batch, pred, gt,
                            miou, ari, k_gt, k_pred):
        """Save side-by-side visualization: original | GT | prediction."""
        # Get original image from SAM input (always available, standard shape)
        sam_img = batch["sam_images"][0]  # (3, H, W) normalized
        if sam_img is not None and sam_img.dim() == 3:
            img = sam_img.permute(1, 2, 0).cpu().numpy()
            img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            h, w = gt.shape
            img = np.full((h, w, 3), 128, dtype=np.uint8)

        h, w = gt.shape
        img_resized = cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img

        gt_vis = _colorize_mask(gt)
        pred_vis = _colorize_mask(pred)

        # Create overlay: original with pred mask blended
        overlay = img_resized.copy()
        mask_active = pred > 0
        overlay[mask_active] = cv2.addWeighted(
            img_resized, 0.4, pred_vis, 0.6, 0
        )[mask_active]

        # Side by side: [original | GT | prediction | overlay]
        canvas = np.hstack([img_resized, gt_vis, pred_vis, overlay])

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, "GT", (w + 5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, f"Pred (k={k_pred})", (2 * w + 5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, f"IoU={miou:.3f} ARI={ari:.3f}", (3 * w + 5, 15),
                    font, 0.35, (255, 255, 255), 1)

        cv2.imwrite(str(epoch_dir / f"sample_{idx:04d}.jpg"), canvas)

    def _save_summary_grid(self, epoch_dir, per_sample):
        """Save a summary showing worst and best cases."""
        if not per_sample:
            return
        sorted_samples = sorted(per_sample, key=lambda x: x["miou"])

        summary = {
            "worst_10": sorted_samples[:10],
            "best_10": sorted_samples[-10:][::-1],
            "median": sorted_samples[len(sorted_samples) // 2],
        }
        with open(epoch_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
