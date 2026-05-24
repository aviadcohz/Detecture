"""
V5 Training loop for Qwen2SAM-Detecture.

Architecture: [SEG] Token + Bottleneck Projector + Loss Masking.

Stage 1 (epochs 1-N): Projector + Qwen LoRA (SAM LoRA frozen)
  - Qwen generates text with <|seg|> tokens (teacher-forced)
  - LM loss: -100 on ALL text tokens, CE only on <|seg|> positions
  - Mask loss gradients: SAM → Projector → [SEG] hidden states → Qwen LoRA
  - The pixels teach the [SEG] token where to go

Stage 2 (epochs N+1 onwards): + SAM LoRA unfrozen
  - Joint fine-tuning of projector + Qwen LoRA + SAM LoRA
  - Same loss masking

No distillation loss, no InverseProjector, no pre-computed GT embeddings.
Live Qwen forward in EVERY epoch.

Usage:
    cd Qwen2SAM_Detecture
    python -m training.train --config configs/detecture.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detecture import Qwen2SAMDetecture
from models.losses import combined_loss
from data.dataset import DetectureDataset, DetectureCollator
from training.utils import (
    load_config, set_seed, AverageMeter, WarmupCosineScheduler,
    get_lr, save_checkpoint, load_checkpoint,
)
from training.monitor import DataSanityChecker, TrainingLogger, PlotGenerator, TestEvaluator


# ===================================================================== #
#  Curriculum                                                             #
# ===================================================================== #

def _set_qwen_lora_grad(model, requires_grad: bool):
    for n, p in model.qwen.named_parameters():
        if "lora" in n.lower():
            p.requires_grad = requires_grad


def _set_seg_row_grad(model, requires_grad: bool):
    """Toggle training of the <|seg|> token's embed + lm_head rows (the
    per-row gradient mask installed in qwen2sam_detecture restricts actual
    updates to the SEG row, but the full tensor must be requires_grad=True
    for gradients to be computed at all)."""
    rows = getattr(model, "_seg_row_params", None)
    if not rows:
        return
    for p in rows:
        p.requires_grad_(requires_grad)


def decay_bridge_lr(optimizer, scheduler, factor: float) -> tuple:
    """Multiply the bridge param group's base LR (and current LR) by `factor`.

    Mutates both `scheduler.base_lrs` and `pg["lr"]` so the next scheduler.step()
    recomputes the cosine-decayed LR from the new base. Returns (old_base, new_base).
    """
    for i, pg in enumerate(optimizer.param_groups):
        if pg.get("name") == "bridge":
            old_base = scheduler.base_lrs[i]
            scheduler.base_lrs[i] = old_base * factor
            pg["lr"] = pg["lr"] * factor
            return old_base, scheduler.base_lrs[i]
    return None, None


def apply_curriculum(model, epoch, cfg):
    """V7 Two-Stage Curriculum (SAM permanently frozen).

    Stage 1 — Bridge Warmup (epoch < projector_warmup_epochs):
      - Bridge + mask head + dustbin train at full base LR (1e-4).
      - Qwen LoRA FROZEN.

    Stage 2 — Qwen Sync + Bridge Decay (epoch >= projector_warmup_epochs):
      - Qwen LoRA UNFROZEN at base * qwen_lr_scale (= 1e-6).
      - Bridge base LR decayed 10× (1e-4 → 1e-5) via decay_bridge_lr.

    SAM3 stays frozen forever (no Stage 3) — Zero-Shot RWTD at 0.928 confirmed
    SAM natively understands our prompts. Unfreezing it destroys OOD generality.

    Returns (phase, loss_overrides). Bridge LR mutation is handled by
    decay_bridge_lr() invoked once at Stage 2 entry in the main loop.
    """
    curriculum = cfg.get("curriculum", {})
    warmup_epochs = curriculum.get("projector_warmup_epochs", 12)

    if epoch < warmup_epochs:
        _set_qwen_lora_grad(model, False)
        _set_seg_row_grad(model, False)
        return 1, {}
    else:
        _set_qwen_lora_grad(model, True)
        _set_seg_row_grad(model, True)
        return 2, {}


# ===================================================================== #
#  Training epoch                                                         #
# ===================================================================== #

def train_one_epoch(
    model, dataloader, optimizer, scheduler, scaler, epoch, cfg, device,
    logger=None, phase=1, loss_overrides=None,
):
    model.train()
    meters = {
        k: AverageMeter() for k in [
            "total", "mask_ce", "mask_dice", "lm_loss", "orthogonal_reg",
        ]
    }

    accum_steps = cfg["training"]["gradient_accumulation_steps"]
    max_grad_norm = cfg["training"]["max_grad_norm"]

    effective_cfg = {**cfg}
    if loss_overrides:
        effective_cfg["loss"] = {**cfg.get("loss", {}), **loss_overrides}

    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(dataloader):
        # Move to device
        qwen_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["qwen_inputs"].items()
        }
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"].to(device)

        # V6: Live Qwen forward with [SEG] token extraction.
        # seg_grad_to_lm=True when Qwen LoRA is unfrozen (Stages 2-3).
        # In Stage 1 (warmup), gradients stop at projector.
        grad_to_lm = phase in (2, 3)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                qwen_inputs=qwen_inputs,
                sam_images=sam_images,
                seg_grad_to_lm=grad_to_lm,
            )

            mask_logits = out["mask_logits"]
            k_preds = out["k_preds"]
            pad_mask = out["pad_mask"]
            lm_loss = out["lm_loss"]
            qwen_logits = out.get("qwen_logits")

            # V6: pass Qwen logits + per-token weights for proximity-decayed LM loss
            batch_lm_weights = batch.get("lm_weights")
            if batch_lm_weights is not None:
                batch_lm_weights = batch_lm_weights.to(device)
            batch_labels = qwen_inputs.get("labels")

            losses = combined_loss(
                mask_logits=mask_logits,
                gt_masks=index_masks,
                pad_mask=pad_mask,
                k_gts=k_gts,
                lm_loss=lm_loss,
                model=model,
                cfg=effective_cfg,
                qwen_logits=qwen_logits,
                labels=batch_labels,
                lm_weights=batch_lm_weights,
            )

            loss = losses["total"] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        for k in meters:
            if k in losses:
                meters[k].update(losses[k].item())

        lr = get_lr(optimizer)
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            print(
                f"  [S{phase}|{step+1}/{len(dataloader)}] "
                f"loss={meters['total'].avg:.4f} "
                f"ce={meters['mask_ce'].avg:.4f} "
                f"dice={meters['mask_dice'].avg:.4f} "
                f"lm={meters['lm_loss'].avg:.4f} "
                f"orth={meters['orthogonal_reg'].avg:.6f} "
                f"lr={lr:.2e} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

        if logger and ((step + 1) % 10 == 0 or step == 0):
            logger.log_step(
                epoch=epoch + 1,
                step=step + 1,
                total_steps=len(dataloader),
                losses={k: m.val for k, m in meters.items()},
                lr=lr,
            )

    return {k: m.avg for k, m in meters.items()}


# ===================================================================== #
#  Validation                                                             #
# ===================================================================== #

@torch.no_grad()
def validate(model, dataloader, cfg, device):
    """V5 validation with live Qwen forward and [SEG] extraction."""
    model.eval()
    ious = []

    for batch in dataloader:
        qwen_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["qwen_inputs"].items()
        }
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                qwen_inputs=qwen_inputs,
                sam_images=sam_images,
                seg_grad_to_lm=False,
            )

        mask_logits = out["mask_logits"]
        pad_mask = out["pad_mask"]

        masked_logits = mask_logits.clone()
        inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked_logits)
        masked_logits[inf_mask] = float("-inf")

        H_gt, W_gt = index_masks.shape[1], index_masks.shape[2]
        if masked_logits.shape[2] != H_gt or masked_logits.shape[3] != W_gt:
            masked_logits = F.interpolate(
                masked_logits.float(), size=(H_gt, W_gt),
                mode="bilinear", align_corners=False,
            )
        preds = masked_logits.argmax(dim=1)

        B = mask_logits.shape[0]
        for b in range(B):
            k = int(k_gts[b].item())
            sample_ious = []
            for c in range(1, k + 1):
                pred_c = (preds[b] == c)
                gt_c = (index_masks[b] == c)
                inter = (pred_c & gt_c).sum().float()
                union = (pred_c | gt_c).sum().float()
                iou = (inter / union.clamp(min=1)).item()
                sample_ious.append(iou)
            if sample_ious:
                ious.append(sum(sample_ious) / len(sample_ious))

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return mean_iou


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/detecture.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--resume-lr-scale", type=float, default=None,
        help="Micro-Warmup Restart on resume: rebuild the LR scheduler with "
             "each param group's peak set to (original base LR * scale), "
             "preserving per-group ratios (Bridge vs Qwen-LoRA vs SEG rows). "
             "A short warmup ramps to that scaled peak, then cosine decays "
             "over the REMAINING epochs (num_epochs - start_epoch). "
             "Typical value: 0.15.",
    )
    parser.add_argument(
        "--resume-warmup-epochs", type=int, default=2,
        help="Warmup length (epochs) for the Micro-Warmup Restart. Only "
             "effective when --resume-lr-scale is set.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Monitoring ---------------------------------------------------- #
    train_cfg = cfg["training"]
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    monitor_cfg = cfg.get("monitor", {})

    log_dir = monitor_cfg.get("log_dir", str(ckpt_dir / "logs"))
    plot_dir = monitor_cfg.get("plot_dir", str(ckpt_dir / "plots"))
    logger = TrainingLogger(log_dir)

    baselines = {}
    baseline_path = monitor_cfg.get("baseline_results")
    if baseline_path and Path(baseline_path).exists():
        import json as _json
        with open(baseline_path) as f:
            bl_data = _json.load(f)
        for approach, metrics in bl_data.items():
            baselines[approach] = {
                "miou": metrics.get("mean_iou", 0),
                "mari": metrics.get("mean_ari", 0),
            }
        print(f"Loaded {len(baselines)} baselines from {baseline_path}")
    plotter = PlotGenerator(plot_dir, baselines=baselines)

    test_evaluator = None
    test_metadata = monitor_cfg.get("test_metadata")
    if test_metadata and Path(test_metadata).exists():
        test_eval_every = monitor_cfg.get("test_eval_every", 5)
        test_output_dir = monitor_cfg.get("test_output_dir",
                                           str(ckpt_dir / "test_results"))
        test_evaluator = TestEvaluator(
            test_metadata=test_metadata,
            output_dir=test_output_dir,
            image_size=cfg["data"].get("image_size", 1008),
            eval_every=test_eval_every,
        )

    # ---- Model --------------------------------------------------------- #
    print("Building Qwen2SAM-Detecture V7 model...")
    model = Qwen2SAMDetecture(cfg, device=str(device))
    params = model.num_trainable_params()
    print(f"Trainable parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # ---- Data ---------------------------------------------------------- #
    data_cfg = cfg["data"]

    train_ds = DetectureDataset(
        data_cfg["train_metadata"],
        image_size=data_cfg.get("image_size", 1008),
        augment=data_cfg.get("augment", True),
        qwen_gt_embeds_path=data_cfg.get("qwen_gt_embeds_path"),
    )
    val_ds = DetectureDataset(
        data_cfg["val_metadata"],
        image_size=data_cfg.get("image_size", 1008),
        augment=False,
        qwen_gt_embeds_path=data_cfg.get("qwen_gt_embeds_path"),
    )

    collator = DetectureCollator(model.processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collator,
    )

    # ---- Sanity check -------------------------------------------------- #
    sanity_checker = DataSanityChecker(
        output_dir=str(ckpt_dir / "sanity"),
        n_batches=monitor_cfg.get("sanity_check_batches", 5),
    )
    sanity_ok = sanity_checker.check(train_loader, train_ds)
    if not sanity_ok and not monitor_cfg.get("skip_sanity_check", False):
        print("CRITICAL: Data sanity check failed.")
        sys.exit(1)

    # ---- Optimizer + scheduler ----------------------------------------- #
    curriculum = cfg.get("curriculum", {})
    alignment_epochs = curriculum.get("alignment_epochs", 15)

    steps_per_epoch = len(train_loader) // train_cfg["gradient_accumulation_steps"]
    param_groups = model.get_parameter_groups(train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=train_cfg["num_epochs"],
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=max(steps_per_epoch, 1),
    )
    # GradScaler is a fp16 feature — bf16 doesn't need loss scaling and
    # `_unscale_grads_` isn't implemented for bf16 gradients, so keep the
    # scaler disabled. All scaler.* calls in train_one_epoch become no-ops;
    # scaler.scale(loss) returns loss unchanged, scaler.step() just calls
    # optimizer.step(), scaler.update() / scaler.unscale_() are no-ops.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ---- Resume -------------------------------------------------------- #
    start_epoch = 0
    best_iou = 0.0
    stage_initialized = {1: False, 2: False}

    resume_path = args.resume
    if resume_path == "auto":
        ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
        if ckpt_files:
            resume_path = str(ckpt_files[-1])
        elif (ckpt_dir / "best.pt").exists():
            resume_path = str(ckpt_dir / "best.pt")
        else:
            resume_path = None

    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        start_epoch = load_checkpoint(model, optimizer, resume_path, device=str(device))
        start_epoch += 1
        best_iou = state.get("val_iou", 0.0)

        if args.resume_lr_scale is not None:
            # Micro-Warmup Restart: rebuild the scheduler around a scaled peak
            # LR over the REMAINING epochs. Preserves per-group ratios, keeps
            # optimizer moments, and avoids shocking weights that are already
            # in a good basin.
            scale = float(args.resume_lr_scale)
            remaining = train_cfg["num_epochs"] - start_epoch
            if remaining <= 0:
                raise RuntimeError(
                    f"--resume-lr-scale requires remaining epochs > 0 "
                    f"(start_epoch={start_epoch}, num_epochs={train_cfg['num_epochs']}). "
                    f"Bump num_epochs in the config."
                )

            # Recover each group's original peak LR by re-invoking
            # get_parameter_groups with the config base LR. Match by name
            # (with positional fallback for any unnamed groups).
            original_groups = model.get_parameter_groups(train_cfg["learning_rate"])
            name_to_lr = {g["name"]: g["lr"] for g in original_groups if "name" in g}
            for i, pg in enumerate(optimizer.param_groups):
                name = pg.get("name")
                orig_lr = name_to_lr.get(name)
                if orig_lr is None and i < len(original_groups):
                    orig_lr = original_groups[i]["lr"]
                if orig_lr is None:
                    orig_lr = pg["lr"]
                pg["lr"] = orig_lr * scale

            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=args.resume_warmup_epochs,
                total_epochs=remaining,
                min_lr=train_cfg.get("min_lr", 1e-6),
                steps_per_epoch=max(steps_per_epoch, 1),
            )
            # The scale already encodes the Stage-2 Bridge decay, so suppress
            # decay_bridge_lr() firing again at the first post-resume epoch.
            stage_initialized[2] = True

            print(f"  Micro-Warmup Restart engaged:")
            print(f"    LR scale          : {scale}")
            print(f"    Warmup            : {args.resume_warmup_epochs} epochs")
            print(f"    Cosine decay over : {remaining} remaining epochs")
            print(f"    Per-group peak LRs (scaled):")
            for pg in optimizer.param_groups:
                nm = pg.get("name", "unnamed")
                print(f"      {nm:<12s} peak = {pg['lr']:.2e}")
        else:
            total_steps_done = start_epoch * max(steps_per_epoch, 1)
            for _ in range(total_steps_done):
                scheduler.step()

        print(f"  Resumed at epoch {start_epoch + 1}, best_iou={best_iou:.4f}")

    # ---- Training loop ------------------------------------------------- #

    warmup_ep = curriculum.get("projector_warmup_epochs", 12)
    bridge_decay_s2 = curriculum.get("projector_lr_decay_at_stage2", 0.1)

    print(f"\nStarting V7 Two-Stage Curriculum: epochs {start_epoch+1}-{train_cfg['num_epochs']}")
    print(f"  {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"  Effective batch: {train_cfg['batch_size']} × "
          f"{train_cfg['gradient_accumulation_steps']} = "
          f"{train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Stage 1: Bridge @ base LR (epochs 1-{warmup_ep})")
    print(f"  Stage 2: + Qwen LoRA; Bridge base LR ×{bridge_decay_s2} "
          f"(epochs {warmup_ep+1}-{train_cfg['num_epochs']})")
    print(f"  SAM3 frozen permanently (no Stage 3).")
    print(f"  Exponential LM weights active (α=2.0).")

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        phase, loss_overrides = apply_curriculum(model, epoch, cfg)

        if not stage_initialized[phase]:
            stage_initialized[phase] = True
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if phase == 1:
                print(f"\n  *** STAGE 1: Bridge Warmup (epoch {epoch+1}) ***")
                print(f"  Qwen LoRA FROZEN. SAM frozen. Only Bridge + mask head trains.")
            elif phase == 2:
                old_base, new_base = decay_bridge_lr(optimizer, scheduler, bridge_decay_s2)
                print(f"\n  *** STAGE 2: Qwen Sync + Bridge Decay (epoch {epoch+1}) ***")
                print(f"  Qwen LoRA UNFROZEN (base × qwen_lr_scale).")
                if old_base is not None:
                    print(f"  Bridge base LR DECAYED: {old_base:.1e} → {new_base:.1e} ({bridge_decay_s2}x)")
            print(f"  Trainable params: {n_trainable:,}")

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{train_cfg['num_epochs']} [Stage {phase}]")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, cfg, device,
            logger=logger, phase=phase, loss_overrides=loss_overrides,
        )

        val_iou = validate(model, val_loader, cfg, device)
        print(f"\n  Val mIoU: {val_iou:.4f}")

        lr = get_lr(optimizer)
        is_best = val_iou > best_iou
        logger.log_epoch(
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_miou=val_iou,
            lr=lr,
            is_best=is_best,
            extra={"phase": phase},
        )

        if is_best:
            best_iou = val_iou
            save_checkpoint(
                model, optimizer, epoch,
                str(ckpt_dir / "best.pt"),
                extra={"val_iou": val_iou, "phase": phase},
            )
            print(f"  ** New best: {best_iou:.4f} **")

        if (epoch + 1) % train_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                model, optimizer, epoch,
                str(ckpt_dir / f"epoch_{epoch+1}.pt"),
                extra={"val_iou": val_iou, "phase": phase},
            )

        if test_evaluator and test_evaluator.should_evaluate(epoch):
            print(f"\n  Running test evaluation (RWTD)...")
            try:
                test_metrics = test_evaluator.evaluate(
                    model, model.processor, device, epoch,
                )
                logger.log_test(epoch=epoch + 1, metrics=test_metrics)
            except Exception as e:
                print(f"  WARNING: Test evaluation failed: {e}")

        plotter.update(logger)

    logger.close()
    print(f"\nTraining complete. Best val mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
