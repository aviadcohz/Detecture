#!/usr/bin/env python3
"""
Offline Oracle evaluator for monitoring the V7 Bridge during Stage 1.

Bypasses Qwen generation entirely. For each RWTD sample we:
  1. Load perfect, descriptive ground-truth text from a JSON file (produced
     by a ZS SAM 3 experiment) and build a teacher-forced assistant text
     of the form "TEXTURE_1: <desc_a> <|seg|>\\nTEXTURE_2: <desc_b> <|seg|>".
  2. Run the standard training forward() through Qwen (teacher-forced, with
     the block-diagonal attention mask and assistant-region <|seg|> filter),
     extract the two <|seg|> hidden states.
  3. Pass them through the Bridge (trainable) and SAM 3's frozen resizer
     (1024 -> 256), build the 7-slot query (DUSTBIN + 2 texture + 4 PAD),
     run SAM 3 to produce masks.
  4. Compute per-sample mIoU vs the GT binary masks with Hungarian matching.

Why this script exists:
  During V7 Stage 1 (epochs 1-12) Qwen's LoRA is frozen, so it will not
  emit <|seg|> during live generation. The training-loop TestEvaluator
  therefore returns ~0 mIoU on RWTD in Stage 1 (unhelpful). This script
  provides a teacher-forced probe of the Bridge's geometric learning,
  analogous to the V5 "Oracle" metric.

GPU:
  The script allocates its own copy of Qwen3-VL-8B + SAM 3 on GPU. If a
  training run is active on the same GPU the script will contend for
  memory and compute. Plan accordingly.

Usage:
    python scripts/evaluate_bridge_oracle.py \\
        --checkpoint checkpoints/epoch_4.pt \\
        --gt-json RWTD_GT.json \\
        --output-json checkpoints/bridge_oracle_epoch_4.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detecture import Qwen2SAMDetecture, MAX_TEXTURES, SEG_TOKEN
from data.dataset import (
    DetectureCollator, SYSTEM_PROMPT, TRAIN_USER_PROMPT, build_assistant_text,
    preprocess_image_for_sam3, SAM3_SIZE,
)
from training.utils import load_config, load_checkpoint
from PIL import Image


RWTD_IMAGES_DIR = Path("~/datasets/RWTD/images")
RWTD_TEXTURES_MASK_DIR = Path("~/datasets/RWTD/textures_mask")


# --------------------------------------------------------------------- #
#  Data loading                                                           #
# --------------------------------------------------------------------- #

def load_rwtd_sample(crop_name: str, image_size: int = SAM3_SIZE):
    """Load image + binary masks for one RWTD sample.

    Returns:
        image_pil: PIL image (RGB, image_size x image_size)
        sam_image: preprocessed tensor (3, image_size, image_size)
        index_mask: np.int64 (H_gt, W_gt) with values in {0, 1, 2}
                    where 1 = texture A, 2 = texture B
    """
    img_path = RWTD_IMAGES_DIR / f"{crop_name}.jpg"
    mask_a_path = RWTD_TEXTURES_MASK_DIR / f"{crop_name}_mask_a.png"
    mask_b_path = RWTD_TEXTURES_MASK_DIR / f"{crop_name}_mask_b.png"

    for p in (img_path, mask_a_path, mask_b_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    # Image -> RGB, resized to image_size for SAM and PIL pathways
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (image_size, image_size),
                         interpolation=cv2.INTER_LINEAR)
    image_pil = Image.fromarray(img_rgb)
    sam_image = preprocess_image_for_sam3(img_rgb, image_size)

    # GT masks kept at their native resolution for scoring.
    mask_a = cv2.imread(str(mask_a_path), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(str(mask_b_path), cv2.IMREAD_GRAYSCALE)
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    H, W = mask_a.shape
    index_mask = np.zeros((H, W), dtype=np.int64)
    index_mask[mask_a > 127] = 1
    # If any pixel is labelled in both (shouldn't happen for RWTD), texture B wins.
    index_mask[mask_b > 127] = 2

    return image_pil, sam_image, index_mask


# --------------------------------------------------------------------- #
#  Metric                                                                 #
# --------------------------------------------------------------------- #

def hungarian_miou(pred: np.ndarray, gt: np.ndarray,
                   k_pred: int, k_gt: int) -> float:
    if k_pred == 0 or k_gt == 0:
        return 0.0
    cost = np.zeros((k_pred, k_gt))
    for pi in range(k_pred):
        pred_c = (pred == pi + 1)
        for gi in range(k_gt):
            gt_c = (gt == gi + 1)
            inter = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()
            cost[pi, gi] = 1.0 - inter / max(union, 1)
    r, c = linear_sum_assignment(cost)
    ious = [1.0 - cost[ri, ci] for ri, ci in zip(r, c)
            if ri < k_pred and ci < k_gt]
    return float(np.mean(ious)) if ious else 0.0


# --------------------------------------------------------------------- #
#  Oracle forward                                                         #
# --------------------------------------------------------------------- #

@torch.no_grad()
def oracle_forward_one(
    model, processor, sample_batch, device,
) -> tuple:
    """Run the teacher-forced forward on one batched sample and return
    (pred_index_mask_at_gt_resolution, k_pred)."""
    qwen_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample_batch["qwen_inputs"].items()
    }
    sam_images = sample_batch["sam_images"].to(device)
    gt_index = sample_batch["index_masks"]  # (1, H_gt, W_gt) CPU

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward(
            qwen_inputs=qwen_inputs,
            sam_images=sam_images,
            seg_grad_to_lm=False,
        )

    mask_logits = out["mask_logits"]   # (1, 7, H, W)
    pad_mask = out["pad_mask"]
    k_pred = int(out["k_preds"][0].item())

    masked = mask_logits.clone()
    inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
    masked[inf_mask] = float("-inf")

    H_gt, W_gt = gt_index.shape[1], gt_index.shape[2]
    if masked.shape[2] != H_gt or masked.shape[3] != W_gt:
        masked = F.interpolate(
            masked.float(), size=(H_gt, W_gt),
            mode="bilinear", align_corners=False,
        )
    pred = masked[0].argmax(dim=0).cpu().numpy()  # (H_gt, W_gt), values in {0..6}
    return pred, k_pred


# --------------------------------------------------------------------- #
#  Main                                                                   #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detecture.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to V7 checkpoint (e.g. checkpoints/epoch_4.pt)")
    parser.add_argument(
        "--gt-json",
        default=str(_PROJECT_ROOT / "RWTD_GT.json"),
        help="JSON file with {crop_name, parsed:{desc_a, desc_b}} per sample. "
             "Default: RWTD_GT.json in the project root.",
    )
    parser.add_argument("--output-json", default=None,
                        help="Where to save per-sample results. Defaults to "
                             "<checkpoint-stem>_bridge_oracle.json next to the "
                             "checkpoint.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: evaluate only the first N samples (debugging)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output_json) if args.output_json else \
        ckpt_path.with_name(f"{ckpt_path.stem}_bridge_oracle.json")

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    cfg = load_config(args.config)
    device = torch.device("cuda")

    # Warn about GPU contention
    try:
        free, total = torch.cuda.mem_get_info()
        free_gb, total_gb = free / 2**30, total / 2**30
        print(f"GPU memory: {free_gb:.1f} / {total_gb:.1f} GB free")
        if free_gb < 20:
            print(f"  WARNING: low free GPU memory — training may be running. "
                  f"Script will still try; OOM is possible.")
    except Exception:
        pass

    print(f"Loading model...")
    model = Qwen2SAMDetecture(cfg, device=str(device))
    print(f"Loading checkpoint: {ckpt_path}")
    epoch_idx = load_checkpoint(model, None, str(ckpt_path), device=str(device))
    print(f"  Resumed at epoch index {epoch_idx} (displayed Epoch {epoch_idx + 1})")
    model.eval()

    collator = DetectureCollator(model.processor, inference=False)

    # ------------------------------------------------------------------ #
    # Load GT text
    # ------------------------------------------------------------------ #
    with open(args.gt_json) as f:
        gt_items = json.load(f)

    # Filter to samples with parse_ok and a valid desc_a/desc_b pair,
    # and whose images + binary masks exist on disk.
    valid = []
    for it in gt_items:
        if not it.get("parse_ok", False):
            continue
        parsed = it.get("parsed", {})
        da, db = parsed.get("desc_a"), parsed.get("desc_b")
        if not (da and db):
            continue
        crop = it["crop_name"]
        if not (RWTD_IMAGES_DIR / f"{crop}.jpg").exists():
            continue
        if not (RWTD_TEXTURES_MASK_DIR / f"{crop}_mask_a.png").exists():
            continue
        if not (RWTD_TEXTURES_MASK_DIR / f"{crop}_mask_b.png").exists():
            continue
        valid.append(it)

    if args.limit:
        valid = valid[: args.limit]
    print(f"Evaluating {len(valid)} samples (skipped "
          f"{len(gt_items) - len(valid)} with missing data or failed parse).")

    # ------------------------------------------------------------------ #
    # Evaluate
    # ------------------------------------------------------------------ #
    per_sample = []
    mious = []
    t0 = time.time()

    for idx, it in enumerate(valid):
        crop = it["crop_name"]
        da = it["parsed"]["desc_a"]
        db = it["parsed"]["desc_b"]

        try:
            image_pil, sam_image, gt_index = load_rwtd_sample(
                crop, image_size=cfg["data"].get("image_size", SAM3_SIZE),
            )
        except FileNotFoundError as e:
            print(f"  [{idx}] {crop}: SKIP — {e}")
            continue

        # Build the teacher-forced assistant text:
        #   "TEXTURE_1: {desc_a} <|seg|>\nTEXTURE_2: {desc_b} <|seg|>"
        assistant_text = build_assistant_text([da, db])

        # Convert to collator's per-sample dict format.
        # We pass a zero tensor for qwen_gt_embeds (unused in teacher-forced forward).
        sample = {
            "image": image_pil,
            "assistant_text": assistant_text,
            "sam_image": sam_image,
            "index_mask": torch.from_numpy(gt_index),
            "k_gt": 2,
            "descriptions": [da, db],
            "qwen_gt_embeds": torch.zeros(MAX_TEXTURES, 4096),
        }
        batch = collator([sample])

        try:
            pred, k_pred = oracle_forward_one(model, model.processor, batch, device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [{idx}] {crop}: OOM — aborting. "
                  f"Stop training and retry if you need this sample.")
            torch.cuda.empty_cache()
            break

        miou = hungarian_miou(pred, gt_index, k_pred=k_pred, k_gt=2)
        mious.append(miou)

        per_sample.append({
            "idx": idx,
            "crop_name": crop,
            "k_pred": k_pred,
            "k_gt": 2,
            "miou": miou,
            "assistant_text": assistant_text,
        })

        running = float(np.mean(mious))
        if (idx + 1) % 10 == 0 or idx < 3:
            elapsed = time.time() - t0
            print(f"  [{idx+1:3d}/{len(valid)}] {crop:<10s} "
                  f"k_pred={k_pred}  mIoU={miou:.3f}  "
                  f"running_avg={running:.4f}  ({elapsed:.0f}s)")

    mean_iou = float(np.mean(mious)) if mious else 0.0

    # ------------------------------------------------------------------ #
    # Report + save
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  BRIDGE ORACLE RESULT @ {ckpt_path.name}")
    print(f"{'='*60}")
    print(f"  Samples:        {len(mious)}")
    print(f"  Mean mIoU:      {mean_iou:.4f}")
    if mious:
        arr = np.array(mious)
        print(f"  Median mIoU:    {float(np.median(arr)):.4f}")
        print(f"  Samples > 0.7:  {(arr > 0.7).sum()}/{len(arr)}")
        print(f"  Samples < 0.3:  {(arr < 0.3).sum()}/{len(arr)}")
    print(f"  Time:           {(time.time() - t0) / 60:.1f} min")

    results = {
        "checkpoint": str(ckpt_path),
        "epoch_index": epoch_idx,
        "displayed_epoch": epoch_idx + 1,
        "gt_json": args.gt_json,
        "n_samples": len(per_sample),
        "mean_iou": mean_iou,
        "median_iou": float(np.median(mious)) if mious else None,
        "per_sample": per_sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
