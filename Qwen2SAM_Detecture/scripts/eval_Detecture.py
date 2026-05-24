#!/usr/bin/env python3
"""
Multi-dataset E2E evaluation of a trained Detecture checkpoint.

Runs the same inference pipeline that `ablation_exact_k2_rwtd.py` uses for
RWTD, but per-dataset: each dataset gets the prompt framing the Zero-Shot
benchmark uses in `evaluate_zero_shot_pipeline.py::build_user_prompt`,
extended with the Detecture-mandatory `<|seg|>` token at the end of each
TEXTURE_i line so the model emits a grounding anchor we can extract.

Dataset → K-rule + prompt:
  RWTD              → "exactly 2"  (foreground / background wording)
  STLD              → "exactly 2"  (foreground / background wording)
  ADE20k_Detecture  → "between 1 and 6"   (open-range prompt)
  CAID              → "exactly 1"  (water-surface wording)

The validation gate is RWTD first: with `best.pt`, the expected honest
mean_iou is ~0.813 (matches `ablation_exact_k2_rwtd.py` condition B).

Usage:
    python scripts/eval_Detecture.py \
        --checkpoint checkpoints/best.pt \
        --config configs/detecture.yaml

Outputs (under `checkpoints/test_results/eval_Detecture_<ckpt_stem>/`):
    <dataset>.json              per-dataset result with per-sample rows
    summary.json                combined mIoU / mARI / k_dist per dataset
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detecture import Qwen2SAMDetecture, MAX_TEXTURES
from data.dataset import (
    DetectureDataset, DetectureCollator,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
)
from training.utils import load_config, load_checkpoint
from training.monitor import _compute_ari
from scipy.optimize import linear_sum_assignment


# ===================================================================== #
#  Per-dataset prompt library                                             #
# ===================================================================== #
# IMPORTANT: We use the TRAINING prompt template USER_PROMPT_TEMPLATE
# verbatim for every dataset, substituting only the count N. LoRA adapters
# are prompt-distribution-sensitive; any wording drift from the training
# template shifts the [SEG] embedding distribution and hurts mIoU. The
# reference script `ablation_exact_k2_rwtd.py` uses exactly this pattern
# with N="2" to produce the 0.813 number on RWTD with best.pt.

def build_dataset_config(name: str, root: Path) -> dict:
    """Return the (prompt, metadata_path) pair for each dataset.

    Prompt is always the training template with N substituted:
      RWTD, STLD        -> N="2"              (two textures per image)
      CAID              -> N="1"              (single water surface)
      ADE20k_Detecture  -> N="between 1 and 6" (open range; matches how
                                                 other ZS models are
                                                 prompted on this set)
    """
    if name == "RWTD":
        return {
            "prompt": USER_PROMPT_TEMPLATE.format(N="2"),
            "metadata": root / "RWTD" / "metadata.json",
            "label": "RWTD (N=2)",
        }
    if name == "STLD":
        return {
            "prompt": USER_PROMPT_TEMPLATE.format(N="2"),
            "metadata": root / "STLD" / "metadata.json",
            "label": "STLD (N=2)",
        }
    if name == "ADE20k_Detecture":
        return {
            "prompt": USER_PROMPT_TEMPLATE.format(N="between 1 and 6"),
            "metadata": root / "ADE20k_Detecture" / "metadata.json",
            "label": "ADE20k_Detecture (N=1..6)",
        }
    if name == "CAID":
        return {
            "prompt": USER_PROMPT_TEMPLATE.format(N="1"),
            "metadata": root / "CAID" / "metadata.json",
            "label": "CAID (N=1)",
        }
    raise ValueError(f"Unknown dataset: {name}")


# ===================================================================== #
#  Collator — variant of DetectureCollator with a custom user prompt      #
# ===================================================================== #

class PromptCollator(DetectureCollator):
    """DetectureCollator with the user prompt replaced by an arbitrary string.

    Mirrors `ExactKCollator` from ablation_exact_k2_rwtd.py but parameterised
    by the full prompt string instead of a count.
    """
    def __init__(self, processor, user_prompt: str):
        super().__init__(processor, inference=True)
        self.user_prompt = user_prompt

    def __call__(self, samples):
        texts, images = [], []
        for s in samples:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.user_prompt},
                ]},
            ]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            images.append(s["image"])

        qwen_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )
        qwen_inputs.pop("token_type_ids", None)
        return {
            "qwen_inputs": qwen_inputs,
            "sam_images": torch.stack([s["sam_image"] for s in samples]),
            "index_masks": torch.stack([s["index_mask"] for s in samples]),
            "k_gts": torch.tensor([s["k_gt"] for s in samples], dtype=torch.long),
            "qwen_gt_embeds": torch.stack([s["qwen_gt_embeds"] for s in samples]),
        }


# ===================================================================== #
#  Metrics (verbatim from ablation_exact_k2_rwtd.py)                      #
# ===================================================================== #

def compute_miou_hungarian(pred, gt, k_pred, k_gt):
    if k_pred == 0 or k_gt == 0:
        return 0.0
    cost = np.zeros((k_pred, k_gt))
    for pi in range(k_pred):
        for gi in range(k_gt):
            inter = ((pred == pi + 1) & (gt == gi + 1)).sum()
            union = ((pred == pi + 1) | (gt == gi + 1)).sum()
            cost[pi, gi] = 1.0 - inter / max(union, 1)
    r, c = linear_sum_assignment(cost)
    ious = [1.0 - cost[ri, ci] for ri, ci in zip(r, c) if ri < k_pred and ci < k_gt]
    return np.mean(ious) if ious else 0.0


# ===================================================================== #
#  Evaluation loop (one dataset)                                          #
# ===================================================================== #

def evaluate(model, loader, device, label):
    model.eval()
    ious, aris, k_preds_list, per_sample = [], [], [], []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            qwen_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["qwen_inputs"].items()
            }
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference_forward(
                    qwen_inputs=qwen_inputs,
                    sam_images=sam_images,
                )

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())
            gen_text = out.get("generated_text", [""])[0]
            k_preds_list.append(k_pred)

            masked = mask_logits.clone()
            inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
            masked[inf_mask] = float("-inf")
            H, W = index_masks.shape[1], index_masks.shape[2]
            if masked.shape[2] != H:
                masked = F.interpolate(
                    masked.float(), size=(H, W),
                    mode="bilinear", align_corners=False,
                )
            pred = masked[0].argmax(dim=0).cpu().numpy()
            gt = index_masks[0].numpy()

            miou = compute_miou_hungarian(pred, gt, k_pred, k_gt)
            ari = _compute_ari(pred, gt)
            ious.append(miou)
            aris.append(ari)

            per_sample.append({
                "idx": idx,
                "k_gt": k_gt,
                "k_pred": k_pred,
                "miou": float(miou),
                "ari": float(ari),
                "generated_text": gen_text,
            })

            if (idx + 1) % 50 == 0:
                print(f"    [{label}] {idx+1}/{len(loader)} "
                      f"avg mIoU={np.mean(ious):.4f} "
                      f"mARI={np.mean(aris):.4f}", flush=True)

    mean_iou = float(np.mean(ious))
    mean_ari = float(np.mean(aris))
    k_dist = Counter(k_preds_list)
    good = sum(1 for x in ious if x > 0.7)
    bad = sum(1 for x in ious if x < 0.3)

    print(f"\n  [{label}] RESULTS:")
    print(f"    mIoU: {mean_iou:.4f}")
    print(f"    mARI: {mean_ari:.4f}")
    print(f"    k_pred distribution: {dict(sorted(k_dist.items()))}")
    print(f"    Samples > 0.7: {good}/{len(ious)}   < 0.3: {bad}/{len(ious)}")

    return {
        "label": label,
        "mean_iou": mean_iou,
        "mean_ari": mean_ari,
        "k_dist": {int(k): int(v) for k, v in k_dist.items()},
        "n_samples": len(per_sample),
        "per_sample": per_sample,
    }


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detecture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--datasets-root", default="~/datasets")
    parser.add_argument("--output-dir", default=None,
                        help="Default: checkpoints/test_results/eval_Detecture_<ckpt_stem>/")
    parser.add_argument("--datasets", nargs="+",
                        default=["RWTD", "STLD", "ADE20k_Detecture", "CAID"],
                        help="Order matters. RWTD first validates ~0.813 gate.")
    parser.add_argument("--rwtd-gate", type=float, default=0.80,
                        help="Abort after RWTD if mean_iou < this "
                             "(best.pt expected ~0.813; default 0.80 "
                             "leaves headroom).")
    args = parser.parse_args()

    ckpt_stem = Path(args.checkpoint).stem
    out_dir = Path(args.output_dir or
                   f"checkpoints/test_results/eval_Detecture_{ckpt_stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Datasets (in order): {args.datasets}")

    cfg = load_config(args.config)
    device = torch.device("cuda")

    print("\nBuilding model...")
    model = Qwen2SAMDetecture(cfg, device="cuda")
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    root = Path(args.datasets_root)
    all_results = {}

    for i, ds_name in enumerate(args.datasets):
        ds_cfg = build_dataset_config(ds_name, root)
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(args.datasets)}]  {ds_cfg['label']}")
        print(f"  metadata: {ds_cfg['metadata']}")
        print(f"{'='*70}")

        if not ds_cfg["metadata"].exists():
            print(f"  SKIP — metadata file not found.")
            all_results[ds_name] = {"status": "metadata missing"}
            continue

        test_ds = DetectureDataset(
            str(ds_cfg["metadata"]),
            image_size=cfg["data"].get("image_size", 1008),
            augment=False,
            qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
        )
        print(f"  samples: {len(test_ds)}")

        collator = PromptCollator(model.processor, ds_cfg["prompt"])
        loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collator,
        )

        result = evaluate(model, loader, device, ds_cfg["label"])
        result["checkpoint"] = str(args.checkpoint)
        result["dataset"] = ds_name
        result["prompt"] = ds_cfg["prompt"]

        with open(out_dir / f"{ds_name}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  written: {out_dir / f'{ds_name}.json'}")

        all_results[ds_name] = {
            "mean_iou": result["mean_iou"],
            "mean_ari": result["mean_ari"],
            "k_dist":   result["k_dist"],
            "n_samples": result["n_samples"],
        }

        # ---- Validation gate after RWTD --------------------------------- #
        if i == 0 and ds_name == "RWTD":
            gate_ok = result["mean_iou"] >= args.rwtd_gate
            tag = "PASS" if gate_ok else "FAIL"
            print(f"\n  RWTD validation gate: {result['mean_iou']:.4f} "
                  f"vs threshold {args.rwtd_gate:.2f}  ->  {tag}")
            if not gate_ok:
                print("  Aborting: RWTD below gate (expected ~0.813 for "
                      "best.pt). Review prompt / collator / checkpoint "
                      "before proceeding to remaining datasets.")
                # Still write the summary of what we have so it's preserved.
                _write_summary(out_dir, args.checkpoint, all_results)
                sys.exit(1)

    _write_summary(out_dir, args.checkpoint, all_results)


def _write_summary(out_dir: Path, checkpoint: str, results: dict):
    """Write the combined summary.json + print the final table."""
    summary = {"checkpoint": str(checkpoint), "datasets": results}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY ({checkpoint})")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20} {'mIoU':>8}  {'mARI':>8}  {'N':>5}  {'k_dist':<30}")
    print(f"  {'-'*70}")
    for ds_name, r in results.items():
        if "mean_iou" not in r:
            print(f"  {ds_name:<20} {'—':>8}  {'—':>8}       {r.get('status', '?')}")
            continue
        kd = " ".join(f"{k}:{v}" for k, v in sorted(r["k_dist"].items()))
        print(f"  {ds_name:<20} {r['mean_iou']:>8.4f}  "
              f"{r['mean_ari']:>8.4f}  {r['n_samples']:>5}  {kd}")
    print(f"\n  Results written to: {out_dir}")


if __name__ == "__main__":
    main()
