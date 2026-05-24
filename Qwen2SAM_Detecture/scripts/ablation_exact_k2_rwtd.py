#!/usr/bin/env python3
"""
Proof of concept: "exactly 2" prompt vs "1 to 6" prompt on RWTD.

Hypothesis: Over-segmentation (k_pred=6 when k_gt=2) is the primary
cause of mIoU degradation on RWTD. By telling Qwen "exactly 2", we
eliminate 4 garbage channels and let the 2 real textures dominate.

Usage:
    python scripts/ablation_exact_k2_rwtd.py \
        --checkpoint checkpoints/epoch_5.pt
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
#  Custom collator with "exactly 2" prompt                                #
# ===================================================================== #

class ExactKCollator(DetectureCollator):
    """Collator that uses 'exactly K' in the user prompt instead of '1 to 6'."""

    def __init__(self, processor, k: int = 2):
        super().__init__(processor, inference=True)
        self.k = k
        self.user_prompt = USER_PROMPT_TEMPLATE.format(N=str(k))

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


def evaluate(model, loader, device, label):
    model.eval()
    ious = []
    aris = []
    k_preds_list = []
    per_sample = []

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
                print(f"    {idx+1}/{len(loader)} avg mIoU={np.mean(ious):.4f} "
                      f"mARI={np.mean(aris):.4f}", flush=True)

    mean_iou = float(np.mean(ious))
    mean_ari = float(np.mean(aris))
    k_dist = Counter(k_preds_list)

    print(f"\n  [{label}] RESULTS:")
    print(f"    mIoU:     {mean_iou:.4f}")
    print(f"    mARI:     {mean_ari:.4f}")
    print(f"    k_pred distribution: {dict(sorted(k_dist.items()))}")
    print(f"    Samples > 0.7: {sum(1 for x in ious if x > 0.7)}/{len(ious)}")
    print(f"    Samples < 0.3: {sum(1 for x in ious if x < 0.3)}/{len(ious)}")

    return {
        "label": label,
        "mean_iou": mean_iou,
        "mean_ari": mean_ari,
        "k_dist": {int(k): int(v) for k, v in k_dist.items()},
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detecture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_5.pt")
    parser.add_argument("--test-metadata",
                        default="~/datasets/RWTD/metadata.json")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save per-condition results JSON. "
                             "Defaults to checkpoints/ablation_<ckpt_stem>/")
    args = parser.parse_args()

    ckpt_stem = Path(args.checkpoint).stem
    output_dir = Path(args.output_dir or f"checkpoints/ablation_{ckpt_stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    cfg = load_config(args.config)
    device = torch.device("cuda")

    print("Building model...")
    model = Qwen2SAMDetecture(cfg, device="cuda")
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    test_ds = DetectureDataset(
        args.test_metadata,
        image_size=cfg["data"].get("image_size", 1008),
        augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    print(f"Test dataset: {len(test_ds)} samples")

    # ---- Condition A: "1 to 6" (current default) ----------------------- #
    print(f"\n{'='*70}")
    print(f"  CONDITION A: prompt='1 to 6' (current default)")
    print(f"{'='*70}")
    collator_1to6 = DetectureCollator(model.processor, inference=True)
    loader_1to6 = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collator_1to6,
    )
    res_1to6 = evaluate(model, loader_1to6, device, "1_to_6")
    res_1to6["checkpoint"] = str(args.checkpoint)
    with open(output_dir / "condition_A_1_to_6.json", "w") as f:
        json.dump(res_1to6, f, indent=2)

    # ---- Condition B: "exactly 2" -------------------------------------- #
    print(f"\n{'='*70}")
    print(f"  CONDITION B: prompt='exactly 2'")
    print(f"{'='*70}")
    collator_k2 = ExactKCollator(model.processor, k=2)
    loader_k2 = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collator_k2,
    )
    res_k2 = evaluate(model, loader_k2, device, "exactly_2")
    res_k2["checkpoint"] = str(args.checkpoint)
    with open(output_dir / "condition_B_exactly_2.json", "w") as f:
        json.dump(res_k2, f, indent=2)

    # ---- Summary -------------------------------------------------------- #
    summary = {
        "checkpoint": str(args.checkpoint),
        "n_samples": len(res_1to6["per_sample"]),
        "condition_A_1_to_6": {
            "mean_iou": res_1to6["mean_iou"],
            "mean_ari": res_1to6["mean_ari"],
            "k_dist": res_1to6["k_dist"],
        },
        "condition_B_exactly_2": {
            "mean_iou": res_k2["mean_iou"],
            "mean_ari": res_k2["mean_ari"],
            "k_dist": res_k2["k_dist"],
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ABLATION: Prompt Count vs mIoU/mARI on RWTD")
    print(f"{'='*70}")
    print(f"  {'Prompt':<20} {'mIoU':>8}  {'mARI':>8}  {'k_pred dist':<40}")
    print(f"  {'-'*70}")
    kd1 = " ".join(f"{k}:{v}" for k, v in sorted(res_1to6["k_dist"].items()))
    kd2 = " ".join(f"{k}:{v}" for k, v in sorted(res_k2["k_dist"].items()))
    print(f"  {'1 to 6':<20} {res_1to6['mean_iou']:>8.4f}  "
          f"{res_1to6['mean_ari']:>8.4f}  {kd1}")
    print(f"  {'exactly 2':<20} {res_k2['mean_iou']:>8.4f}  "
          f"{res_k2['mean_ari']:>8.4f}  {kd2}")
    delta = res_k2["mean_iou"] - res_1to6["mean_iou"]
    print(f"\n  Delta (mIoU): {delta:+.4f}")
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
