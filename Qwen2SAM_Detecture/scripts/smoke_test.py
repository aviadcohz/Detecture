#!/usr/bin/env python3
"""
Run inference on any test dataset using a saved checkpoint.

Usage:
    cd Qwen2SAM_Detecture
    python scripts/test_on_dataset.py \
        --config configs/detecture.yaml \
        --checkpoint checkpoints/best.pt \
        --test-metadata ~/datasets/ADE20k_Detecture/metadata.json \
        --output-dir checkpoints/test_ade20k_detecture
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detecture import Qwen2SAMDetecture, MAX_TEXTURES
from data.dataset import DetectureDataset, DetectureCollator
from training.utils import load_config, load_checkpoint
from training.monitor import _compute_ari, _colorize_mask

from scipy.optimize import linear_sum_assignment


def compute_matched_miou(pred, gt, k_pred, k_gt):
    """Compute mIoU with Hungarian matching between pred and GT classes."""
    cost = np.zeros((max(k_pred, 1), max(k_gt, 1)))
    for pi in range(k_pred):
        pred_c = (pred == pi + 1)
        for gi in range(k_gt):
            gt_c = (gt == gi + 1)
            inter = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()
            cost[pi, gi] = 1.0 - (inter / max(union, 1))

    row_ind, col_ind = linear_sum_assignment(cost)
    ious = []
    for r, c in zip(row_ind, col_ind):
        if r < k_pred and c < k_gt:
            ious.append(1.0 - cost[r, c])
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser(description="Test model on a dataset")
    parser.add_argument("--config", type=str, default="configs/detecture.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--test-metadata", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-vis", action="store_true", default=True,
                        help="Save per-sample visualizations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = Path(args.test_metadata).parent.name
    output_dir = Path(args.output_dir or f"checkpoints/test_{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + checkpoint ----
    print(f"Loading model...")
    model = Qwen2SAMDetecture(cfg, device=str(device))
    ckpt_path = args.checkpoint
    state = torch.load(ckpt_path, map_location=device)
    epoch = load_checkpoint(model, None, ckpt_path, device=str(device))
    print(f"Loaded checkpoint: {ckpt_path} (epoch {epoch + 1})")
    model.eval()

    # ---- Load test dataset ----
    test_ds = DetectureDataset(
        args.test_metadata,
        image_size=cfg["data"].get("image_size", 1008),
        augment=False,
    )
    collator = DetectureCollator(model.processor, inference=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator,
    )
    print(f"Test dataset: {args.test_metadata} ({len(test_ds)} samples)")

    # ---- Run inference ----
    all_ious = []
    all_aris = []
    per_sample = []
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            qwen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch["qwen_inputs"].items()}
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gts = batch["k_gts"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference_forward(
                    qwen_inputs=qwen_inputs, sam_images=sam_images,
                )

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())

            # WTA: upsample logits to GT resolution, then argmax
            masked_logits = mask_logits.clone()
            inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked_logits)
            masked_logits[inf_mask] = float("-inf")

            H_gt, W_gt = index_masks.shape[1], index_masks.shape[2]
            if masked_logits.shape[2] != H_gt or masked_logits.shape[3] != W_gt:
                masked_logits = F.interpolate(
                    masked_logits.float(), size=(H_gt, W_gt),
                    mode="bilinear", align_corners=False,
                )
            pred = masked_logits[0].argmax(dim=0).cpu().numpy()
            gt = index_masks[0].numpy()
            k_gt = int(k_gts[0].item())

            # mIoU with Hungarian matching
            sample_iou = compute_matched_miou(pred, gt, k_pred, k_gt)
            all_ious.append(sample_iou)

            # ARI
            ari = _compute_ari(pred, gt)
            all_aris.append(ari)

            generated_text = out.get("generated_text", [""])[0]

            per_sample.append({
                "idx": idx,
                "miou": float(sample_iou),
                "ari": float(ari),
                "k_gt": k_gt,
                "k_pred": k_pred,
                "generated_text": generated_text,
            })

            # Save visualization
            if args.save_vis:
                sam_img = batch["sam_images"][0].permute(1, 2, 0).cpu().numpy()
                img = ((sam_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                h, w = gt.shape
                img_r = cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img
                gt_vis = _colorize_mask(gt)
                pred_vis = _colorize_mask(pred)

                overlay = img_r.copy()
                active = pred > 0
                overlay[active] = cv2.addWeighted(img_r, 0.4, pred_vis, 0.6, 0)[active]

                canvas = np.hstack([img_r, gt_vis, pred_vis, overlay])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, "Original", (5, 15), font, 0.4, (255, 255, 255), 1)
                cv2.putText(canvas, "GT", (w + 5, 15), font, 0.4, (255, 255, 255), 1)
                cv2.putText(canvas, f"Pred k={k_pred}", (2 * w + 5, 15), font, 0.4, (255, 255, 255), 1)
                cv2.putText(canvas, f"IoU={sample_iou:.3f} ARI={ari:.3f}", (3 * w + 5, 15),
                            font, 0.35, (255, 255, 255), 1)
                cv2.imwrite(str(output_dir / f"sample_{idx:04d}.jpg"), canvas)

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{idx+1}/{len(test_ds)}] {elapsed:.0f}s")

    # ---- Results ----
    test_miou = np.mean(all_ious)
    test_mari = np.mean(all_aris)
    sorted_samples = sorted(per_sample, key=lambda x: x["miou"])

    results = {
        "dataset": dataset_name,
        "checkpoint": str(ckpt_path),
        "epoch": epoch + 1,
        "n_samples": len(per_sample),
        "test_miou": float(test_miou),
        "test_mari": float(test_mari),
        "per_sample": per_sample,
        "worst_10": sorted_samples[:10],
        "best_10": sorted_samples[-10:][::-1],
        "median": sorted_samples[len(sorted_samples) // 2],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Print Summary ----
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Checkpoint: {ckpt_path} (epoch {epoch + 1})")
    print(f"  Samples:    {len(per_sample)}")
    print(f"  mIoU:       {test_miou:.4f}")
    print(f"  mARI:       {test_mari:.4f}")
    print(f"")
    print(f"  Worst 5:")
    for s in sorted_samples[:5]:
        print(f"    #{s['idx']:4d}: mIoU={s['miou']:.3f} k_gt={s['k_gt']} k_pred={s['k_pred']}")
    print(f"  Best 5:")
    for s in sorted_samples[-5:][::-1]:
        print(f"    #{s['idx']:4d}: mIoU={s['miou']:.3f} k_gt={s['k_gt']} k_pred={s['k_pred']}")
    print(f"  Median: #{results['median']['idx']}: mIoU={results['median']['miou']:.3f}")
    print(f"")
    print(f"  ZS baseline (RWTD): mIoU=0.7063  mARI=0.6879")
    print(f"  Output: {output_dir}")
    print(f"  Time: {(time.time() - t0) / 60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
