#!/usr/bin/env python3
"""Steelman benchmark — Model 3: SAM3 vanilla (Oracle K=2 + inverse trick).

Pipeline per sample (K assumed = 2):
  1. Text prompt = the literal word "texture".
  2. SAM3's VE text encoder → fusion encoder → Semantic Seg Head produces a
     single logit map `m1` of shape (gt_h, gt_w).
  3. Second class = the mathematical inverse `m2 = -m1`.
  4. Stack [m1, m2] and score via metrics_utils.compute_sample_metrics
     (softmax + static dustbin logit=0.0 + Hungarian + ARI).

With dustbin=0, argmax([m1, -m1, 0]) assigns every pixel to class 0 where
m1 > 0 and class 1 where m1 < 0, modulo a measure-zero tie at m1 == 0 —
i.e. a clean binary partition driven purely by SAM3's zero-shot
understanding of the word "texture".

Helpers (SAM3 load, text → semantic logit) are imported from the parent
suite's eval_sam3_vanilla.py so feature extraction matches the existing
baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

# --- path wiring -------------------------------------------------------- #
_STEELMAN_ROOT = Path(__file__).resolve().parent
_SUITE_ROOT = _STEELMAN_ROOT.parent
if str(_SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUITE_ROOT))

_SAM3_ROOT = Path(os.environ.get("SAM3_ROOT", str(Path.home() / "sam3")))
if str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))

from metrics_utils import compute_sample_metrics                        # noqa: E402
from data_utils import (                                                # noqa: E402
    load_samples, load_gt_masks, load_image_rgb,
    preprocess_image_for_sam3,
)

from eval_sam3_vanilla import (                                         # noqa: E402
    load_sam3, sam3_text_to_semantic_logits,
)


# --------------------------------------------------------------------- #
# Per-sample eval                                                         #
# --------------------------------------------------------------------- #

def evaluate_sample(
    sam3, sample: dict, text: str,
    dustbin_logit: float, image_size: int, device: torch.device,
) -> dict:
    sid = sample["id"]
    try:
        image_rgb = load_image_rgb(sample["image_path"])
    except FileNotFoundError:
        return {"id": sid, "status": "image_read_failed"}

    gt_masks, kept_idx = load_gt_masks(sample["gt_masks"])
    if len(gt_masks) == 0:
        return {"id": sid, "status": "no_gt_masks"}
    gt_descs = [sample["gt_descs"][i] for i in kept_idx]
    gt_h, gt_w = gt_masks[0].shape

    sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
        backbone_out = sam3.backbone.forward_image(sam_img)
        backbone_out["img_batch_all_stages"] = sam_img
        m1 = sam3_text_to_semantic_logits(
            sam3, backbone_out, text, gt_h, gt_w, device,
        )

    logits = np.stack([m1, -m1], axis=0).astype(np.float32)
    metrics = compute_sample_metrics(logits, gt_masks, dustbin_logit=dustbin_logit)

    return {
        "id": sid, "status": "ok",
        "text": text,
        "n_pred": metrics["n_pred"], "n_gt": metrics["n_gt"],
        "panoptic_iou": metrics["panoptic_iou"],
        "panoptic_dice": metrics["panoptic_dice"],
        "matched_mean_iou": metrics["matched_mean_iou"],
        "matched_mean_dice": metrics["matched_mean_dice"],
        "assignment": metrics["assignment"],
        "ari": metrics["ari"],
        "bg_coverage": metrics["bg_coverage"],
        "gt_descs": gt_descs,
    }


# --------------------------------------------------------------------- #
# Aggregation                                                             #
# --------------------------------------------------------------------- #

def aggregate(results: list) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"n_total": len(results), "n_ok": 0}
    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "panoptic_iou": float(np.mean([r["panoptic_iou"] for r in ok])),
        "panoptic_dice": float(np.mean([r["panoptic_dice"] for r in ok])),
        "matched_mean_iou": float(np.mean([r["matched_mean_iou"] for r in ok])),
        "matched_mean_dice": float(np.mean([r["matched_mean_dice"] for r in ok])),
        "mean_ari": float(np.mean([r["ari"] for r in ok])),
        "mean_bg_coverage": float(np.mean([r["bg_coverage"] for r in ok])),
    }


# --------------------------------------------------------------------- #
# Main                                                                    #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["RWTD", "STLD"])
    parser.add_argument("--metadata", default=None,
                        help="Override metadata.json (defaults to "
                             "~/datasets/<dataset>/metadata.json)")
    parser.add_argument("--text", default="texture",
                        help="SAM3 text prompt (default: 'texture').")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--dustbin-logit", type=float, default=0.0)
    args = parser.parse_args()

    metadata = args.metadata or f"~/datasets/{args.dataset}/metadata.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "zero_shot_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(metadata, limit=args.limit)
    print(f"[data] {args.dataset} n={len(samples)}  text={args.text!r}  "
          f"(inverse trick, K=2)", flush=True)

    sam3 = load_sam3(device)

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        sid = sample["id"]
        try:
            res = evaluate_sample(
                sam3, sample, args.text,
                args.dustbin_logit, args.image_size, device,
            )
        except Exception as e:                                   # noqa: BLE001
            res = {"id": sid, "status": "exception", "error": repr(e)}
        results.append(res)

        if res.get("status") == "ok":
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"K={res['n_gt']}  "
                  f"pIoU={res['panoptic_iou']:.4f}  "
                  f"mIoU={res['matched_mean_iou']:.4f}  "
                  f"ARI={res['ari']:.4f}  "
                  f"bg={res['bg_coverage']:.2f}")
        else:
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"SKIP status={res.get('status')}")

    elapsed = time.time() - t0
    summary = aggregate(results)
    summary.update({
        "model_family": "sam3_vanilla_inverse",
        "dataset": args.dataset,
        "metadata_path": metadata,
        "text": args.text,
        "dustbin_logit": args.dustbin_logit,
        "image_size": args.image_size,
        "elapsed_seconds": elapsed,
    })

    json_path.write_text(json.dumps(
        {"summary": summary, "samples": results},
        indent=2, default=str,
    ))

    print("\n" + "=" * 72)
    print(f"  sam3_vanilla_inverse [{args.dataset}] — "
          f"{summary.get('n_ok', 0)}/{summary.get('n_total', 0)} ok "
          f"in {elapsed:.1f}s")
    if summary.get("n_ok", 0):
        print(f"  pIoU = {summary['panoptic_iou']:.4f}   "
              f"mIoU = {summary['matched_mean_iou']:.4f}   "
              f"ARI  = {summary['mean_ari']:.4f}")
    print(f"  wrote {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
