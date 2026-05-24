#!/usr/bin/env python3
"""Steelman benchmark — Model 2: Grounded-SAM3 (Oracle K=2 + inverse trick).

Oracle advantage: we KNOW K=2 on RWTD/STLD. Procedure:
  1. Grounding DINO consumes a dot-separated text query for textures and
     returns boxes + scores. We keep the SINGLE TOP-1 box by confidence.
  2. That box is fed through SAM3's geometry encoder → fusion encoder →
     Semantic Seg Head (skipping forward_grounding) to produce one logit
     map `m1` of shape (gt_h, gt_w).
  3. The second class is the mathematical inverse `m2 = -m1`.
  4. Stack [m1, m2] and score via metrics_utils.compute_sample_metrics
     (softmax + static dustbin at logit=0.0, Hungarian, ARI, Coverage).

The [m1, -m1, 0] stack collapses to a binary partition of the image
(dustbin wins only at m1 == 0 exactly — measure zero), so the baseline
gets a geometry- and detector-grounded two-region split it could not
produce from a single prompt without this oracle hint.

Pipeline helpers (DINO load, SAM3 load, box → semantic logit) are
imported from the parent suite's eval_grounded_sam3.py so the two
scripts stay in lockstep on feature extraction.
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

# Reuse the parent suite's DINO + SAM3 box-prompt helpers verbatim so
# feature extraction is identical to the non-steelman baseline.
from eval_grounded_sam3 import (                                        # noqa: E402
    load_grounding_dino, load_sam3, detect_boxes, xyxy_to_cxcywh_norm,
    sam3_box_to_semantic_logits,
)


# --------------------------------------------------------------------- #
# Per-sample eval                                                         #
# --------------------------------------------------------------------- #

def evaluate_sample(
    sam3, dino_model, dino_proc,
    sample: dict, dino_text: str,
    dustbin_logit: float, missing_logit: float, image_size: int,
    box_threshold: float, text_threshold: float,
    device: torch.device,
) -> dict:
    from PIL import Image

    sid = sample["id"]
    try:
        image_rgb = load_image_rgb(sample["image_path"])
    except FileNotFoundError:
        return {"id": sid, "status": "image_read_failed"}
    image_pil = Image.fromarray(image_rgb)

    gt_masks, kept_idx = load_gt_masks(sample["gt_masks"])
    if len(gt_masks) == 0:
        return {"id": sid, "status": "no_gt_masks"}
    gt_descs = [sample["gt_descs"][i] for i in kept_idx]
    gt_h, gt_w = gt_masks[0].shape

    # --- DINO top-1 box ----------------------------------------------- #
    boxes_xyxy, scores = detect_boxes(
        dino_model, dino_proc, image_pil, dino_text, device,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )
    n_det = int(boxes_xyxy.shape[0])
    if n_det == 0:
        # No detection — both classes fall back to the missing-logit plane
        # so Hungarian charges 0 IoU for both GT masks.
        logits = np.full((2, gt_h, gt_w), missing_logit, dtype=np.float32)
        top1_score = None
        top1_box_xyxy = None
    else:
        top1_box = boxes_xyxy[:1]                            # (1, 4)
        top1_score = float(scores[0].item())
        top1_box_xyxy = [float(v) for v in top1_box[0].tolist()]

        # --- SAM3 semantic logit for the top-1 box ------------------- #
        H, W = image_pil.size[1], image_pil.size[0]
        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            backbone_out = sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img
            cx, cy, w, h = xyxy_to_cxcywh_norm(top1_box[0], H, W)
            m1 = sam3_box_to_semantic_logits(
                sam3, backbone_out, (cx, cy, w, h), gt_h, gt_w, device,
            )
        # Inverse trick: second class = negated logit.
        logits = np.stack([m1, -m1], axis=0).astype(np.float32)

    metrics = compute_sample_metrics(logits, gt_masks, dustbin_logit=dustbin_logit)

    return {
        "id": sid, "status": "ok",
        "n_detected_boxes": n_det,
        "top1_score": top1_score,
        "top1_box_xyxy": top1_box_xyxy,
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
        "n_no_detection": int(sum(1 for r in ok if r["n_detected_boxes"] == 0)),
    }


# --------------------------------------------------------------------- #
# Main                                                                    #
# --------------------------------------------------------------------- #

DINO_TEXT_BY_DATASET = {
    "RWTD": "foreground texture . background texture .",
    "STLD": "foreground texture . background texture .",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["RWTD", "STLD"])
    parser.add_argument("--metadata", default=None,
                        help="Override metadata.json (defaults to "
                             "~/datasets/<dataset>/metadata.json)")
    parser.add_argument("--dino-text", default=None,
                        help="Override the DINO text query.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--dustbin-logit", type=float, default=0.0)
    parser.add_argument("--missing-logit", type=float, default=-1.0e4)
    args = parser.parse_args()

    metadata = args.metadata or f"~/datasets/{args.dataset}/metadata.json"
    dino_text = args.dino_text or DINO_TEXT_BY_DATASET[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "zero_shot_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(metadata, limit=args.limit)
    print(f"[data] {args.dataset} n={len(samples)}  dino_text={dino_text!r}  "
          f"(top-1 box + inverse trick, K=2)", flush=True)

    dino_model, dino_proc = load_grounding_dino(device, args.dino_model)
    sam3 = load_sam3(device)

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        sid = sample["id"]
        try:
            res = evaluate_sample(
                sam3, dino_model, dino_proc, sample, dino_text,
                args.dustbin_logit, args.missing_logit, args.image_size,
                args.box_threshold, args.text_threshold, device,
            )
        except Exception as e:                                   # noqa: BLE001
            res = {"id": sid, "status": "exception", "error": repr(e)}
        results.append(res)

        if res.get("status") == "ok":
            score = res.get("top1_score")
            score_s = f"{score:.3f}" if score is not None else "—"
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"det={res['n_detected_boxes']:<2d}  "
                  f"top1={score_s}  K={res['n_gt']}  "
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
        "model_family": "grounded_sam3_top1_inverse",
        "dataset": args.dataset,
        "metadata_path": metadata,
        "dino_text": dino_text,
        "dino_model": args.dino_model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "dustbin_logit": args.dustbin_logit,
        "missing_logit": args.missing_logit,
        "image_size": args.image_size,
        "elapsed_seconds": elapsed,
    })

    json_path.write_text(json.dumps(
        {"summary": summary, "samples": results},
        indent=2, default=str,
    ))

    print("\n" + "=" * 72)
    print(f"  grounded_sam3_top1_inverse [{args.dataset}] — "
          f"{summary.get('n_ok', 0)}/{summary.get('n_total', 0)} ok "
          f"in {elapsed:.1f}s")
    if summary.get("n_ok", 0):
        print(f"  pIoU = {summary['panoptic_iou']:.4f}   "
              f"mIoU = {summary['matched_mean_iou']:.4f}   "
              f"ARI  = {summary['mean_ari']:.4f}   "
              f"no-det={summary['n_no_detection']}")
    print(f"  wrote {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
