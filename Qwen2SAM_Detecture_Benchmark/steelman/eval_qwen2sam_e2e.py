#!/usr/bin/env python3
"""Benchmark evaluator — Qwen2SAM-DeTexture (E2E live generation).

The count prompt is passed in via `--prompt-n`, substituted into
`USER_PROMPT_TEMPLATE`. Typical usage from master_runner.py:

  RWTD / STLD  (K=2 oracle regime)    →  --prompt-n 2
  ADE20k_Det.  (autonomous regime)    →  --prompt-n "between 1 and 6"
  CAID         (single-region)        →  --prompt-n 1

Pipeline mirrors Qwen2SAM_Detecture/scripts/eval_Detecture.py: generation
+ adaptive [SEG] extraction + SAM3 semantic head, returning `mask_logits`
of shape (B, 7, H, W), `k_preds`, `pad_mask`. Scoring uses the shared
metrics_utils.hungarian_match + adjusted_rand_index so numbers are
directly comparable to the Grounded-SAM3 and SAM3-vanilla scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- path wiring -------------------------------------------------------- #
_STEELMAN_ROOT = Path(__file__).resolve().parent
_SUITE_ROOT = _STEELMAN_ROOT.parent
_QWEN_REPO = _SUITE_ROOT.parent / "Qwen2SAM_Detecture"

for p in (_SUITE_ROOT, _QWEN_REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metrics_utils import hungarian_match, adjusted_rand_index         # noqa: E402
from data_utils import load_samples, load_gt_masks, load_image_rgb      # noqa: E402

# Qwen repo imports
from models.qwen2sam_detecture import (                                 # noqa: E402
    Qwen2SAMDetecture, MAX_TEXTURES, NUM_QUERY_SLOTS,
)
from data.dataset import (                                              # noqa: E402
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
)
from training.utils import load_config, load_checkpoint                 # noqa: E402


# --------------------------------------------------------------------- #
# Batch construction (one sample at a time)                              #
# --------------------------------------------------------------------- #

SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _sam3_preprocess(image_rgb: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - SAM3_MEAN) / SAM3_STD
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))


def build_qwen_batch(processor, image_rgb: np.ndarray, image_size: int,
                     user_prompt: str, device: torch.device):
    """Assemble the single-sample batch expected by `inference_forward`.

    Mirrors DetectureCollator but uses an arbitrary user prompt string.
    Image is resized to image_size for both Qwen and SAM3 branches.
    """
    from PIL import Image

    img_resized = cv2.resize(
        image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR,
    )
    pil = Image.fromarray(img_resized)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    qwen_inputs = processor(
        text=[text], images=[pil], return_tensors="pt", padding=True,
    )
    qwen_inputs.pop("token_type_ids", None)
    qwen_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in qwen_inputs.items()
    }

    sam_image = _sam3_preprocess(image_rgb, image_size).unsqueeze(0).to(device)
    return qwen_inputs, sam_image


# --------------------------------------------------------------------- #
# Native Qwen partition → masks (slot 0 dustbin, slots 1..k_pred = tex)  #
# --------------------------------------------------------------------- #

def qwen_partition(mask_logits: torch.Tensor, pad_mask: torch.Tensor,
                   k_pred: int, gt_h: int, gt_w: int) -> tuple:
    """Argmax over the 7-slot Qwen mask stack, masked by `pad_mask`.

    Returns:
        argmax_map: (gt_h, gt_w) int32, values in {0..k_pred}. 0 = learned
                    dustbin (Qwen's own background channel), 1..k_pred = each
                    predicted texture.
        pred_masks: list of k_pred binary (gt_h, gt_w) masks, texture i at
                    index i-1.
    """
    masked = mask_logits.clone()
    inf_plane = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
    masked[inf_plane] = float("-inf")

    logits = masked[0]                                    # (7, H, W)
    if logits.shape[-2:] != (gt_h, gt_w):
        logits = F.interpolate(
            logits.unsqueeze(0).float(),
            size=(gt_h, gt_w), mode="bilinear", align_corners=False,
        ).squeeze(0)

    argmax_map = logits.argmax(dim=0).cpu().numpy().astype(np.int32)
    pred_masks = [
        (argmax_map == i).astype(np.float32) for i in range(1, k_pred + 1)
    ]
    return argmax_map, pred_masks


# --------------------------------------------------------------------- #
# Per-sample eval                                                         #
# --------------------------------------------------------------------- #

def evaluate_sample(model, processor, sample: dict, image_size: int,
                    user_prompt: str, device: torch.device) -> dict:
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

    qwen_inputs, sam_image = build_qwen_batch(
        processor, image_rgb, image_size, user_prompt, device,
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
        out = model.inference_forward(
            qwen_inputs=qwen_inputs, sam_images=sam_image,
        )

    mask_logits = out["mask_logits"]                   # (1, 7, H, W)
    pad_mask = out["pad_mask"]                         # (1, 7) bool
    k_pred = int(out["k_preds"][0].item())
    gen_text = out.get("generated_text", [""])[0]

    argmax_map, pred_masks = qwen_partition(
        mask_logits, pad_mask, k_pred, gt_h, gt_w,
    )

    match = hungarian_match(pred_masks, gt_masks)
    ari = adjusted_rand_index(argmax_map, gt_masks)
    bg_coverage = float((argmax_map == 0).mean())

    return {
        "id": sid, "status": "ok",
        "k_pred": k_pred, "n_pred": match["n_preds"], "n_gt": match["n_gts"],
        "panoptic_iou": match["panoptic_iou"],
        "panoptic_dice": match["panoptic_dice"],
        "matched_mean_iou": match["matched_mean_iou"],
        "matched_mean_dice": match["matched_mean_dice"],
        "assignment": match["assignment"],
        "ari": ari,
        "bg_coverage": bg_coverage,
        "generated_text": gen_text,
        "gt_descs": gt_descs,
    }


# --------------------------------------------------------------------- #
# Aggregation                                                             #
# --------------------------------------------------------------------- #

def aggregate(results: list) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"n_total": len(results), "n_ok": 0}
    from collections import Counter
    k_dist = Counter(r["k_pred"] for r in ok)
    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "panoptic_iou": float(np.mean([r["panoptic_iou"] for r in ok])),
        "panoptic_dice": float(np.mean([r["panoptic_dice"] for r in ok])),
        "matched_mean_iou": float(np.mean([r["matched_mean_iou"] for r in ok])),
        "matched_mean_dice": float(np.mean([r["matched_mean_dice"] for r in ok])),
        "mean_ari": float(np.mean([r["ari"] for r in ok])),
        "mean_bg_coverage": float(np.mean([r["bg_coverage"] for r in ok])),
        "mean_k_pred": float(np.mean([r["k_pred"] for r in ok])),
        "k_pred_dist": {int(k): int(v) for k, v in sorted(k_dist.items())},
    }


# --------------------------------------------------------------------- #
# Main                                                                    #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (RWTD, STLD, ADE20k_Detecture, CAID, ...)")
    parser.add_argument("--prompt-n", default="between 1 and 6",
                        help="Value substituted into USER_PROMPT_TEMPLATE's "
                             "{N} slot. Common values: '2' (RWTD/STLD oracle "
                             "K=2), 'between 1 and 6' (ADE20k autonomous), "
                             "'1' (CAID).")
    parser.add_argument("--metadata", default=None,
                        help="Override metadata.json (defaults to "
                             "~/datasets/<dataset>/metadata.json)")
    parser.add_argument("--config", default=str(_QWEN_REPO / "configs" / "detecture.yaml"))
    parser.add_argument("--checkpoint", default=str(_QWEN_REPO / "checkpoints" / "best.pt"))
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metadata = args.metadata or f"~/datasets/{args.dataset}/metadata.json"
    user_prompt = USER_PROMPT_TEMPLATE.format(N=args.prompt_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "zero_shot_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(metadata, limit=args.limit)
    print(f"[data] {args.dataset} n={len(samples)}  prompt_N={args.prompt_n!r}",
          flush=True)

    print(f"[model] loading Qwen2SAMDetecture from {args.checkpoint}", flush=True)
    cfg = load_config(args.config)
    model = Qwen2SAMDetecture(cfg, device=str(device))
    load_checkpoint(model, None, args.checkpoint, device=str(device))
    model.eval()
    processor = model.processor

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        sid = sample["id"]
        try:
            res = evaluate_sample(model, processor, sample,
                                  args.image_size, user_prompt, device)
        except Exception as e:                                   # noqa: BLE001
            res = {"id": sid, "status": "exception", "error": repr(e)}
        results.append(res)

        if res.get("status") == "ok":
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"k_pred={res['k_pred']}  K={res['n_gt']}  "
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
        "model_family": "qwen2sam_e2e",
        "dataset": args.dataset,
        "metadata_path": metadata,
        "checkpoint": str(args.checkpoint),
        "prompt_N": args.prompt_n,
        "image_size": args.image_size,
        "elapsed_seconds": elapsed,
    })

    json_path.write_text(json.dumps(
        {"summary": summary, "samples": results},
        indent=2, default=str,
    ))

    print("\n" + "=" * 72)
    print(f"  qwen2sam_e2e [{args.dataset}, N={args.prompt_n!r}] — "
          f"{summary.get('n_ok', 0)}/{summary.get('n_total', 0)} ok "
          f"in {elapsed:.1f}s")
    if summary.get("n_ok", 0):
        print(f"  pIoU = {summary['panoptic_iou']:.4f}   "
              f"mIoU = {summary['matched_mean_iou']:.4f}   "
              f"ARI  = {summary['mean_ari']:.4f}   "
              f"<k> = {summary['mean_k_pred']:.2f}   "
              f"k_dist = {summary['k_pred_dist']}")
    print(f"  wrote {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
