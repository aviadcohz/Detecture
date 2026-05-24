#!/usr/bin/env python3
"""
Sanity check: reproduce the 0.9289 mIoU baseline on RWTD using ONLY frozen SAM 3
with native text encoding — no Qwen, no Bridge, no trained weights.

Source of truth:
  <retired>/qwen2sam/scripts/run_semseg_only.py
  <retired>/qwen2sam/scripts/pair_optimizer.py
  <retired>/qwen2sam/models/qwen2sam_v3.py

Exact replication:
  1. For each RWTD sample, use the GT descriptions from RWTD_GT.json
     (parsed.desc_a / parsed.desc_b) — exactly ONE description per texture.
  2. Preprocess image with SAM 3's preprocessing (1008×1008, SAM3 mean/std).
  3. Encode each description via sam3.backbone.forward_text([desc]) → get
     the FULL sequence text features (language_features + language_mask).
  4. Pass that sequence prompt through SAM 3's full DETR pipeline
     (encoder fusion → DETR decoder → segmentation_head) and take
     the semantic_mask output — the text-independent head fed with
     pixel_embed enriched via cross-attention with the text sequence.
  5. Resize heatmaps to GT resolution (bilinear). WTA: a = (hm_a > hm_b).
     Degenerate fallback: if min pixel-coverage < 2%, threshold at 0.5.
  6. Label swap: pick whichever A/B assignment maximizes sum-IoU.
  7. Compute per-texture IoU (binarized at 0.5), mean → per-sample mIoU.

Because we use the SINGLE GT description per texture (not the 5-way diverse
generation with pair-optimizer), we expect a score *slightly below* the
reference 0.9289 — but close (≈ 0.88-0.92). Anything below ~0.85 means
something is off with our SAM 3 install, preprocessing, or forward path.

Usage:
    python scripts/verify_sam_baseline.py
    python scripts/verify_sam_baseline.py --limit 20         # smoke test
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, os.environ.get("SAM3_ROOT", str(Path.home() / "sam3")))

from data.dataset import preprocess_image_for_sam3, SAM3_SIZE

_DATASETS = Path(os.path.expanduser(
    os.environ.get("DETECTURE_DATASETS_ROOT", "~/datasets")
))
RWTD_IMAGES_DIR = _DATASETS / "RWTD" / "images"
RWTD_MASKS_DIR = _DATASETS / "RWTD" / "textures_mask"


# --------------------------------------------------------------------- #
#  SAM 3 loader + native text path                                        #
# --------------------------------------------------------------------- #

def build_sam3(device):
    import sam3
    from sam3.model_builder import build_sam3_image_model
    bpe = Path(sam3.__path__[0]) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        str(bpe), checkpoint_path=None, device=str(device),
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def run_sam3_semantic_with_text(sam3, backbone_out, text: str, device):
    """Replicate `_run_sam3_from_backbone()` from qwen2sam_v3.py, using the
    TEXT SEQUENCE prompt produced by SAM 3's own forward_text.

    Returns the semantic_mask logit tensor (1, 1, H, W).
    """
    from sam3.model.model_misc import inverse_sigmoid
    from sam3.model.box_ops import box_cxcywh_to_xyxy

    # 1. Encode text via SAM's native text path.
    t_out = sam3.backbone.forward_text([text], device=device)
    prompt = t_out["language_features"].squeeze(1)  # (N_tokens, 256)
    prompt_mask = t_out["language_mask"].squeeze(0)  # (N_tokens,) True=PAD
    B = 1
    prompt = prompt.unsqueeze(0)            # (B, N, 256)
    prompt_mask = prompt_mask.unsqueeze(0)  # (B, N)

    # 2. Get image features indexed to the (B,) img ids.
    img_ids = torch.arange(B, device=device)
    n_levels = sam3.num_feature_levels
    vis_feats = backbone_out["backbone_fpn"][-n_levels:]
    vis_pos_enc = backbone_out["vision_pos_enc"][-n_levels:]
    vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
    img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
    img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1)
                      for x in vis_pos_enc]

    # 3. Seq-first prompt for the transformer.
    prompt_sf = prompt.transpose(0, 1)          # (N, B, 256)
    prompt_pos = torch.zeros_like(prompt_sf)

    # 4. Fusion encoder.
    mem = sam3.transformer.encoder(
        src=[f.clone() for f in img_feats],
        src_key_padding_mask=None,
        src_pos=[p.clone() for p in img_pos_embeds],
        prompt=prompt_sf,
        prompt_pos=prompt_pos,
        prompt_key_padding_mask=prompt_mask,
        feat_sizes=vis_feat_sizes,
    )

    encoder_hs = mem["memory"]
    pos_embed = mem["pos_embed"]
    padding_mask = mem["padding_mask"]

    # 5. DETR decoder with learned object queries.
    query_embed = sam3.transformer.decoder.query_embed.weight
    tgt = query_embed.unsqueeze(1).expand(-1, B, -1).clone()
    hs, ref_boxes, _, _ = sam3.transformer.decoder(
        tgt=tgt,
        memory=encoder_hs,
        memory_key_padding_mask=padding_mask,
        pos=pos_embed,
        reference_boxes=None,
        level_start_index=mem["level_start_index"],
        spatial_shapes=mem["spatial_shapes"],
        valid_ratios=mem["valid_ratios"],
        tgt_mask=None,
        memory_text=prompt_sf,
        text_attention_mask=prompt_mask,
        apply_dac=False,
    )

    # 6. Segmentation head. Enrich pixel features via cross-attn with text,
    #    then run the text-independent semantic_seg_head on pixel_embed.
    seg_head = sam3.segmentation_head
    enc_hs = encoder_hs
    if seg_head.cross_attend_prompt is not None:
        tgt2 = seg_head.cross_attn_norm(enc_hs)
        tgt2 = seg_head.cross_attend_prompt(
            query=tgt2, key=prompt_sf, value=prompt_sf,
            key_padding_mask=prompt_mask,
        )[0]
        enc_hs = tgt2 + enc_hs

    pixel_embed = seg_head._embed_pixels(
        backbone_feats=backbone_out["backbone_fpn"],
        image_ids=img_ids,
        encoder_hidden_states=enc_hs,
    )

    semantic_mask = seg_head.semantic_seg_head(pixel_embed)   # (B, 1, H, W)
    return semantic_mask


# --------------------------------------------------------------------- #
#  Metric helpers — reproducing pair_optimizer.compute_iou / postprocess  #
# --------------------------------------------------------------------- #

def postprocess_to_prob(mask_logit: torch.Tensor, gt_h: int, gt_w: int) -> np.ndarray:
    """Bilinear-upsample to GT size, apply sigmoid, return numpy float map."""
    if mask_logit.ndim == 2:
        mask_logit = mask_logit[None, None]
    elif mask_logit.ndim == 3:
        mask_logit = mask_logit.unsqueeze(0)
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit.float(), size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        )
    return mask_logit.sigmoid().squeeze().float().cpu().numpy()


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return 0.0 if union == 0 else float(inter / union)


# --------------------------------------------------------------------- #
#  Per-sample pipeline                                                    #
# --------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_one(sam3, crop: str, desc_a: str, desc_b: str,
                 image_size: int, device):
    img_path = RWTD_IMAGES_DIR / f"{crop}.jpg"
    mask_a_path = RWTD_MASKS_DIR / f"{crop}_mask_a.png"
    mask_b_path = RWTD_MASKS_DIR / f"{crop}_mask_b.png"
    if not (img_path.exists() and mask_a_path.exists() and mask_b_path.exists()):
        return None

    # Image preprocessing — identical to the baseline script.
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sam_img = preprocess_image_for_sam3(rgb, image_size).unsqueeze(0).to(device)

    gt_a = cv2.imread(str(mask_a_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    gt_b = cv2.imread(str(mask_b_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    gt_h, gt_w = gt_a.shape

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        backbone_out = sam3.backbone.forward_image(sam_img)
        backbone_out["img_batch_all_stages"] = sam_img

        sem_a = run_sam3_semantic_with_text(sam3, backbone_out, desc_a, device)
        sem_b = run_sam3_semantic_with_text(sam3, backbone_out, desc_b, device)

    # Upsample + sigmoid → heatmaps at GT resolution.
    hm_a = postprocess_to_prob(sem_a[0, 0], gt_h, gt_w)
    hm_b = postprocess_to_prob(sem_b[0, 0], gt_h, gt_w)

    # WTA (exact copy of run_semseg_only.py logic).
    wta_a = (hm_a > hm_b).astype(np.float32)
    wta_b = (hm_b > hm_a).astype(np.float32)
    pix_a = wta_a.sum() / wta_a.size
    pix_b = wta_b.sum() / wta_b.size
    if min(pix_a, pix_b) < 0.02:
        wta_a = (hm_a > 0.5).astype(np.float32)
        wta_b = (hm_b > 0.5).astype(np.float32)

    # Label-swap optimisation.
    iou_d = compute_iou(wta_a, gt_a) + compute_iou(wta_b, gt_b)
    iou_s = compute_iou(wta_a, gt_b) + compute_iou(wta_b, gt_a)
    if iou_s > iou_d:
        wta_a, wta_b = wta_b, wta_a

    iou_a = compute_iou(wta_a, gt_a)
    iou_b = compute_iou(wta_b, gt_b)
    miou = (iou_a + iou_b) / 2.0

    return {
        "crop_name": crop,
        "iou_a": iou_a,
        "iou_b": iou_b,
        "mean_iou": miou,
        "desc_a": desc_a,
        "desc_b": desc_b,
    }


# --------------------------------------------------------------------- #
#  Main                                                                   #
# --------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-json", default=str(_PROJECT_ROOT / "RWTD_GT.json"))
    p.add_argument("--output-json",
                   default=str(_PROJECT_ROOT / "checkpoints/sam_baseline_verify.json"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--image-size", type=int, default=SAM3_SIZE)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM 3 (frozen, no Qwen, no Bridge)...")
    sam3 = build_sam3(device)
    print(f"SAM 3 loaded. image_size={args.image_size}")

    with open(args.gt_json) as f:
        items = [it for it in json.load(f) if it.get("parse_ok", False)]
    if args.limit:
        items = items[: args.limit]
    print(f"Evaluating {len(items)} RWTD samples (GT descriptions).")

    results = []
    ious = []
    t0 = time.time()

    for i, it in enumerate(items):
        crop = it["crop_name"]
        da = it["parsed"]["desc_a"]
        db = it["parsed"]["desc_b"]
        out = evaluate_one(sam3, crop, da, db,
                           image_size=args.image_size, device=device)
        if out is None:
            print(f"  [{i+1}/{len(items)}] {crop}: missing files, skipping")
            continue
        results.append(out)
        ious.append(out["mean_iou"])

        if (i + 1) % 25 == 0 or i < 3:
            running = float(np.mean(ious))
            elapsed = time.time() - t0
            print(f"  [{i+1:3d}/{len(items)}] {crop:<10s} "
                  f"iou_a={out['iou_a']:.3f} iou_b={out['iou_b']:.3f} "
                  f"miou={out['mean_iou']:.3f}  running_avg={running:.4f}  "
                  f"({elapsed:.0f}s)")

    arr = np.array(ious)
    summary = {
        "n_samples": len(ious),
        "mean_miou": float(arr.mean()) if len(arr) else 0.0,
        "median_miou": float(np.median(arr)) if len(arr) else 0.0,
        "std_miou": float(arr.std()) if len(arr) else 0.0,
        "samples_gt_0_7": int((arr > 0.7).sum()),
        "samples_lt_0_3": int((arr < 0.3).sum()),
        "reference_score": 0.9289,
        "reference_source": "eval_results/sam3_oracle_points/metrics_qwen3_semseg.csv "
                            "(5-description pair-optimizer path)",
        "per_sample": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  FROZEN SAM 3 BASELINE — RWTD Sanity Check")
    print(f"{'='*60}")
    print(f"  Samples:         {summary['n_samples']}")
    print(f"  Mean mIoU:       {summary['mean_miou']:.4f}")
    print(f"  Median mIoU:     {summary['median_miou']:.4f}")
    print(f"  Samples > 0.7:   {summary['samples_gt_0_7']}/{summary['n_samples']}")
    print(f"  Samples < 0.3:   {summary['samples_lt_0_3']}/{summary['n_samples']}")
    print(f"  Reference:       {summary['reference_score']:.4f} "
          f"(5-desc pair-optimizer)")
    print(f"  This run:        1-desc per texture → slight drop expected.")
    print(f"  Time:            {(time.time() - t0) / 60:.1f} min")
    print(f"  Saved:           {out_path}")
    print(f"{'='*60}")

    gap_from_ref = summary["mean_miou"] - 0.9289
    if summary["mean_miou"] < 0.85:
        print(f"\n  WARNING: mIoU ({summary['mean_miou']:.4f}) far below reference "
              f"({0.9289:.4f}). Something is likely off with SAM 3 install, "
              f"image preprocessing, or the forward_text path.")
    else:
        print(f"\n  OK: {gap_from_ref:+.4f} from reference — within expected range.")


if __name__ == "__main__":
    main()
