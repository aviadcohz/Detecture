from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from rwtd_miner.utils.logging import get_logger


def _resolve_device(pref: str) -> str:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _keyword_pass(caption: str, keep_keywords: list[str], reject_keywords: list[str]) -> bool:
    c = (caption or "").lower()
    if any(k.lower() in c for k in reject_keywords):
        return False
    if any(k.lower() in c for k in keep_keywords):
        return True
    return False


def run_stage_c(df: pd.DataFrame, cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> pd.DataFrame:
    log = get_logger("stage_c")
    out = df.copy()
    out["stageC_caption"] = None
    out["stageC_pass"] = None
    out["stageC_error"] = None

    shortlist_cap = int(cfg.get("shortlist_cap", 10000))
    shortlist = out[out["stageB_pass"].fillna(False)].copy()
    shortlist = shortlist.sort_values("stageB_clip_score", ascending=False).head(shortlist_cap)

    if shortlist.empty:
        return out

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except Exception as exc:
        log.warning("Stage C skipped: transformers BLIP unavailable: %s", exc)
        out.loc[shortlist.index, "stageC_error"] = "caption_model_unavailable"
        return out

    model_name = cfg.get("model_name", "Salesforce/blip-image-captioning-base")
    device = _resolve_device(runtime_cfg.get("device_preference", "auto"))
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()

    keep_keywords = list(cfg.get("keep_keywords", []))
    reject_keywords = list(cfg.get("reject_keywords", []))
    batch_size = int(cfg.get("caption_batch_size", 8))

    idxs = shortlist.index.tolist()
    for s in tqdm(range(0, len(idxs), batch_size), desc="stageC_caption", unit="batch"):
        batch_idxs = idxs[s : s + batch_size]
        images = []
        valid = []
        for i in batch_idxs:
            p = Path(str(out.at[i, "image_path"]))
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
                valid.append(i)
            except Exception:
                out.at[i, "stageC_error"] = "image_read_failed"

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=24)
            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for i, cap in zip(valid, captions):
            out.at[i, "stageC_caption"] = cap
            out.at[i, "stageC_pass"] = bool(_keyword_pass(cap, keep_keywords=keep_keywords, reject_keywords=reject_keywords))

    return out
