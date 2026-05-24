from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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


@dataclass
class EmbedBatchResult:
    embeddings: np.ndarray
    valid_local_indices: list[int]


class ClipEmbedder:
    def __init__(self, cfg: dict[str, Any], device_pref: str = "auto"):
        self.log = get_logger("ClipEmbedder")
        self.cfg = cfg
        self.device = _resolve_device(device_pref)
        self.backend = "open_clip"

        model_cfg = cfg.get("model", {})
        self._open_clip_model_name = model_cfg.get("open_clip_model_name", "ViT-B-32")
        self._open_clip_pretrained = model_cfg.get("open_clip_pretrained", "laion2b_s34b_b79k")
        self._hf_model_name = model_cfg.get("hf_clip_model_name", "openai/clip-vit-base-patch32")

        self._open_clip = None
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        self._hf_model = None
        self._hf_processor = None

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self._open_clip_model_name,
                pretrained=self._open_clip_pretrained,
                device=self.device,
            )
            tokenizer = open_clip.get_tokenizer(self._open_clip_model_name)
            model.eval()
            self._open_clip = open_clip
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self.backend = "open_clip"
            self.log.info("Stage B backend: open_clip (%s, %s) on %s", self._open_clip_model_name, self._open_clip_pretrained, self.device)
            return
        except Exception as exc:
            self.log.warning("open_clip unavailable, falling back to transformers CLIP: %s", exc)

        from transformers import CLIPModel, CLIPProcessor

        self._hf_model = CLIPModel.from_pretrained(self._hf_model_name)
        self._hf_model.to(self.device)
        self._hf_model.eval()
        self._hf_processor = CLIPProcessor.from_pretrained(self._hf_model_name)
        self.backend = "transformers_clip"
        self.log.info("Stage B backend: transformers CLIP (%s) on %s", self._hf_model_name, self.device)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        if self.backend == "open_clip":
            toks = self._tokenizer(texts).to(self.device)
            emb = self._model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.detach().cpu().numpy().astype(np.float32)

        inputs = self._hf_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        emb = self._hf_model.get_text_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_images(self, image_paths: list[Path]) -> EmbedBatchResult:
        valid_local: list[int] = []

        if self.backend == "open_clip":
            tensors = []
            for i, p in enumerate(image_paths):
                try:
                    with Image.open(p) as im:
                        im_rgb = im.convert("RGB")
                    tensors.append(self._preprocess(im_rgb))
                    valid_local.append(i)
                except Exception:
                    continue
            if not tensors:
                return EmbedBatchResult(embeddings=np.zeros((0, 512), dtype=np.float32), valid_local_indices=[])
            batch = torch.stack(tensors, dim=0).to(self.device)
            emb = self._model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return EmbedBatchResult(embeddings=emb.detach().cpu().numpy().astype(np.float32), valid_local_indices=valid_local)

        images = []
        for i, p in enumerate(image_paths):
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
                valid_local.append(i)
            except Exception:
                continue
        if not images:
            return EmbedBatchResult(embeddings=np.zeros((0, 512), dtype=np.float32), valid_local_indices=[])

        inputs = self._hf_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        emb = self._hf_model.get_image_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return EmbedBatchResult(embeddings=emb.detach().cpu().numpy().astype(np.float32), valid_local_indices=valid_local)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    k = min(max(1, int(k)), scores.size)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int64)


def run_stage_b(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    cache_dir: Path,
    min_short_side: int,
) -> pd.DataFrame:
    log = get_logger("stage_b")
    if len(df) == 0:
        return df

    # Use position-based indexing internally to avoid failures with non-consecutive DataFrame indices.
    out = df.reset_index(drop=True).copy()

    require_stage_a_pass = bool(cfg.get("require_stage_a_pass", True))
    if require_stage_a_pass:
        candidate_mask = out.get("stageA_pass", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    else:
        candidate_mask = pd.Series(True, index=out.index)
    short_side = np.minimum(out["width"].fillna(0).astype(int), out["height"].fillna(0).astype(int))
    candidate_mask &= short_side >= int(min_short_side)

    candidate_idx = np.where(candidate_mask.values)[0]
    if candidate_idx.size == 0:
        out["stageB_pos_score"] = np.nan
        out["stageB_neg_score"] = np.nan
        out["stageB_clip_score"] = np.nan
        out["stageB_rank"] = np.nan
        out["stageB_pass"] = False
        out["stageB_error"] = "no_candidates"
        return out

    embedder = ClipEmbedder(cfg=cfg, device_pref=runtime_cfg.get("device_preference", "auto"))

    pos_queries = list(cfg.get("positive_queries", []))
    neg_queries = list(cfg.get("negative_queries", []))
    if not pos_queries or not neg_queries:
        raise ValueError("Stage B requires non-empty positive_queries and negative_queries")

    pos_text_emb = embedder.encode_text(pos_queries)
    neg_text_emb = embedder.encode_text(neg_queries)
    emb_dim = int(pos_text_emb.shape[1])

    cache_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = cache_dir / "stage_b_image_embeddings.f32"
    emb_mmap = np.memmap(mmap_path, mode="w+", dtype=np.float32, shape=(candidate_idx.size, emb_dim))
    valid_mask = np.zeros(candidate_idx.size, dtype=bool)

    batch_size = int(cfg.get("image_batch_size", 32))
    candidate_paths = [Path(str(out.iloc[i]["image_path"])) for i in candidate_idx]

    for start in tqdm(range(0, len(candidate_paths), batch_size), desc="stageB_embed", unit="batch"):
        end = min(len(candidate_paths), start + batch_size)
        paths = candidate_paths[start:end]
        res = embedder.encode_images(paths)
        if res.embeddings.shape[0] == 0:
            continue
        for local_i, emb in zip(res.valid_local_indices, res.embeddings):
            global_i = start + local_i
            emb_mmap[global_i] = emb
            valid_mask[global_i] = True

    valid_local_idx = np.where(valid_mask)[0]
    pos_scores_local = np.full(candidate_idx.size, np.nan, dtype=np.float32)
    neg_scores_local = np.full(candidate_idx.size, np.nan, dtype=np.float32)
    clip_scores_local = np.full(candidate_idx.size, np.nan, dtype=np.float32)
    pass_local = np.zeros(candidate_idx.size, dtype=bool)

    if valid_local_idx.size > 0:
        emb_valid = np.asarray(emb_mmap[valid_local_idx], dtype=np.float32)
        emb_valid = np.nan_to_num(emb_valid, nan=0.0, posinf=0.0, neginf=0.0)
        pos_text_emb = np.nan_to_num(pos_text_emb, nan=0.0, posinf=0.0, neginf=0.0)
        neg_text_emb = np.nan_to_num(neg_text_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # Re-normalize after numeric sanitation to keep cosine-like behavior stable.
        emb_norm = np.linalg.norm(emb_valid, axis=1, keepdims=True)
        emb_valid = emb_valid / np.clip(emb_norm, 1e-6, None)
        pos_norm = np.linalg.norm(pos_text_emb, axis=1, keepdims=True)
        neg_norm = np.linalg.norm(neg_text_emb, axis=1, keepdims=True)
        pos_text_emb = pos_text_emb / np.clip(pos_norm, 1e-6, None)
        neg_text_emb = neg_text_emb / np.clip(neg_norm, 1e-6, None)

        sim_pos = emb_valid @ pos_text_emb.T
        sim_neg = emb_valid @ neg_text_emb.T

        pos_vals = sim_pos.max(axis=1)
        neg_vals = sim_neg.max(axis=1)
        alpha = float(cfg.get("alpha", 0.7))
        clip_vals = pos_vals - alpha * neg_vals

        pos_scores_local[valid_local_idx] = pos_vals
        neg_scores_local[valid_local_idx] = neg_vals
        clip_scores_local[valid_local_idx] = clip_vals

        threshold = float(cfg.get("clip_score_threshold", 0.20))
        thr_pass = clip_vals >= threshold

        top_k = int(cfg.get("top_k_per_positive_query", 2000))
        top_union_local_valid = set()
        for q in range(sim_pos.shape[1]):
            top_local_valid = _topk_indices(sim_pos[:, q], top_k)
            top_union_local_valid.update(top_local_valid.tolist())

        top_pass = np.zeros_like(thr_pass, dtype=bool)
        if top_union_local_valid:
            top_pass[list(top_union_local_valid)] = True

        combined_pass_valid = thr_pass | top_pass
        pass_local[valid_local_idx] = combined_pass_valid

    out["stageB_pos_score"] = np.nan
    out["stageB_neg_score"] = np.nan
    out["stageB_clip_score"] = np.nan
    out["stageB_rank"] = np.nan
    out["stageB_pass"] = False
    out["stageB_error"] = None

    out.loc[candidate_idx, "stageB_pos_score"] = pos_scores_local
    out.loc[candidate_idx, "stageB_neg_score"] = neg_scores_local
    out.loc[candidate_idx, "stageB_clip_score"] = clip_scores_local
    out.loc[candidate_idx, "stageB_pass"] = pass_local

    valid_global = candidate_idx[valid_local_idx]
    if valid_global.size > 0:
        rank_order = np.argsort(np.nan_to_num(out.loc[valid_global, "stageB_clip_score"].to_numpy(), nan=-1e9))[::-1]
        ranks = np.empty_like(rank_order, dtype=np.int32)
        ranks[rank_order] = np.arange(1, rank_order.size + 1)
        out.loc[valid_global, "stageB_rank"] = ranks

    invalid_global = np.setdiff1d(candidate_idx, valid_global)
    if invalid_global.size > 0:
        out.loc[invalid_global, "stageB_error"] = "image_read_or_embed_failed"

    # Strictness warnings
    pass_rate = float(out["stageB_pass"].mean()) if len(out) else 0.0
    if pass_rate < 0.001:
        log.warning("Stage B appears too strict (<0.1%% pass)")
    if pass_rate > 0.5:
        log.warning("Stage B appears too loose (>50%% pass)")

    emb_mmap.flush()
    return out
