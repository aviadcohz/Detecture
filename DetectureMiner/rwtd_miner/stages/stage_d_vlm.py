from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from rwtd_miner.utils.logging import get_logger

ALLOWED_DECISIONS = {"match", "borderline", "not_match"}
ALLOWED_FLAGS = {
    "mosaic_or_collage",
    "synthetic_or_graphic",
    "object_centric",
    "too_many_objects",
    "no_clear_texture_boundary",
    "low_texture_content",
    "text_overlay_or_ui",
    "salient_semantic_subject",
}


@dataclass
class VLMScore:
    score_0_100: int | None
    decision: str | None
    main_reason: str | None
    flags: list[str]
    overlay_score_0_100: int | None = None
    overlay_pass: bool | None = None
    overlay_reason: str | None = None


class BaseVLMScorer:
    def score_image_vlm(self, image_path: Path, overlay_path: Path | None = None) -> VLMScore:
        raise NotImplementedError


class StubVLMScorer(BaseVLMScorer):
    def score_image_vlm(self, image_path: Path, overlay_path: Path | None = None) -> VLMScore:
        _ = image_path, overlay_path
        return VLMScore(score_0_100=None, decision=None, main_reason=None, flags=[])


class ExternalCommandVLMScorer(BaseVLMScorer):
    def __init__(self, command: str, prompt: str, overlay_prompt: str, overlay_score_min: int):
        self.command = command
        self.prompt = prompt
        self.overlay_prompt = overlay_prompt
        self.overlay_score_min = int(overlay_score_min)

    def score_image_vlm(self, image_path: Path, overlay_path: Path | None = None) -> VLMScore:
        payload = {"image_path": str(image_path), "prompt": self.prompt}
        proc = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"External VLM command failed: {proc.stderr.strip()}")

        raw = proc.stdout.strip()
        if not raw:
            raise RuntimeError("External VLM command returned empty output")
        parsed = json.loads(raw)
        out = _normalize_vlm_score(parsed)

        if overlay_path is not None and overlay_path.exists():
            overlay_payload = {
                "image_path": str(overlay_path),
                "prompt": self.overlay_prompt or self.prompt,
            }
            proc_ov = subprocess.run(
                shlex.split(self.command),
                input=json.dumps(overlay_payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                check=False,
            )
            if proc_ov.returncode == 0 and proc_ov.stdout.strip():
                parsed_ov = json.loads(proc_ov.stdout.strip())
                ov = _normalize_vlm_score(parsed_ov)
                ov_score = ov.score_0_100
                ov_pass = bool(ov_score is not None and ov_score >= int(self.overlay_score_min))
                out.overlay_score_0_100 = ov_score
                out.overlay_pass = ov_pass
                out.overlay_reason = ov.main_reason
                if ov_pass is False and "no_clear_texture_boundary" not in out.flags:
                    out.flags.append("no_clear_texture_boundary")
        return out


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


def _parse_yes_no(text: str) -> bool | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    yes_terms = ("yes", "y", "true", "mostly yes", "definitely yes")
    no_terms = ("no", "n", "false", "mostly no", "definitely no")
    if any(t == x or t.startswith(f"{x} ") for x in yes_terms):
        return True
    if any(t == x or t.startswith(f"{x} ") for x in no_terms):
        return False
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return None


def _parse_region_count(text: str) -> int | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    m = re.search(r"\d+", t)
    if m:
        return int(m.group(0))
    word_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    for w, v in word_map.items():
        if w in t:
            return v
    return None


def _extract_first_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty VLM response")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("No JSON object found in VLM response")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed VLM response is not a JSON object")
    return obj


def _normalize_vlm_score(parsed: dict[str, Any]) -> VLMScore:
    score_raw = parsed.get("score_0_100")
    try:
        score = int(round(float(score_raw)))
    except Exception:
        score = None
    if score is not None:
        score = max(0, min(100, score))

    decision = str(parsed.get("decision") or "").strip().lower() if parsed.get("decision") is not None else None
    if decision and decision not in ALLOWED_DECISIONS:
        decision = None
    if decision is None and score is not None:
        if score >= 80:
            decision = "match"
        elif score >= 60:
            decision = "borderline"
        else:
            decision = "not_match"

    reason = str(parsed.get("main_reason") or "").strip() if parsed.get("main_reason") is not None else None
    if not reason:
        reason = None

    flags: list[str] = []
    flags_raw = parsed.get("flags")
    if isinstance(flags_raw, list):
        for f in flags_raw:
            fs = str(f).strip()
            if fs in ALLOWED_FLAGS and fs not in flags:
                flags.append(fs)

    return VLMScore(
        score_0_100=score,
        decision=decision,
        main_reason=reason,
        flags=flags,
    )


def _apply_hard_decision_rules(result: VLMScore, cfg: dict[str, Any]) -> VLMScore:
    force_obj_not_match = bool(cfg.get("force_not_match_on_object_centric", False))
    force_semantic_not_match = bool(cfg.get("force_not_match_on_salient_semantic_subject", True))
    if force_obj_not_match and ("object_centric" in (result.flags or [])):
        score = result.score_0_100
        cap = int(cfg.get("object_centric_score_cap", 59))
        if score is not None:
            score = min(int(score), cap)
        return VLMScore(
            score_0_100=score,
            decision="not_match",
            main_reason=result.main_reason,
            flags=result.flags,
            overlay_score_0_100=result.overlay_score_0_100,
            overlay_pass=result.overlay_pass,
            overlay_reason=result.overlay_reason,
        )
    if force_semantic_not_match and ("salient_semantic_subject" in (result.flags or [])):
        score = result.score_0_100
        cap = int(cfg.get("salient_semantic_subject_score_cap", 45))
        if score is not None:
            score = min(int(score), cap)
        return VLMScore(
            score_0_100=score,
            decision="not_match",
            main_reason=result.main_reason,
            flags=result.flags,
            overlay_score_0_100=result.overlay_score_0_100,
            overlay_pass=result.overlay_pass,
            overlay_reason=result.overlay_reason,
        )

    overlay_enabled = bool(cfg.get("overlay_vlm_validation_enabled", True))
    overlay_required = bool(cfg.get("overlay_vlm_validation_required", True))
    if overlay_enabled:
        ov_pass = result.overlay_pass
        if ov_pass is False:
            score = result.score_0_100
            cap = int(cfg.get("overlay_fail_score_cap", 55))
            penalty = int(cfg.get("overlay_penalty_on_fail", 24))
            if score is not None:
                score = min(int(score), cap)
                score = max(0, score - penalty)
            flags = list(result.flags or [])
            if "no_clear_texture_boundary" not in flags:
                flags.append("no_clear_texture_boundary")
            reason = result.main_reason or "overlay failed"
            if result.overlay_reason:
                reason = f"{reason}; overlay: {result.overlay_reason}"
            return VLMScore(
                score_0_100=score,
                decision="not_match",
                main_reason=reason,
                flags=flags,
                overlay_score_0_100=result.overlay_score_0_100,
                overlay_pass=result.overlay_pass,
                overlay_reason=result.overlay_reason,
            )
        if overlay_required and ov_pass is None:
            score = result.score_0_100
            cap = int(cfg.get("overlay_missing_score_cap", 45))
            if score is not None:
                score = min(int(score), cap)
            flags = list(result.flags or [])
            if "no_clear_texture_boundary" not in flags:
                flags.append("no_clear_texture_boundary")
            reason = result.main_reason or "overlay missing"
            return VLMScore(
                score_0_100=score,
                decision="not_match",
                main_reason=f"{reason}; missing overlay validation",
                flags=flags,
                overlay_score_0_100=result.overlay_score_0_100,
                overlay_pass=False,
                overlay_reason=result.overlay_reason or "missing overlay",
            )
        if ov_pass is True:
            bonus = int(cfg.get("overlay_bonus_on_pass", 6))
            score = result.score_0_100
            if score is not None:
                score = min(100, int(score) + bonus)
            return VLMScore(
                score_0_100=score,
                decision=result.decision,
                main_reason=result.main_reason,
                flags=result.flags,
                overlay_score_0_100=result.overlay_score_0_100,
                overlay_pass=result.overlay_pass,
                overlay_reason=result.overlay_reason,
            )
    return result


def _extract_generated_text_from_pipeline_output(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, dict):
        for key in ("generated_text", "text", "output_text"):
            value = raw_output.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list) and value:
                for item in reversed(value):
                    if isinstance(item, dict):
                        content = item.get("content")
                        if isinstance(content, str) and content.strip():
                            return content
                    elif isinstance(item, str) and item.strip():
                        return item
        return ""

    if isinstance(raw_output, list) and raw_output:
        # Common HF return: list[dict]
        for item in reversed(raw_output):
            txt = _extract_generated_text_from_pipeline_output(item)
            if txt:
                return txt
    return ""


class HfVlmChatScorer(BaseVLMScorer):
    def __init__(self, cfg: dict[str, Any]):
        self.log = get_logger("stage_d_hf_vlm_chat")
        self.cfg = cfg
        self.device = _resolve_device(str(cfg.get("device_preference", "auto")))
        self.model_name = str(cfg.get("hf_vlm_model_name", "Qwen/Qwen2.5-VL-3B-Instruct"))
        self.max_new_tokens = int(cfg.get("hf_vlm_max_new_tokens", 220))
        self.temperature = float(cfg.get("hf_vlm_temperature", 0.0))
        self.load_in_4bit = bool(cfg.get("hf_vlm_load_in_4bit", True))
        self.prompt = str(cfg.get("prompt", "")).strip()
        self.overlay_prompt = str(cfg.get("overlay_prompt", "")).strip()
        self.overlay_score_min = int(cfg.get("overlay_score_min", 65))
        self._pipe = None
        self._init_pipe()

    def _init_pipe(self) -> None:
        from transformers import pipeline

        pipe_kwargs: dict[str, Any] = {
            "task": "image-text-to-text",
            "model": self.model_name,
        }
        model_kwargs: dict[str, Any] = {}

        # On GPU, 4-bit load is the preferred default for 7B-class VLMs.
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            if self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            pipe_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
            pipe_kwargs["device_map"] = "auto"

        if model_kwargs:
            pipe_kwargs["model_kwargs"] = model_kwargs

        self._pipe = pipeline(**pipe_kwargs)
        self.log.info("Stage D backend: hf_vlm_chat (%s) on %s", self.model_name, self.device)

    def _run_inference(self, image_path: Path, prompt: str) -> str:
        if self._pipe is None:
            raise RuntimeError("VLM pipeline is not initialized")

        # Try chat-style call first (works for many newer VLMs).
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            out = self._pipe(
                text=messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=max(1e-4, self.temperature),
            )
            txt = _extract_generated_text_from_pipeline_output(out)
            if txt:
                return txt
        except Exception:
            pass

        # Fallback for non-chat image-text-to-text pipelines.
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        out = self._pipe(
            image=image,
            text=prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=max(1e-4, self.temperature),
        )
        txt = _extract_generated_text_from_pipeline_output(out)
        if not txt:
            raise RuntimeError("HF VLM chat backend produced empty output")
        return txt

    def score_image_vlm(self, image_path: Path, overlay_path: Path | None = None) -> VLMScore:
        strict_prompt = (
            f"{self.prompt}\n\n"
            "Return only valid JSON with fields: score_0_100, decision, main_reason, flags. "
            "Do not wrap in markdown."
        )
        raw_text = self._run_inference(image_path=image_path, prompt=strict_prompt)
        parsed = _extract_first_json_object(raw_text)
        out = _normalize_vlm_score(parsed)

        if overlay_path is not None and overlay_path.exists():
            ov_prompt_base = self.overlay_prompt or self.prompt
            ov_prompt = (
                f"{ov_prompt_base}\n\n"
                "This is a texture-boundary overlay visualization. "
                "Judge only boundary quality: coarse, coherent, region-level transitions (2-4 regions). "
                "Return only valid JSON with fields: score_0_100, decision, main_reason, flags."
            )
            raw_ov = self._run_inference(image_path=overlay_path, prompt=ov_prompt)
            parsed_ov = _extract_first_json_object(raw_ov)
            ov = _normalize_vlm_score(parsed_ov)
            out.overlay_score_0_100 = ov.score_0_100
            out.overlay_pass = bool(ov.score_0_100 is not None and int(ov.score_0_100) >= int(self.overlay_score_min))
            out.overlay_reason = ov.main_reason
            if out.overlay_pass is False and "no_clear_texture_boundary" not in out.flags:
                out.flags.append("no_clear_texture_boundary")
        return out


class HfBlipVqaRubricScorer(BaseVLMScorer):
    def __init__(self, cfg: dict[str, Any]):
        self.log = get_logger("stage_d_blip_vqa")
        self.cfg = cfg
        self.device = _resolve_device(str(cfg.get("device_preference", "auto")))
        self.model_name = str(cfg.get("hf_blip_vqa_model_name", "Salesforce/blip-vqa-base"))
        self.max_new_tokens = int(cfg.get("hf_blip_max_new_tokens", 12))

        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.log.info("Stage D backend: hf_blip_vqa (%s) on %s", self.model_name, self.device)

    @torch.no_grad()
    def _ask(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return str(text).strip().lower()

    def _score_overlay_texture_boundary(self, overlay_image: Image.Image) -> tuple[int | None, bool | None, str | None, list[str]]:
        q_clear = (
            "In this texture-boundary overlay, are there clear coherent boundaries between large regions? "
            "Answer yes or no."
        )
        q_coarse = (
            "Are the boundaries mostly coarse and region-level (not many tiny fragmented lines)? "
            "Answer yes or no."
        )
        q_regions = "How many large regions are separated by boundaries in this overlay? answer a number from 1 to 6."
        q_noise = (
            "Does this overlay look noisy or over-segmented with too many tiny boundary fragments? "
            "Answer yes or no."
        )

        clear_ans = self._ask(overlay_image, q_clear)
        coarse_ans = self._ask(overlay_image, q_coarse)
        regions_ans = self._ask(overlay_image, q_regions)
        noise_ans = self._ask(overlay_image, q_noise)

        is_clear = _parse_yes_no(clear_ans)
        is_coarse = _parse_yes_no(coarse_ans)
        region_count = _parse_region_count(regions_ans)
        is_noisy = _parse_yes_no(noise_ans)

        ov_score = 50.0
        ov_flags: list[str] = []
        reasons: list[str] = []

        if is_clear is True:
            ov_score += 22.0
            reasons.append("clear boundaries")
        elif is_clear is False:
            ov_score -= 28.0
            ov_flags.append("no_clear_texture_boundary")
            reasons.append("unclear boundaries")

        if is_coarse is True:
            ov_score += 16.0
            reasons.append("coarse region-level boundaries")
        elif is_coarse is False:
            ov_score -= 20.0
            ov_flags.append("no_clear_texture_boundary")
            reasons.append("too fine/fragmented boundaries")

        if region_count is not None:
            if 2 <= region_count <= 4:
                ov_score += 16.0
                reasons.append(f"{region_count} large regions")
            elif region_count <= 1:
                ov_score -= 14.0
                ov_flags.append("no_clear_texture_boundary")
            else:
                ov_score -= 12.0
                ov_flags.append("too_many_objects")

        if is_noisy is True:
            ov_score -= 14.0
            ov_flags.append("no_clear_texture_boundary")
            reasons.append("noisy/over-segmented")
        elif is_noisy is False:
            ov_score += 6.0

        ov_score_i = int(max(0, min(100, round(ov_score))))
        ov_min = int(self.cfg.get("overlay_score_min", 65))
        ov_pass = bool(ov_score_i >= ov_min)
        ov_reason = ", ".join(reasons[:3]) if reasons else "overlay validation unavailable"

        dedup_flags: list[str] = []
        for f in ov_flags:
            if f in ALLOWED_FLAGS and f not in dedup_flags:
                dedup_flags.append(f)
        return ov_score_i, ov_pass, ov_reason, dedup_flags

    def score_image_vlm(self, image_path: Path, overlay_path: Path | None = None) -> VLMScore:
        with Image.open(image_path) as im:
            image = im.convert("RGB")

        q_real = "Is this a real natural photograph (not collage, not graphic design)? Answer yes or no."
        q_texture = "Are most pixels textured surfaces or materials rather than distinct objects? Answer yes or no."
        q_boundary = "Is there a clear boundary or transition between two or more texture/material regions? Answer yes or no."
        q_object = "Is the image dominated by a single salient object such as person, car, animal, or product? Answer yes or no."
        q_semantic_subject = (
            "Are prominent animals, people, vehicles, or other clearly recognizable objects visually salient in this image? "
            "Answer yes or no."
        )
        q_regions = "How many large texture/material regions are visible? answer a number from 1 to 5."

        real_ans = self._ask(image, q_real)
        texture_ans = self._ask(image, q_texture)
        boundary_ans = self._ask(image, q_boundary)
        object_ans = self._ask(image, q_object)
        semantic_subject_ans = self._ask(image, q_semantic_subject)
        regions_ans = self._ask(image, q_regions)

        is_real = _parse_yes_no(real_ans)
        is_texture_dom = _parse_yes_no(texture_ans)
        has_boundary = _parse_yes_no(boundary_ans)
        object_dominant = _parse_yes_no(object_ans)
        has_salient_semantic_subject = _parse_yes_no(semantic_subject_ans)
        n_regions = _parse_region_count(regions_ans)

        flags: list[str] = []
        score = 50.0
        reasons: list[str] = []

        if is_real is True:
            score += 10.0
            reasons.append("real photo")
        elif is_real is False:
            score -= 35.0
            flags.extend(["mosaic_or_collage", "synthetic_or_graphic"])
            reasons.append("not real photo")

        if is_texture_dom is True:
            score += 22.0
            reasons.append("texture-dominant")
        elif is_texture_dom is False:
            score -= 24.0
            flags.append("low_texture_content")
            reasons.append("low texture dominance")

        if has_boundary is True:
            score += 20.0
            reasons.append("clear texture boundary")
        elif has_boundary is False:
            score -= 20.0
            flags.append("no_clear_texture_boundary")
            reasons.append("weak boundary")

        if object_dominant is True:
            score -= 30.0
            flags.append("object_centric")
            reasons.append("object-centric")
        elif object_dominant is False:
            score += 10.0
            reasons.append("not object-centric")

        # Strongly reject explicit semantic subjects (animals/people/vehicles/products).
        # Rare keep cases are handled later by strict override logic in final selection.
        if has_salient_semantic_subject is True:
            score -= 34.0
            flags.extend(["salient_semantic_subject", "object_centric"])
            reasons.append("prominent semantic subject")
        elif has_salient_semantic_subject is False:
            score += 4.0

        if n_regions is not None:
            if 2 <= n_regions <= 4:
                score += 12.0
                reasons.append(f"{n_regions} large regions")
            elif n_regions <= 1:
                score -= 10.0
            else:
                score -= 8.0
                flags.append("too_many_objects")

        score_i = int(max(0, min(100, round(score))))
        if score_i >= 80:
            decision = "match"
        elif score_i >= 60:
            decision = "borderline"
        else:
            decision = "not_match"

        # Keep flags stable and unique.
        dedup_flags: list[str] = []
        for f in flags:
            if f not in dedup_flags:
                dedup_flags.append(f)

        overlay_score_i: int | None = None
        overlay_pass: bool | None = None
        overlay_reason: str | None = None
        if overlay_path is not None and overlay_path.exists():
            with Image.open(overlay_path) as ov_im:
                ov_image = ov_im.convert("RGB")
            overlay_score_i, overlay_pass, overlay_reason, ov_flags = self._score_overlay_texture_boundary(ov_image)
            for f in ov_flags:
                if f not in dedup_flags:
                    dedup_flags.append(f)

        reason = ", ".join(reasons[:3]) if reasons else "insufficient VLM signal"
        return VLMScore(
            score_0_100=score_i,
            decision=decision,
            main_reason=reason,
            flags=dedup_flags,
            overlay_score_0_100=overlay_score_i,
            overlay_pass=overlay_pass,
            overlay_reason=overlay_reason,
        )


def build_vlm_scorer(cfg: dict[str, Any]) -> BaseVLMScorer:
    backend = str(cfg.get("backend", "stub"))
    if backend == "stub":
        return StubVLMScorer()
    if backend == "external_command":
        command = str(cfg.get("external_command", "")).strip()
        if not command:
            raise ValueError("stage_d.external_command is required for external_command backend")
        prompt = str(cfg.get("prompt", ""))
        overlay_prompt = str(cfg.get("overlay_prompt", "")).strip()
        overlay_score_min = int(cfg.get("overlay_score_min", 65))
        return ExternalCommandVLMScorer(
            command=command,
            prompt=prompt,
            overlay_prompt=overlay_prompt,
            overlay_score_min=overlay_score_min,
        )
    if backend == "hf_blip_vqa":
        return HfBlipVqaRubricScorer(cfg=cfg)
    if backend == "hf_vlm_chat":
        return HfVlmChatScorer(cfg=cfg)
    raise ValueError(f"Unsupported stage_d backend: {backend}")


def _resolve_overlay_path_for_row(row: pd.Series, batch_dir: Path | None = None) -> Path | None:
    raw = str(row.get("geom_overlay_rel") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    if batch_dir is not None:
        cand = (batch_dir / "review" / raw).resolve()
        if cand.exists():
            return cand
    # Fallback for manifests that stored review-relative paths.
    image_path = Path(str(row.get("image_path") or ""))
    if image_path.exists():
        # Try batch layout inferred from image path location.
        # This only succeeds if geom_overlay_rel was actually a nearby path.
        cand = (image_path.parent / raw).resolve()
        if cand.exists():
            return cand
    return None


def run_stage_d(df: pd.DataFrame, cfg: dict[str, Any], batch_dir: Path | None = None) -> pd.DataFrame:
    log = get_logger("stage_d")
    out = df.copy()
    out["stageD_score_0_100"] = None
    out["stageD_decision"] = None
    out["stageD_flags"] = None
    out["stageD_reason"] = None
    out["stageD_error"] = None
    out["stageD_overlay_score_0_100"] = None
    out["stageD_overlay_pass"] = None
    out["stageD_overlay_reason"] = None

    scorer = build_vlm_scorer(cfg)

    top_n = int(cfg.get("score_top_n_from_stage_b", 2000))
    shortlist = out[out["stageB_pass"].fillna(False)].copy()
    shortlist = shortlist.sort_values("stageB_clip_score", ascending=False).head(top_n)

    if shortlist.empty:
        return out

    for idx, row in tqdm(shortlist.iterrows(), total=len(shortlist), desc="stageD_vlm", unit="img"):
        image_path = Path(str(row["image_path"]))
        overlay_path = _resolve_overlay_path_for_row(row, batch_dir=batch_dir)
        try:
            result = scorer.score_image_vlm(image_path=image_path, overlay_path=overlay_path)
            result = _apply_hard_decision_rules(result=result, cfg=cfg)
            out.at[idx, "stageD_score_0_100"] = result.score_0_100
            out.at[idx, "stageD_decision"] = result.decision
            out.at[idx, "stageD_flags"] = "|".join(result.flags)
            out.at[idx, "stageD_reason"] = result.main_reason
            out.at[idx, "stageD_overlay_score_0_100"] = result.overlay_score_0_100
            out.at[idx, "stageD_overlay_pass"] = result.overlay_pass
            out.at[idx, "stageD_overlay_reason"] = result.overlay_reason
        except Exception as exc:
            out.at[idx, "stageD_error"] = str(exc)
            log.warning("VLM scoring failed for %s: %s", image_path, exc)

    return out
