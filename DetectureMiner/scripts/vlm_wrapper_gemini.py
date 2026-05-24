#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

ALLOWED_DECISIONS = {"match", "borderline", "not_match"}
ALLOWED_FLAGS = {
    "mosaic_or_collage",
    "synthetic_or_graphic",
    "object_centric",
    "too_many_objects",
    "no_clear_texture_boundary",
    "low_texture_content",
    "text_overlay_or_ui",
}


def _read_stdin_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("Expected JSON input on stdin with fields: image_path, prompt")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("stdin JSON must be an object")
    return payload


def _image_to_jpeg_b64(image_path: Path, max_side: int = 1024, jpeg_quality: int = 88) -> tuple[str, str]:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if max_side > 0 and max(w, h) > max_side:
            if w >= h:
                nw = max_side
                nh = int(round(h * (max_side / float(w))))
            else:
                nh = max_side
                nw = int(round(w * (max_side / float(h))))
            rgb = rgb.resize((max(1, nw), max(1, nh)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "image/jpeg", b64


def _extract_text_response(resp_json: dict[str, Any]) -> str:
    candidates = resp_json.get("candidates")
    if not isinstance(candidates, list):
        return ""
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        if texts:
            return "\n".join(texts).strip()
    return ""


def _parse_json_block(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model text response")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Recover first JSON object from markdown or mixed text.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model response")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed response is not a JSON object")
    return obj


def _normalize_output(obj: dict[str, Any]) -> dict[str, Any]:
    score_raw = obj.get("score_0_100")
    try:
        score = int(round(float(score_raw)))
    except Exception:
        score = 0
    score = max(0, min(100, score))

    decision = str(obj.get("decision") or "").strip().lower()
    if decision not in ALLOWED_DECISIONS:
        if score >= 80:
            decision = "match"
        elif score >= 60:
            decision = "borderline"
        else:
            decision = "not_match"

    reason = str(obj.get("main_reason") or "").strip()
    if not reason:
        reason = "No reason provided by external VLM"

    flags_raw = obj.get("flags")
    flags: list[str] = []
    if isinstance(flags_raw, list):
        for f in flags_raw:
            fs = str(f).strip()
            if fs in ALLOWED_FLAGS and fs not in flags:
                flags.append(fs)

    return {
        "score_0_100": score,
        "decision": decision,
        "main_reason": reason,
        "flags": flags,
    }


def _call_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    mime_type: str,
    image_b64: str,
    timeout_s: float,
    max_retries: int,
) -> dict[str, Any]:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Ask Gemini to return strict JSON matching Stage-D schema.
    strict_prompt = (
        f"{prompt.strip()}\n\n"
        "Return only valid JSON with fields: score_0_100, decision, main_reason, flags.\n"
        "Do not wrap in markdown."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": strict_prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "topP": 0.95,
            "maxOutputTokens": 300,
            "responseMimeType": "application/json",
        },
    }

    data = json.dumps(payload).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            body = json.loads(raw)
            text = _extract_text_response(body)
            parsed = _parse_json_block(text)
            return _normalize_output(parsed)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            last_err = RuntimeError(f"Gemini HTTP {e.code}: {body[:500]}")
            # Retry on quota/rate/server issues.
            if e.code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(min(10.0, 1.5 * (attempt + 1)))
                continue
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(min(10.0, 1.5 * (attempt + 1)))
                continue
            break

    if last_err:
        raise last_err
    raise RuntimeError("Gemini call failed with unknown error")


def main() -> int:
    parser = argparse.ArgumentParser(description="External VLM wrapper using Gemini API (remote inference)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model name")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY", help="Environment variable containing API key")
    parser.add_argument("--max_side", type=int, default=1024, help="Resize longer image side before upload")
    parser.add_argument("--jpeg_quality", type=int, default=88, help="JPEG quality for uploaded image")
    parser.add_argument("--timeout_s", type=float, default=45.0, help="HTTP timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries on transient HTTP errors")
    args = parser.parse_args()

    import os

    payload = _read_stdin_payload()
    image_path = Path(str(payload.get("image_path", ""))).expanduser()
    prompt = str(payload.get("prompt", "")).strip()
    if not image_path.exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")
    if not prompt:
        raise ValueError("Missing prompt in stdin payload")

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Set environment variable: {args.api_key_env}")

    mime_type, image_b64 = _image_to_jpeg_b64(
        image_path=image_path,
        max_side=int(args.max_side),
        jpeg_quality=int(args.jpeg_quality),
    )

    result = _call_gemini(
        api_key=api_key,
        model=str(args.model),
        prompt=prompt,
        mime_type=mime_type,
        image_b64=image_b64,
        timeout_s=float(args.timeout_s),
        max_retries=int(args.max_retries),
    )

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
