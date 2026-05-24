#!/usr/bin/env python3
"""
Step 3: Generate GT texture descriptions using Claude API with vision.

Reads extract_manifest.json, sends each image + overlay to Claude,
and generates descriptions in the Qwen2SAM_Detecture format.

Output: describe_manifest.json (resume-friendly)
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_output_dir

DESCRIPTION_PROMPT = """\
You are generating ground-truth texture region descriptions for a segmentation training dataset.

This image has {n_textures} texture regions. Each region is a visually distinct surface/material area.
The overlay image shows each region highlighted in a different color (Red=1, Green=2, Blue=3, Yellow=4, Cyan=5).

{hints}

For EACH region, write a single descriptive phrase following this EXACT format:
"Texture of <semantic name>, <distinct visual features>, <spatial context>"

Requirements:
- Start with "Texture of "
- 10-15 words total (after "Texture of")
- Include: (1) semantic material/surface name, (2) distinctive visual features (color, pattern, roughness), (3) spatial position (foreground, background, top-left, center, etc.)
- Describe the ENTIRE region as a collective surface, NOT individual objects within it
- Be specific to what you actually see — avoid generic descriptions

Respond with ONLY a JSON object (no markdown, no extra text):
{format_example}
"""


def encode_image_base64(image_path: str) -> str:
    """Read image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext.lstrip("."), "image/jpeg"
    )


def build_prompt(sample: dict) -> str:
    """Build the description prompt for a sample."""
    descriptions = sample.get("existing_descriptions", [])
    n = sample.get("num_textures", len(sample.get("mask_paths", descriptions)))

    # Use class names if available, else existing descriptions
    class_names = sample.get("class_names", [])
    hints_lines = []
    for i in range(n):
        label = chr(ord("A") + i)
        if i < len(class_names) and class_names[i]:
            hints_lines.append(f"- Region {i+1} (color {label}): ADE20K class \"{class_names[i]}\"")
        elif i < len(descriptions) and descriptions[i]:
            hints_lines.append(f"- Region {i+1} (color {label}): \"{descriptions[i]}\"")
    if hints_lines:
        hints = "Region info from semantic labels (use as context):\n" + "\n".join(hints_lines)
    else:
        hints = "(No prior labels — describe what you see in the overlay.)"

    keys = [f'"texture_{chr(ord("a") + i)}"' for i in range(n)]
    format_example = "{" + ", ".join(f'{k}: "Texture of ..."' for k in keys) + "}"

    return DESCRIPTION_PROMPT.format(
        n_textures=n, hints=hints, format_example=format_example
    )


def call_claude_api(client, model: str, image_path: str, overlay_path: str | None,
                    prompt: str, max_tokens: int) -> str:
    """Call Claude API with image(s) and prompt, return text response."""
    content = []

    # Add main image
    img_data = encode_image_base64(image_path)
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": get_media_type(image_path), "data": img_data},
    })
    content.append({"type": "text", "text": "Main image above."})

    # Add overlay if available
    if overlay_path and Path(overlay_path).exists():
        overlay_data = encode_image_base64(overlay_path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": get_media_type(overlay_path), "data": overlay_data},
        })
        content.append({"type": "text", "text": "Overlay showing texture boundaries above."})

    # Add the prompt
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


def parse_description_response(response_text: str, n_textures: int) -> dict | None:
    """Parse Claude's JSON response into a dict of descriptions."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Validate keys
    expected_keys = [f"texture_{chr(ord('a') + i)}" for i in range(n_textures)]
    for key in expected_keys:
        if key not in result:
            return None
        if not isinstance(result[key], str):
            return None
        if not result[key].startswith("Texture of"):
            result[key] = "Texture of " + result[key]

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate GT texture descriptions")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing VLM descriptions (wrapped with 'Texture of' prefix) instead of calling API")
    parser.add_argument("--limit", type=int, default=None, help="Process only N samples")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = load_config(
        str(project_root / "configs" / "base.yaml"),
        str(project_root / "configs" / f"{args.dataset}.yaml"),
    )
    output_dir = get_output_dir(config)

    # Load extract manifest
    extract_path = output_dir / "extract_manifest.json"
    if not extract_path.exists():
        print("ERROR: extract_manifest.json not found. Run 02_extract.py first.")
        sys.exit(1)

    with open(extract_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from extract_manifest.json")

    if args.limit:
        samples = samples[:args.limit]
        print(f"  Limited to {len(samples)} samples")

    # Load existing progress (resume support)
    describe_path = output_dir / "describe_manifest.json"
    completed = {}
    if config["describe"]["resume"] and describe_path.exists():
        with open(describe_path) as f:
            existing = json.load(f)
        completed = {item["id"]: item for item in existing}
        print(f"  Resuming: {len(completed)} already described")

    if args.dry_run:
        # Show a sample prompt
        sample = samples[0]
        prompt = build_prompt(sample)
        print(f"\n--- Sample prompt for {sample['id']} ---")
        print(prompt)
        print("--- End prompt ---")
        print(f"\n[DRY RUN] Would process {len(samples) - len(completed)} samples.")
        return

    # --use-existing: wrap existing VLM descriptions with prefix, no API call
    if args.use_existing:
        prefix = config["describe"]["prefix"]
        results = list(completed.values())
        for sample in samples:
            if sample["id"] in completed:
                continue
            existing = sample.get("existing_descriptions", [])
            n_masks = len(sample.get("mask_paths", []))
            descs = {}
            for i in range(n_masks):
                key = f"texture_{chr(ord('a') + i)}"
                d = existing[i].strip() if i < len(existing) and existing[i] else f"unknown texture region {i+1}"
                if not d.startswith(prefix):
                    d = prefix + d
                descs[key] = d
            results.append({**sample, "descriptions": descs})
        with open(describe_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nUsed existing descriptions for {len(results)} samples (prefix: '{prefix}')")
        print(f"  Output: {describe_path}")
        return

    # Initialize Claude client
    import anthropic
    client = anthropic.Anthropic()
    model = config["describe"]["model"]
    max_tokens = config["describe"]["max_tokens"]

    results = list(completed.values())
    errors = []

    for i, sample in enumerate(samples):
        if sample["id"] in completed:
            continue

        print(f"  [{i+1}/{len(samples)}] {sample['id']}...", end=" ", flush=True)

        prompt = build_prompt(sample)
        n_textures = sample.get("num_textures", len(sample.get("mask_paths", [])))

        try:
            response_text = call_claude_api(
                client, model,
                sample["image_path"],
                sample.get("overlay_path") if config["describe"]["include_overlay"] else None,
                prompt, max_tokens,
            )
            descriptions = parse_description_response(response_text, n_textures)

            if descriptions is None:
                print(f"PARSE ERROR: {response_text[:100]}")
                errors.append({"id": sample["id"], "error": "parse_error", "raw": response_text})
                continue

            result = {**sample, "descriptions": descriptions}
            results.append(result)
            completed[sample["id"]] = result
            print("OK")

            # Save progress after each sample
            with open(describe_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"id": sample["id"], "error": str(e)})
            # Rate limit backoff
            if "rate" in str(e).lower():
                print("  Rate limited, waiting 60s...")
                time.sleep(60)
            continue

    # Final save
    with open(describe_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save errors
    if errors:
        errors_path = output_dir / "describe_errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)

    print(f"\nDescription generation complete:")
    print(f"  Described: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output: {describe_path}")


if __name__ == "__main__":
    main()
