#!/usr/bin/env bash
set -euo pipefail

PROFILE="imac_cpu"
OUT_DIR=""
ADE_ROOT=""
SELECTED_MIN="60"
BORDERLINE_MIN="50"
REVIEW_LIMIT="1500"
SYNC_HTML_TO=""
SKIP_DOWNLOAD="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"; shift 2 ;;
    --out)
      OUT_DIR="$2"; shift 2 ;;
    --ade_root)
      ADE_ROOT="$2"; shift 2 ;;
    --selected_min)
      SELECTED_MIN="$2"; shift 2 ;;
    --borderline_min)
      BORDERLINE_MIN="$2"; shift 2 ;;
    --review_limit)
      REVIEW_LIMIT="$2"; shift 2 ;;
    --sync_html_to)
      SYNC_HTML_TO="$2"; shift 2 ;;
    --skip_download)
      SKIP_DOWNLOAD="true"; shift 1 ;;
    --dry_run)
      DRY_RUN="true"; shift 1 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  echo "Missing --out" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: bash scripts/bootstrap_env.sh --profile $PROFILE" >&2
  exit 2
fi

# shellcheck disable=SC1091
source .venv/bin/activate

BASE_CFG="$ROOT_DIR/config.yaml"
PROFILE_CFG="$ROOT_DIR/configs/profiles/${PROFILE}.yaml"
MERGED_CFG="$ROOT_DIR/.tmp/config_${PROFILE}.yaml"
mkdir -p "$ROOT_DIR/.tmp"

python "$ROOT_DIR/scripts/merge_config.py" --base "$BASE_CFG" --profile "$PROFILE_CFG" --out "$MERGED_CFG" >/dev/null

VLM_BACKEND="$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$MERGED_CFG","r",encoding="utf-8"))
print(cfg.get("stage_d",{}).get("backend","hf_blip_vqa"))
PY
)"
VLM_TOP_N="$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$MERGED_CFG","r",encoding="utf-8"))
print(cfg.get("stage_d",{}).get("score_top_n_from_stage_b",1200))
PY
)"
VLM_DEVICE="$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$MERGED_CFG","r",encoding="utf-8"))
print(cfg.get("stage_d",{}).get("device_preference","auto"))
PY
)"
VLM_EXTERNAL_COMMAND="$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$MERGED_CFG","r",encoding="utf-8"))
print(cfg.get("stage_d",{}).get("external_command",""))
PY
)"
VLM_MODEL_NAME="$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$MERGED_CFG","r",encoding="utf-8"))
print(cfg.get("stage_d",{}).get("hf_blip_vqa_model_name","Salesforce/blip-vqa-base"))
PY
)"
VLM_CHAT_MODEL_NAME="$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$MERGED_CFG","r",encoding="utf-8"))
print(cfg.get("stage_d",{}).get("hf_vlm_model_name",""))
PY
)"

CMD=(
  python -m rwtd_miner.cli ade20k_full
  --config "$MERGED_CFG"
  --out "$OUT_DIR"
  --selected_min "$SELECTED_MIN"
  --borderline_min "$BORDERLINE_MIN"
  --review_limit "$REVIEW_LIMIT"
  --enable_clip
  --enable_vlm
  --vlm_backend "$VLM_BACKEND"
  --vlm_top_n "$VLM_TOP_N"
  --vlm_device "$VLM_DEVICE"
)

if [[ -n "$ADE_ROOT" ]]; then
  CMD+=(--ade_root "$ADE_ROOT")
fi
if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
  CMD+=(--skip_download)
fi
if [[ -n "$SYNC_HTML_TO" ]]; then
  CMD+=(--sync_html_to "$SYNC_HTML_TO")
fi
if [[ "$VLM_BACKEND" == "external_command" && -n "$VLM_EXTERNAL_COMMAND" ]]; then
  CMD+=(--vlm_external_command "$VLM_EXTERNAL_COMMAND")
fi
if [[ "$VLM_BACKEND" == "hf_blip_vqa" ]]; then
  CMD+=(--vlm_model_name "$VLM_MODEL_NAME")
fi

echo "[run] profile=$PROFILE"
echo "[run] merged_config=$MERGED_CFG"
echo "[run] vlm_backend=$VLM_BACKEND top_n=$VLM_TOP_N device=$VLM_DEVICE"
if [[ "$VLM_BACKEND" == "hf_blip_vqa" ]]; then
  echo "[run] vlm_model_name=$VLM_MODEL_NAME"
fi
if [[ "$VLM_BACKEND" == "hf_vlm_chat" ]]; then
  echo "[run] hf_vlm_model_name=$VLM_CHAT_MODEL_NAME"
fi
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[run] dry_run=true"
  printf '[run] command:'; printf ' %q' "${CMD[@]}"; echo
  exit 0
fi
"${CMD[@]}"
