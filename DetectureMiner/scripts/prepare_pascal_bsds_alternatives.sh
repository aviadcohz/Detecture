#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/data/rwtd_datasets"
MODE="plan"

PASCAL_CONTEXT_URL="https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz"
PASCAL_VOC2010_URL="https://thor.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
BSDS500_URL="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

usage() {
  cat <<EOF
Usage: $0 [--root /path/to/data/rwtd_datasets] [--mode plan|probe|download]

Modes:
  plan      Print alternative public sources and expected output layout.
  probe     Send HEAD requests to verify source availability and content length.
  download  Download/extract Pascal Context + VOC2010 + BSDS500 and prepare rwtd_input folders.
EOF
}

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

download_file() {
  local url="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]]; then
    log "exists: $dst"
    return 0
  fi
  log "wget -> $dst"
  wget -c "$url" -O "$dst"
}

link_images_flat() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "$dst_dir"
  find "$src_dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -print0 \
    | while IFS= read -r -d '' f; do
        local name
        name="$(basename "$f")"
        ln -sfn "$f" "$dst_dir/$name"
      done
}

link_ext_flat() {
  local src_dir="$1"
  local dst_dir="$2"
  local ext="$3"
  mkdir -p "$dst_dir"
  find "$src_dir" -type f -name "*.${ext}" -print0 \
    | while IFS= read -r -d '' f; do
        local name
        name="$(basename "$f")"
        ln -sfn "$f" "$dst_dir/$name"
      done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "plan" && "$MODE" != "probe" && "$MODE" != "download" ]]; then
  echo "Unsupported --mode: $MODE" >&2
  usage
  exit 1
fi

mkdir -p "$ROOT"

PASCAL_DIR="$ROOT/pascal_context"
BSDS_DIR="$ROOT/bsds500"
mkdir -p "$PASCAL_DIR" "$BSDS_DIR"

printf "%-18s | %-10s | %-76s\n" "dataset" "mode" "source_url"
printf '%s\n' "$(printf '%.0s-' {1..120})"
printf "%-18s | %-10s | %-76s\n" "pascal_context" "alt_http" "$PASCAL_CONTEXT_URL"
printf "%-18s | %-10s | %-76s\n" "pascal_voc2010" "alt_http" "$PASCAL_VOC2010_URL"
printf "%-18s | %-10s | %-76s\n" "bsds500" "http_public" "$BSDS500_URL"
echo

if [[ "$MODE" == "plan" ]]; then
  cat <<EOF
Planned output structure:
- $PASCAL_DIR/raw/context
- $PASCAL_DIR/raw/voc2010
- $PASCAL_DIR/rwtd_input_pascal_context/images
- $PASCAL_DIR/rwtd_input_pascal_context/annotations
- $BSDS_DIR/raw
- $BSDS_DIR/rwtd_input_bsds500/images/{train,val,test}
- $BSDS_DIR/rwtd_input_bsds500/annotations/{train,val,test}
EOF
  exit 0
fi

if [[ "$MODE" == "probe" ]]; then
  python - <<PY
import requests
urls = {
  "pascal_context": "${PASCAL_CONTEXT_URL}",
  "pascal_voc2010": "${PASCAL_VOC2010_URL}",
  "bsds500": "${BSDS500_URL}",
}
for name, url in urls.items():
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        print(f"{name:16s} status={r.status_code} size={r.headers.get('Content-Length')} url={r.url}")
    except Exception as exc:
        print(f"{name:16s} ERROR {exc}")
PY
  exit 0
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "wget is required for --mode download" >&2
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required for --mode download" >&2
  exit 1
fi

log "Downloading alternative public archives"
download_file "$PASCAL_CONTEXT_URL" "$PASCAL_DIR/trainval.tar.gz"
download_file "$PASCAL_VOC2010_URL" "$PASCAL_DIR/VOCtrainval_03-May-2010.tar"
download_file "$BSDS500_URL" "$BSDS_DIR/BSR_bsds500.tgz"

log "Extracting Pascal Context labels"
mkdir -p "$PASCAL_DIR/raw/context"
tar -xzf "$PASCAL_DIR/trainval.tar.gz" -C "$PASCAL_DIR/raw/context"

log "Extracting VOC2010 RGB images"
mkdir -p "$PASCAL_DIR/raw/voc2010"
tar -xf "$PASCAL_DIR/VOCtrainval_03-May-2010.tar" -C "$PASCAL_DIR/raw/voc2010"

PASCAL_INPUT="$PASCAL_DIR/rwtd_input_pascal_context"
mkdir -p "$PASCAL_INPUT/images" "$PASCAL_INPUT/annotations"
link_images_flat "$PASCAL_DIR/raw/voc2010/VOCdevkit/VOC2010/JPEGImages" "$PASCAL_INPUT/images"
link_ext_flat "$PASCAL_DIR/raw/context/trainval" "$PASCAL_INPUT/annotations" "mat"
ln -sfn "$PASCAL_DIR/raw/context/labels.txt" "$PASCAL_INPUT/annotations/labels.txt"

log "Extracting BSDS500"
mkdir -p "$BSDS_DIR/raw"
tar -xzf "$BSDS_DIR/BSR_bsds500.tgz" -C "$BSDS_DIR/raw"

BSDS_INPUT="$BSDS_DIR/rwtd_input_bsds500"
mkdir -p "$BSDS_INPUT/images/train" "$BSDS_INPUT/images/val" "$BSDS_INPUT/images/test"
mkdir -p "$BSDS_INPUT/annotations/train" "$BSDS_INPUT/annotations/val" "$BSDS_INPUT/annotations/test"

for split in train val test; do
  link_images_flat "$BSDS_DIR/raw/BSR/BSDS500/data/images/$split" "$BSDS_INPUT/images/$split"
  link_ext_flat "$BSDS_DIR/raw/BSR/BSDS500/data/groundTruth/$split" "$BSDS_INPUT/annotations/$split" "mat"
done

PASCAL_N="$(python - <<PY
from pathlib import Path
p = Path("${PASCAL_INPUT}/images")
print(sum(1 for x in p.rglob('*') if x.is_file()))
PY
)"
BSDS_N="$(python - <<PY
from pathlib import Path
p = Path("${BSDS_INPUT}/images")
print(sum(1 for x in p.rglob('*') if x.is_file()))
PY
)"

log "Prepared inputs:"
log "pascal_context images=$PASCAL_N at $PASCAL_INPUT/images"
log "bsds500 images=$BSDS_N at $BSDS_INPUT/images"

cat <<EOF
Next run commands:
1) Merge profile:
   . .venv/bin/activate && python scripts/merge_config.py --base config.yaml --profile configs/profiles/natural_views_cross_dataset_strict.yaml --out configs/generated/natural_views_cross_dataset_strict_merged.yaml
2) Pascal Context:
   . .venv/bin/activate && python -m rwtd_miner.cli run --input_root $PASCAL_INPUT --out /path/to/rwtd_runs/pascal_context_natural_views_vlm_$(date +%Y%m%d_%H%M) --config /path/to/rwtd_miner_github_repo/configs/generated/natural_views_cross_dataset_strict_merged.yaml --batch_budget_gb 5.0
3) BSDS500:
   . .venv/bin/activate && python -m rwtd_miner.cli run --input_root $BSDS_INPUT --out /path/to/rwtd_runs/bsds500_natural_views_vlm_$(date +%Y%m%d_%H%M) --config /path/to/rwtd_miner_github_repo/configs/generated/natural_views_cross_dataset_strict_merged.yaml --batch_budget_gb 5.0
EOF
