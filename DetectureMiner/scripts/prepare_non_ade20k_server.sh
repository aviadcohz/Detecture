#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/rwtd_datasets"
MODE="plan"

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
    *)
      echo "Unknown arg: $1"
      echo "Usage: $0 [--root /data/rwtd_datasets] [--mode plan|download_public]"
      exit 1
      ;;
  esac
done

mkdir -p "$ROOT"

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

plan_line() {
  local dataset="$1"
  local mode="$2"
  local target="$3"
  local notes="$4"
  printf "%-24s | %-14s | %-42s | %s\n" "$dataset" "$mode" "$target" "$notes"
}

log "Preparing non-ADE20K dataset plan at: $ROOT"
echo
printf "%-24s | %-14s | %-42s | %s\n" "dataset" "access_mode" "target_dir" "notes"
printf '%s\n' "$(printf '%.0s-' {1..140})"

plan_line "coco_stuff_panoptic" "http_public" "$ROOT/coco" "Fully scripted below (COCO images + stuff + panoptic anns)."
plan_line "mapillary_vistas_v2" "gated_manual" "$ROOT/mapillary_vistas" "Accept terms + token/login required."
plan_line "bdd100k_seg" "gated_manual" "$ROOT/bdd100k" "Account login + segmented labels pack."
plan_line "cityscapes" "gated_manual" "$ROOT/cityscapes" "Official account required."
plan_line "pascal_context" "manual" "$ROOT/pascal_context" "Context labels vary by source; place VOC+context files."
plan_line "bsds500" "http_public" "$ROOT/bsds500" "Use scripts/prepare_pascal_bsds_alternatives.sh for Berkeley BSR source."
plan_line "idd_segmentation" "gated_manual" "$ROOT/idd" "Official download request flow."
plan_line "kitti_materials" "manual" "$ROOT/kitti_materials" "Use your preferred mirror and keep metadata."
plan_line "sun_rgbd" "manual" "$ROOT/sunrgbd" "RGB + segmentation labels subset only."
plan_line "nyudv2" "manual" "$ROOT/nyudv2" "RGB + semantic labels subset."
plan_line "camvid" "manual" "$ROOT/camvid" "Use trusted mirror with labels."
plan_line "wilddash2" "manual" "$ROOT/wilddash2" "Public benchmark files; organize in images/labels."
plan_line "apollo_scape" "gated_manual" "$ROOT/apolloscape" "Optional extension dataset."

echo
log "Creating expected folder layout"
for d in coco mapillary_vistas bdd100k cityscapes pascal_context bsds500 idd kitti_materials sunrgbd nyudv2 camvid wilddash2 apolloscape; do
  mkdir -p "$ROOT/$d"
done

if [[ "$MODE" == "download_public" ]]; then
  if ! command -v wget >/dev/null 2>&1; then
    echo "wget is required for --mode download_public" >&2
    exit 1
  fi

  log "Downloading public COCO files into $ROOT/coco"
  COCO_DIR="$ROOT/coco"
  mkdir -p "$COCO_DIR"

  download_file "http://images.cocodataset.org/zips/train2017.zip" "$COCO_DIR/train2017.zip"
  download_file "http://images.cocodataset.org/zips/val2017.zip" "$COCO_DIR/val2017.zip"
  download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$COCO_DIR/annotations_trainval2017.zip"
  download_file "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip" "$COCO_DIR/stuff_annotations_trainval2017.zip"
  download_file "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip" "$COCO_DIR/panoptic_annotations_trainval2017.zip"

  log "Unzipping COCO archives (idempotent directories)"
  mkdir -p "$COCO_DIR/raw"
  unzip -n "$COCO_DIR/train2017.zip" -d "$COCO_DIR/raw" >/dev/null || true
  unzip -n "$COCO_DIR/val2017.zip" -d "$COCO_DIR/raw" >/dev/null || true
  unzip -n "$COCO_DIR/annotations_trainval2017.zip" -d "$COCO_DIR/raw" >/dev/null || true
  unzip -n "$COCO_DIR/stuff_annotations_trainval2017.zip" -d "$COCO_DIR/raw" >/dev/null || true
  unzip -n "$COCO_DIR/panoptic_annotations_trainval2017.zip" -d "$COCO_DIR/raw" >/dev/null || true

  log "Public downloads complete (COCO only)."
else
  log "Plan mode only. No downloads were started."
fi

echo
cat <<EOF
Next step on server:
1) Run this script in plan mode:
   bash scripts/prepare_non_ade20k_server.sh --root $ROOT --mode plan
2) Download public-only assets:
   bash scripts/prepare_non_ade20k_server.sh --root $ROOT --mode download_public
3) Complete gated/manual datasets by placing files in the target folders listed above.
4) Start mining once adapters are configured for each dataset.
EOF
