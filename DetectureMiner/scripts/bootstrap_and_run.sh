#!/usr/bin/env bash
set -euo pipefail

PROFILE="server_rtx3090_fast"
OUT_DIR=""
ADE_ROOT=""
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

bash scripts/bootstrap_env.sh --profile "$PROFILE"

RUN_ARGS=(--profile "$PROFILE" --out "$OUT_DIR")
if [[ -n "$ADE_ROOT" ]]; then
  RUN_ARGS+=(--ade_root "$ADE_ROOT")
fi
if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
  RUN_ARGS+=(--skip_download)
fi
if [[ "$DRY_RUN" == "true" ]]; then
  RUN_ARGS+=(--dry_run)
fi

bash scripts/run_ade20k_rwtd.sh "${RUN_ARGS[@]}"
