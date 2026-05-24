#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_REPO="$ROOT_DIR"
PUBLIC_REPO=""
DATASET_ID=""
RUN_DIR=""
SOURCE_REVIEW=""
MAX_SAMPLES=260
SEED=42
PUSH_ENABLED=1
COMMIT_ENABLED=1

usage() {
  cat <<EOF
Usage: $0 --dataset_id <id> --run_dir <path> [options]

Required:
  --dataset_id <id>         Dataset key (e.g., coco_stuff, mapillary_vistas_v2)
  --run_dir <path>          Dataset run output directory (contains batches/*/review)

Options:
  --source_review <path>    Explicit review dir; if omitted, newest under run_dir is used
  --code_repo <path>        Private code repo root (default: current repo)
  --public_repo <path>      Public site repo root (optional)
  --max_samples <int>       Max samples in published review bundle (default: 260)
  --seed <int>              Bundle sampling seed (default: 42)
  --no_push                 Commit locally but do not push
  --no_commit               Generate files only; do not commit/push
  -h, --help               Show help

Behavior:
1) Builds/refreshes docs/review bundle from latest dataset review output.
2) Appends dataset completion entry to progress/datasets_progress.jsonl.
3) Re-renders progress/datasets_progress_latest.md board.
4) Commits/pushes private repo, then syncs and pushes public site repo if provided.
EOF
}

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_id)
      DATASET_ID="$2"; shift 2 ;;
    --run_dir)
      RUN_DIR="$2"; shift 2 ;;
    --source_review)
      SOURCE_REVIEW="$2"; shift 2 ;;
    --code_repo)
      CODE_REPO="$2"; shift 2 ;;
    --public_repo)
      PUBLIC_REPO="$2"; shift 2 ;;
    --max_samples)
      MAX_SAMPLES="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --no_push)
      PUSH_ENABLED=0; shift ;;
    --no_commit)
      COMMIT_ENABLED=0; PUSH_ENABLED=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET_ID" || -z "$RUN_DIR" ]]; then
  usage
  exit 1
fi

CODE_REPO="$(cd "$CODE_REPO" && pwd)"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

if [[ -z "$SOURCE_REVIEW" ]]; then
  SOURCE_REVIEW="$(RUN_DIR_ENV="$RUN_DIR" python - <<'PY'
import os
from pathlib import Path
run_dir = Path(os.environ["RUN_DIR_ENV"])
cands = [p for p in run_dir.glob('**/batches/batch_*/review') if p.is_dir()]
if not cands:
    print('')
else:
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(cands[0])
PY
)"
fi

if [[ -z "$SOURCE_REVIEW" ]]; then
  echo "Could not find review directory under run_dir: $RUN_DIR" >&2
  exit 1
fi

SOURCE_REVIEW="$(cd "$SOURCE_REVIEW" && pwd)"
if [[ ! -f "$SOURCE_REVIEW/data.json" ]]; then
  echo "Missing $SOURCE_REVIEW/data.json" >&2
  exit 1
fi

log "dataset_id=$DATASET_ID"
log "run_dir=$RUN_DIR"
log "source_review=$SOURCE_REVIEW"
log "code_repo=$CODE_REPO"

mkdir -p "$CODE_REPO/progress"

PYTHON_BIN="$CODE_REPO/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
log "python_bin=$PYTHON_BIN"

# 1) Rebuild docs/review bundle for quick web verification.
log "Building docs/review bundle"
"$PYTHON_BIN" "$CODE_REPO/scripts/build_pages_bundle.py" \
  --source_review "$SOURCE_REVIEW" \
  --out_review "$CODE_REPO/docs/review" \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --dataset_id "$DATASET_ID" \
  --merge_existing \
  --replace_dataset \
  | tee "$CODE_REPO/docs/review/bundle_stdout.json" >/dev/null

# 2) Create progress entry JSON.
ENTRY_JSON="$(DATASET_ID_ENV="$DATASET_ID" RUN_DIR_ENV="$RUN_DIR" SOURCE_REVIEW_ENV="$SOURCE_REVIEW" CODE_REPO_ENV="$CODE_REPO" "$PYTHON_BIN" - <<'PY'
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

dataset_id = os.environ["DATASET_ID_ENV"]
run_dir = Path(os.environ["RUN_DIR_ENV"])
source_review = Path(os.environ["SOURCE_REVIEW_ENV"])
bundle_summary = Path(os.environ["CODE_REPO_ENV"]) / 'docs' / 'review' / 'bundle_summary.json'

rows = json.loads((source_review / 'data.json').read_text(encoding='utf-8'))
status_counts = Counter(str(r.get('status') or 'unknown') for r in rows)

entry = {
    'dataset_id': dataset_id,
    'finished_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'run_dir': str(run_dir),
    'source_review': str(source_review),
    'source_samples': int(len(rows)),
    'source_status_counts': dict(status_counts),
}
if bundle_summary.exists():
    try:
        entry['published_bundle_summary'] = json.loads(bundle_summary.read_text(encoding='utf-8'))
    except Exception:
        pass
print(json.dumps(entry, ensure_ascii=False))
PY
)"

echo "$ENTRY_JSON" >> "$CODE_REPO/progress/datasets_progress.jsonl"

# 3) Render markdown board.
CODE_REPO_ENV="$CODE_REPO" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

progress_jsonl = Path(os.environ["CODE_REPO_ENV"]) / 'progress' / 'datasets_progress.jsonl'
board_md = Path(os.environ["CODE_REPO_ENV"]) / 'progress' / 'datasets_progress_latest.md'
rows = []
for line in progress_jsonl.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        continue

rows.sort(key=lambda x: str(x.get('finished_at_utc', '')), reverse=True)
rows = rows[:50]

lines = []
lines.append('# Dataset Progress Board')
lines.append('')
lines.append('| finished_at_utc | dataset_id | source_samples | selected | borderline | rejected |')
lines.append('|---|---:|---:|---:|---:|---:|')
for r in rows:
    c = r.get('source_status_counts', {}) or {}
    lines.append(
        f"| {r.get('finished_at_utc','')} | {r.get('dataset_id','')} | {r.get('source_samples','')} | {c.get('selected',0)} | {c.get('borderline',0)} | {c.get('rejected',0)} |"
    )

board_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

PRIVATE_COMMIT=""
PUBLIC_COMMIT=""

# 4) Commit/push private repo.
if [[ "$COMMIT_ENABLED" -eq 1 ]]; then
  pushd "$CODE_REPO" >/dev/null
  git add docs/review progress/datasets_progress.jsonl progress/datasets_progress_latest.md
  if ! git diff --cached --quiet; then
    git commit -m "progress: completed ${DATASET_ID} and refreshed review bundle"
    PRIVATE_COMMIT="$(git rev-parse --short HEAD)"
    if [[ "$PUSH_ENABLED" -eq 1 ]]; then
      git push origin main
    fi
  else
    log "No private-repo changes to commit"
  fi
  popd >/dev/null
fi

# 5) Sync/commit/push public site repo (optional).
if [[ -n "$PUBLIC_REPO" && -d "$PUBLIC_REPO/.git" ]]; then
  PUBLIC_REPO="$(cd "$PUBLIC_REPO" && pwd)"
  log "Syncing review bundle to public repo: $PUBLIC_REPO"
  mkdir -p "$PUBLIC_REPO/review"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$CODE_REPO/docs/review/" "$PUBLIC_REPO/review/"
  else
    rm -rf "$PUBLIC_REPO/review"
    mkdir -p "$PUBLIC_REPO/review"
    cp -a "$CODE_REPO/docs/review/." "$PUBLIC_REPO/review/"
  fi

  if [[ "$COMMIT_ENABLED" -eq 1 ]]; then
    pushd "$PUBLIC_REPO" >/dev/null
    git add review
    if ! git diff --cached --quiet; then
      git commit -m "review: update bundle after ${DATASET_ID}"
      PUBLIC_COMMIT="$(git rev-parse --short HEAD)"
      if [[ "$PUSH_ENABLED" -eq 1 ]]; then
        git push origin main
      fi
    else
      log "No public-repo changes to commit"
    fi
    popd >/dev/null
  fi
fi

NOTICE_FILE="$CODE_REPO/progress/last_notice.txt"
{
  echo "DATASET_COMPLETED_NOTICE"
  echo "dataset_id=$DATASET_ID"
  echo "finished_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "private_commit=${PRIVATE_COMMIT:-none}"
  echo "public_commit=${PUBLIC_COMMIT:-none}"
  echo "review_bundle=$CODE_REPO/docs/review/index.html"
} > "$NOTICE_FILE"

cat "$NOTICE_FILE"

if [[ "$PUSH_ENABLED" -eq 1 && -n "$PUBLIC_REPO" ]]; then
  echo
  echo "Website check: inspect the configured public repo review bundle after push."
fi
