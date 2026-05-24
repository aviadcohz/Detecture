#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Detecture Resume Context ==="
echo "repo_root: $ROOT_DIR"
echo "hostname: $(hostname)"
echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

printf "\n[git]\n"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "commit: $(git rev-parse --short HEAD)"
  echo "status:"
  git status --short | head -n 40
} || echo "not a git repository"

printf "\n[env]\n"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -V
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count', torch.cuda.device_count())
PY
else
  echo ".venv not found"
fi

printf "\n[profiles]\n"
ls -1 configs/profiles/*.yaml 2>/dev/null || true

printf "\n[handoff docs]\n"
for f in \
  SERVER_START_HERE.md \
  SERVER_MIGRATION.md \
  SERVER_NON_ADE20K_SCOPE.md \
  CODEX_SERVER_HANDOFF.md \
  README.md; do
  [[ -f "$f" ]] && echo "$f"
done

printf "\n[dataset registry]\n"
if [[ -f configs/datasets/non_ade20k_registry.yaml ]]; then
  echo "configs/datasets/non_ade20k_registry.yaml"
  head -n 80 configs/datasets/non_ade20k_registry.yaml
else
  echo "registry not found"
fi

printf "\n[latest summaries]\n"
find . -type f \( -name 'ade20k_eval_summary*.json' -o -name 'calibration_summary.json' -o -name 'bundle_summary.json' \) -print | head -n 20

printf "\n[progress board]\n"
if [[ -f progress/datasets_progress_latest.md ]]; then
  sed -n '1,60p' progress/datasets_progress_latest.md
else
  echo "progress/datasets_progress_latest.md not found yet"
fi
if [[ -f progress/last_notice.txt ]]; then
  echo
  echo "last notice:"
  cat progress/last_notice.txt
fi

printf "\n[github push health]\n"
echo "auth setup script: scripts/setup_server_github_auth.sh"
echo "push check script: scripts/check_server_github_push.sh"
echo "dataset finalize script: scripts/finalize_dataset_and_publish.sh"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "origin: $(git remote get-url origin 2>/dev/null || true)"
fi

printf "\n[next suggested command]\n"
echo "bash scripts/run_detecture_ade20k.sh --profile server_rtx3090_fast --out /path/to/output --skip_download"
echo "bash scripts/finalize_dataset_and_publish.sh --dataset_id <dataset_id> --run_dir <dataset_run_dir> --public_repo /home/<user>/detecture_public_site"

printf "\n[codex bootstrap prompt]\n"
cat <<'EOF'
Read SERVER_START_HERE.md, SERVER_NON_ADE20K_SCOPE.md, and configs/datasets/non_ade20k_registry.yaml.
Then inspect current adapters and implement/execute the highest-priority pending dataset pipeline with VLM enabled, resumable checkpoints, and review-site outputs.
EOF
