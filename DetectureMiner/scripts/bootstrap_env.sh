#!/usr/bin/env bash
set -euo pipefail

PROFILE="imac_cpu"
PYTHON_BIN="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] root=$ROOT_DIR"
echo "[bootstrap] profile=$PROFILE"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

if [[ "$PROFILE" == server_* ]]; then
  echo "[bootstrap] Installing CUDA-enabled PyTorch (cu121)"
  python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
else
  echo "[bootstrap] Installing default PyTorch (local platform)"
  python -m pip install --upgrade torch torchvision torchaudio
fi

python -m pip install -r requirements.txt

if [[ "$PROFILE" == "server_rtx3090_quality" ]]; then
  echo "[bootstrap] Installing bitsandbytes for 4-bit VLM"
  python -m pip install --upgrade bitsandbytes
fi

python - <<'PY'
import importlib
import torch

mods = [
    "numpy", "pandas", "PIL", "yaml", "tqdm", "matplotlib", "scipy",
    "open_clip", "transformers", "datasets", "pycocotools"
]
print("[check] torch:", torch.__version__)
print("[check] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[check] cuda device count:", torch.cuda.device_count())
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[check] ok: {m}")
    except Exception as e:
        print(f"[check] fail: {m}: {e}")
PY

echo "[bootstrap] done"
