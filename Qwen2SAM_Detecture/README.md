# Qwen2SAM_Detecture — trained model

> *Part of the [Detecture](../) monorepo — this is the **model** component. The **data pipeline** that produced the training set lives at [../DetectureMiner/](../DetectureMiner/); the **evaluation suite** that scores it against baselines lives at [../Qwen2SAM_Detecture_Benchmark/](../Qwen2SAM_Detecture_Benchmark/).*

This directory contains the **Detecture** (Qwen2SAM-DeTexture) architecture,
training code, evaluation scripts, and ablation logs.

The trained checkpoint (`checkpoints/best.pt`, ~7.5 GB) is **not in git** —
download it from Hugging Face (see [root README](../README.md#model-weights)).

---

## What's inside

```
Qwen2SAM_Detecture/
├── models/
│   ├── qwen2sam_detecture.py   # main end-to-end module (Qwen3-VL + Bridge + SAM3)
│   ├── bridge.py               # 4096→1024 Linear+LN+GELU+Dropout Bridge
│   └── losses.py               # Dice + shifted-zero LM-loss cliff
├── training/
│   ├── train.py                # two-stage curriculum driver
│   ├── monitor.py              # epoch-level mIoU/ARI monitor + plot hooks
│   └── utils.py                # load_config / load_checkpoint
├── data/
│   └── dataset.py              # DetectureDataset + collators + prompt templates
├── configs/
│   └── detecture.yaml          # training + eval defaults (image_size, LR, λ_LM, etc.)
├── scripts/
│   ├── eval_Detecture.py       # canonical multi-dataset E2E evaluation
│   ├── ablation_exact_k2_rwtd.py  # ablation table row generator (RWTD, K=2)
│   ├── evaluate_bridge_oracle.py  # teacher-forced oracle eval
│   ├── verify_sam_baseline.py  # zero-shot SAM3 upper-bound check
│   ├── eval_checkpoint_all.py  # batch-evaluate every checkpoint in a sweep
│   └── smoke_test.py           # 10-sample sanity for CI / reviewers
├── ablation/                   # per-version experiment logs (V1–V7)
├── checkpoints/
│   └── best.pt                 # fetched from HF — NOT in git
└── RWTD_GT.json                # RWTD ground-truth descriptions we generated
```

---

## Instantiation

```python
import torch
from pathlib import Path

from training.utils import load_config, load_checkpoint
from models.qwen2sam_detecture import Qwen2SAMDetecture

cfg = load_config("configs/detecture.yaml")
model = Qwen2SAMDetecture(cfg, device="cuda")
load_checkpoint(model, None, "checkpoints/best.pt", device="cuda")
model.eval()
```

The model expects:
- a preprocessed SAM3-size (1008×1008) image as `sam_images` tensor
- Qwen processor inputs via the standard `USER_PROMPT_TEMPLATE` (found in
  `data/dataset.py`), with `N` substituted per regime:
  - `N="2"` on the K=2 datasets (RWTD, STLD)
  - `N="between 1 and 6"` on autonomous multi-texture (ADE20K_Detecture)
  - `N="1"` on single-region (CAID)

At inference time, call `model.inference_forward(qwen_inputs=..., sam_images=...)`
which returns `mask_logits`, `k_preds`, `pad_mask`, `generated_text`. A working
end-to-end example is [scripts/eval_Detecture.py](scripts/eval_Detecture.py).

---

## Running the canonical evaluation

Every paper cell for the Detecture row of Table 3 comes from:

```bash
# Datasets must live under $DETECTURE_DATASETS_ROOT (default ~/datasets).
# See root README for dataset download.
python scripts/eval_Detecture.py \
    --checkpoint checkpoints/best.pt \
    --config configs/detecture.yaml
```

This runs RWTD → STLD → ADE20K_Detecture → CAID with the right prompt-N
per dataset, writes `checkpoints/test_results/eval_Detecture_best/` with
per-dataset JSONs + `summary.json`, and prints the final mIoU / ARI table.

The RWTD cell has a built-in validation gate: if the run finishes under
0.80 mIoU on RWTD with `best.pt`, the script aborts before touching the
other datasets (expected score ≈ 0.8162).

---

## Re-training from scratch

```bash
python training/train.py --config configs/detecture.yaml
```

Two-stage curriculum, ~30 epochs on a single GPU. The Bridge trains first
(frozen Qwen), then Qwen LoRA + masked-row SEG rows join at epoch 9 under
a 10× Bridge LR decay. See paper §5 and [ablation/](ablation/) for the
provenance of every hyperparameter.

---

## Notes for reviewers

- **No absolute paths** — every default resolves to `Path.home()` /
  `os.environ.get("DETECTURE_DATASETS_ROOT", ...)`. Override via
  environment variable if your datasets live elsewhere.
- **SAM3** is not pip-installable: clone `facebookresearch/sam3` somewhere
  and either set `SAM3_ROOT=<path>` or accept the default `~/sam3`.
- **Deterministic seeds** and fixed `image_size=1008` across all scripts
  so numbers reproduce bit-exact on the same hardware.
- The release scripts were pruned: two debug files removed
  (`debug_qwen_generation.py`, `inspect_sam3_text_encoder.py`) and the
  paper-figure generator (`regenerate_unified_plots.py`) moved into the
  private paper folder.
