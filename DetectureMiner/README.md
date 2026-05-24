# DetectureMiner — texture-scene mining & annotation

> *Part of the [Detecture](../) monorepo — this is the **data-pipeline** component that mines texture-rich scenes and produces the training / test datasets consumed by the [model](../Qwen2SAM_Detecture/) and scored by the [benchmark](../Qwen2SAM_Detecture_Benchmark/).*

DetectureMiner transforms generic semantic datasets (ADE20K + TextureSAM's
texture-augmented variant) into the high-quality multi-texture training
set Detecture is trained on. It also produces the held-out
**ADE20K_Detecture** in-domain evaluation split.

---

## What it does

```
Source dataset (ADE20K or TextureSAM Textured-ADE20K)
        │
        ▼
  ┌─────────────────────┐
  │ 1. Filter           │   geometry-first scoring (0–100)
  │    (rwtd_miner/)    │   — region count, boundary quality, texture coverage
  └──────────┬──────────┘
             │ images scoring ≥ 65
             ▼
  ┌─────────────────────┐
  │ 2. Extract regions  │   merge ADE20K class-masks into 2–5 geometric
  │                     │   texture regions per image (≥ 1% area each)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ 3. VLM annotate     │   Claude 3.5 Sonnet describes each region
  │    (training_data/) │   in the "Texture of <name>, <features>,
  │                     │   <spatial>" template (10–15 words)
  └──────────┬──────────┘
             │
             ▼
     training set (~14k samples, 3.2 regions avg)
     + ADE20K_Detecture test split (212 samples)
```

The pipeline is described in detail in the paper's Data Ecosystem
section and Appendix B.

---

## Layout

```
DetectureMiner/
├── rwtd_miner/            # filtering + extraction engine (the core miner)
│   ├── cli.py             # primary entry point
│   ├── stages/            # filter → extract → score stage modules
│   ├── dataset_adapters/  # ADE20K / COCO / TextureSAM loaders
│   └── utils/             # geometry scoring, mask merging
├── training_data/
│   ├── scripts/           # VLM-annotation drivers
│   ├── adapters/          # Anthropic / OpenAI client wrappers
│   ├── configs/           # per-source dataset configs
│   └── utils.py           # path normalisation, metadata schema
├── configs/
│   ├── profiles/          # reusable hardware + scoring presets
│   └── generated/         # materialised merges (`<profile>_merged.yaml`)
├── data/                  # raw datasets — NOT in git (see root README)
├── docs/                  # design notes (review/ is gitignored, auto/ too)
└── config.yaml            # top-level defaults
```

---

## Quick-start

```bash
# 1. Place or symlink the raw datasets under data/raw/
# (ADE20K → data/raw/ade20k/ADEChallengeData2016, etc.)
# See root README for Hugging Face download.

# 2. Run the miner with the matching profile
python -m rwtd_miner.cli \
    --config configs/generated/coco_ultra_strict_merged.yaml \
    --stage all

# 3. Run the VLM annotation pass (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python training_data/scripts/annotate.py \
    --config training_data/configs/ade20k.yaml
```

The annotated training set is written as a `metadata.json` per dataset
in the unified schema consumed by
[Qwen2SAM_Detecture](../Qwen2SAM_Detecture/) and
[Qwen2SAM_Detecture_Benchmark](../Qwen2SAM_Detecture_Benchmark/).

---

## Configuration

Every config path uses `Path.home()` / `DETECTURE_DATASETS_ROOT`
expansion — no absolute `/home/...` anywhere. Override the default
dataset root:

```bash
export DETECTURE_DATASETS_ROOT=/mnt/fast_storage/datasets
```

Per-source configs live in `training_data/configs/`:
- [ade20k.yaml](training_data/configs/ade20k.yaml) — vanilla ADE20K
- [ade20k_texturesam.yaml](training_data/configs/ade20k_texturesam.yaml) — TextureSAM's `η<1` variants
- [ADE20K_textured_images.yaml](training_data/configs/ADE20K_textured_images.yaml) — TextureSAM's `η=1` pure-texture variant

---

## Output schema

Each mined image becomes one JSON record:

```json
{
  "image_path": "~/datasets/ADE20k_Detecture/images/xxx.jpg",
  "id": "training_ADE_train_00000001",
  "textures": [
    {
      "description": "Texture of weathered brown concrete, coarse aggregate pattern, lower foreground",
      "mask_path": "~/datasets/ADE20k_Detecture/masks/xxx_region0.png"
    },
    ...
  ]
}
```

The per-image region count is whatever the filtering + extraction
pipeline produced (typically 2–5).

---

## Reproducibility

Every step is deterministic given the same `--seed` and the same
source dataset. The mined training set used for the paper's Detecture
checkpoint is ~14k samples; the exact `metadata.json` is shipped via
the Hugging Face dataset archive (see root README).
