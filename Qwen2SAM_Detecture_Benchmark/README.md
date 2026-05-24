# Qwen2SAM_Detecture_Benchmark — evaluation suite

> *Part of the [Detecture](../) monorepo — this is the **evaluation** component. The **model under evaluation** lives at [../Qwen2SAM_Detecture/](../Qwen2SAM_Detecture/); the **data pipeline** that produced its training set lives at [../DetectureMiner/](../DetectureMiner/).*

Unified benchmark harness that runs **four** method families on **four**
datasets under a strict, dataset-dependent fairness protocol. One command
reproduces every row of the paper's main comparison table.

---

## Methods evaluated

| Name | Pipeline | Role |
| --- | --- | --- |
| **Detecture (ours)** | trained Qwen2SAM_Detecture (`best.pt`) → Bridge → SAM3 | our model |
| **SAM3** | SAM3 text encoder → Semantic Seg Head | baseline — single-prompt |
| **Grounding_SAM3** | Grounding-DINO boxes → SAM3 box-prompt → Semantic Seg Head | baseline — box-driven |
| **SA2VA** | ByteDance/Sa2VA-4B end-to-end VLM | baseline — big VLM |

A fifth row (**TextureSAM**) is included by copying the published paper
numbers; we do not re-run that checkpoint.

---

## Datasets + evaluation protocol

Per-dataset "regime" — controls which oracle hints (if any) each method
receives. Enforced by the `DISPATCH` table inside
[master_runner.py](master_runner.py) so no ad-hoc prompt can silently
leak K_GT back in.

| Dataset | $K$ | Regime | How fairness is enforced |
| --- | :---: | --- | --- |
| **RWTD** (253, natural 2-texture) | 2 | **Oracle K=2** | Detecture + SA2VA: explicit "exactly 2" prompt. SAM3 + Grounding_SAM3: `[m1, -m1]` inverse-mask trick over Softmax, which is mathematically equivalent to a K=2 partition without needing a text prior that can express a count. |
| **STLD** (200, synthetic 2-texture) | 2 | **Oracle K=2** | Same as RWTD. |
| **DeTexture ADE20K** (212, multi-texture) | 1–6 | **Autonomous** | No method is told $K_\text{GT}$. Detecture gets the open-range "between 1 and 6" prompt. Grounding_SAM3 uses DINO top-$N$ above score threshold, cap $N$≤6. SA2VA emits whatever `<p>entity</p>` tags its caption produces. SAM3 is the **one exception** — its proposal decoder has no prior over how many masks to keep, so it gets a top-$K_\text{GT}$ concession (SAM3-specific limitation, matches TextureSAM paper's treatment). |
| **CAID** (3091, single-region) | 1 | K=1 (trivial) | Single-class route; the K-regime distinction is moot. |

Every method's output is scored through the same `metrics_utils.py`
scorer — Softmax + static dustbin + `scipy.optimize.linear_sum_assignment`
on `1 − IoU` + ARI on the predicted partition — so mIoU and ARI are
directly comparable across methods and datasets.

---

## Install

```bash
cd ~/Detecture
pip install -r requirements.txt      # root requirements.txt

# SAM3 — clone separately and point to it (not pip-installable).
git clone https://github.com/facebookresearch/sam3.git ~/sam3
pip install -e ~/sam3
export SAM3_ROOT=~/sam3               # optional; defaults to ~/sam3

# For SA2VA: install a stub flash_attn (no real kernels) so the model loads.
# Minimal stub — a package directory named `flash_attn` with the names listed
# in eval_vlm_end2end.py (index_first_axis, pad_input, unpad_input, etc.).
# Details inline in that file.
```

HuggingFace weights download on first run:
- `Qwen/Qwen3-VL-8B-Instruct`
- `ByteDance/Sa2VA-4B` **pinned to revision `b5ffed22`** (eval_vlm_end2end.py)
- `IDEA-Research/grounding-dino-tiny`
- SAM3 via `build_sam3_image_model(load_from_HF=True)`

---

## Datasets layout

All four datasets use a unified metadata JSON. Paths are read from
[datasets_config.yaml](datasets_config.yaml); the defaults expect:

| Dataset key          | Metadata path                              |
| -------------------- | ------------------------------------------ |
| `CAID`               | `~/datasets/CAID/metadata.json`            |
| `RWTD`               | `~/datasets/RWTD/metadata.json`            |
| `STLD`               | `~/datasets/STLD/metadata.json`            |
| `ADE20k_Detecture`   | `~/datasets/ADE20k_Detecture/metadata.json`|

Override the root with `DETECTURE_DATASETS_ROOT=/some/where` or edit the
config. See the [root README](../README.md#datasets) for Hugging Face
download instructions.

Unified metadata entry:
```json
{
  "image_path": "/abs/path/image.jpg",
  "id": "sample_id",
  "textures": [
    {"description": "…", "mask_path": "/abs/path/mask_0.png"},
    {"description": "…", "mask_path": "/abs/path/mask_1.png"}
  ]
}
```

---

## Run

```bash
# Everything (4 methods × 4 datasets = 16 cells)
python master_runner.py

# One method, all datasets
python master_runner.py --model detecture

# One dataset, all methods (smoke)
python master_runner.py --dataset RWTD --limit 10

# One specific cell
python master_runner.py --model sa2va --dataset STLD --limit 10

# Preview commands without spawning anything
python master_runner.py --dry-run
```

Each (model × dataset) cell spawns as a subprocess — clean GPU state per
model (Detecture alone occupies ~16 GB in bf16 and cannot co-reside with
a second backbone).

Outputs land under `results/<model>/<dataset>/zero_shot_results.json`.
The runner also writes a combined
`results/paper_benchmark_summary.json` with the dispatch metadata and
per-cell summaries.

---

## Paper results — what to expect

With `best.pt` + SAM3 + Grounding-DINO-tiny + SA2VA-4B (rev `b5ffed22`):

| Method           | RWTD mIoU/ARI     | STLD mIoU/ARI     | ADE20K mIoU/ARI   | CAID mIoU/ARI      |
| ---              | ---:              | ---:              | ---:              | ---:               |
| SAM3             | 0.6337 / 0.4427   | 0.5042 / 0.1378   | 0.3194 / 0.3219   | **0.9006 / 0.8169**|
| Grounding_SAM3   | 0.4640 / 0.1959   | 0.4489 / 0.0539   | 0.4518 / 0.4794   | 0.6217 / 0.4043    |
| SA2VA            | 0.3561 / 0.5593   | 0.3739 / 0.7140   | 0.7141 / 0.7011   | 0.7986 / 0.7151    |
| TextureSAM (ref) | 0.4684 / 0.6163   | 0.4677 / 0.6849   | 0.4798 / 0.3566   | 0.6691 / 0.5080    |
| **Detecture (ours)** | **0.8162 / 0.6895** | **0.7441 / 0.6062** | **0.7419 / 0.7138** | 0.7450 / 0.5883 |

Detecture leads mIoU on every multi-texture route. CAID (single
"water surface" class) is where SAM3 wins honestly — noted in the paper
as the expected cost of generality.

Aggregate + paper-ready CSV / LaTeX:
```bash
python aggregate_results.py --csv results/summary.csv --latex results/summary.tex
```

---

## Scripts — what runs when

| Script | Role |
| --- | --- |
| `master_runner.py` | canonical entry point — dispatches every (method × dataset) cell through `REGIMES` + `DISPATCH` |
| `eval_sam3_vanilla.py` | SAM3 text-encoder baseline |
| `eval_grounded_sam3.py` | Grounding-DINO → SAM3 box-prompt baseline |
| `eval_vlm_end2end.py` | Sa2VA (and future VLM backends) end-to-end |
| `eval_qwen2sam_zs.py` | legacy Qwen→SAM3 zero-shot shim — kept for ablation only |
| `steelman/*.py` | K=2 inverse-trick variants used on RWTD/STLD |
| `metrics_utils.py` | shared Softmax+dustbin+Hungarian+ARI scorer (every cell routes through this) |
| `aggregate_results.py` | per-cell JSONs → Markdown / CSV / LaTeX table |

---

## Reproducibility

- All CLI defaults resolve through `Path.home()` / environment variables
  — no absolute paths anywhere in the codebase.
- Sa2VA's HF `revision` is **pinned** to `b5ffed22` in
  [eval_vlm_end2end.py](eval_vlm_end2end.py) so upstream modeling
  changes don't silently alter numbers.
- Each subprocess logs the exact command it ran into the per-cell JSON
  `summary` block for traceability.
