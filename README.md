# Detecture

> **Sub-Semantic Language Grounding Bridges Texture Perception and Segmentation.**
> End-to-end VLM-guided multi-texture segmentation that couples
> Qwen3-VL-8B with SAM3 through a learned Bridge and a Shifted-Zero
> LM-loss cliff, trained in ~8.2 M parameters on frozen backbones.

This is the public monorepo for the Detecture paper (NeurIPS 2026).
It bundles three components that together reproduce every number in
the paper's main comparison table:

| Component | Role |
| --- | --- |
| [`Qwen2SAM_Detecture/`](Qwen2SAM_Detecture/) | **Model.** Architecture, training, evaluation scripts, ablation logs. |
| [`Qwen2SAM_Detecture_Benchmark/`](Qwen2SAM_Detecture_Benchmark/) | **Benchmark.** 4-method × 4-dataset evaluation suite with unified fairness protocol. |
| [`DetectureMiner/`](DetectureMiner/) | **Data pipeline.** Filters ADE20K/TextureSAM-Textured-ADE20K and produces the ~14k-sample multi-texture training set. |

Each sub-dir has its own README with deeper details; this file covers
**setup + dataset download + end-to-end reproduction** of the paper
numbers.

---

## Install

```bash
# 1. Clone
git clone https://github.com/aviadcohz/Detecture.git
cd Detecture

# 2. Python env
conda create -n detecture python=3.10 -y
conda activate detecture
pip install -r requirements.txt

# 3. SAM3 — clone separately (not pip-installable) and point to it
git clone https://github.com/facebookresearch/sam3.git ~/sam3
pip install -e ~/sam3
export SAM3_ROOT=~/sam3                 # optional; default is ~/sam3

# 4. (for SA2VA only) install a flash_attn stub so the model loads.
#    The stub has zero real kernels; SA2VA runs with use_flash_attn=False.
#    Details + minimal stub contents in Qwen2SAM_Detecture_Benchmark/README.md.

# 5. Download the trained checkpoint and the four evaluation datasets
#    from Hugging Face (see "Model weights & datasets" below for details).
huggingface-cli download anon-detecture-neurips-2026/Detecture-NeurIPS \
    --local-dir Qwen2SAM_Detecture/checkpoints
mkdir -p ~/datasets
for DS in RWTD STLD ADE20k_Detecture CAID; do
  huggingface-cli download anon-detecture-neurips-2026/$DS \
      --repo-type dataset --local-dir ~/datasets/$DS
done
```

After step 5 you have everything needed to reproduce the paper's main
comparison table — jump straight to
[Quick-start](#quick-start-reproduce-the-papers-main-comparison-table).

---

## Model weights & datasets

All artifacts are hosted on Hugging Face under the
[`anon-detecture-neurips-2026`](https://huggingface.co/anon-detecture-neurips-2026)
organization (code MIT, weights/data CC-BY-4.0):

| Type | Repo | Size |
| --- | --- | ---: |
| Model checkpoint (`best.pt`) | [`anon-detecture-neurips-2026/Detecture-NeurIPS`](https://huggingface.co/anon-detecture-neurips-2026/Detecture-NeurIPS) | 7.5 GB |
| RWTD dataset (253 imgs) | [`anon-detecture-neurips-2026/RWTD`](https://huggingface.co/datasets/anon-detecture-neurips-2026/RWTD) | ~150 MB |
| STLD dataset (200 imgs) | [`anon-detecture-neurips-2026/STLD`](https://huggingface.co/datasets/anon-detecture-neurips-2026/STLD) | ~120 MB |
| ADE20k_Detecture (212 imgs) | [`anon-detecture-neurips-2026/ADE20k_Detecture`](https://huggingface.co/datasets/anon-detecture-neurips-2026/ADE20k_Detecture) | ~80 MB |
| CAID dataset (3091 imgs) | [`anon-detecture-neurips-2026/CAID`](https://huggingface.co/datasets/anon-detecture-neurips-2026/CAID) | ~4 GB |

### Download with `huggingface-cli`

The `huggingface_hub` package is already in [`requirements.txt`](requirements.txt),
so the CLI is on your `PATH` after `pip install -r requirements.txt`.

```bash
# 1. Model checkpoint  →  Qwen2SAM_Detecture/checkpoints/best.pt
huggingface-cli download anon-detecture-neurips-2026/Detecture-NeurIPS \
    --local-dir Qwen2SAM_Detecture/checkpoints

# 2. Verify checkpoint integrity
md5sum Qwen2SAM_Detecture/checkpoints/best.pt
# expected: 1f69377996e487fdc6b70120a42d2b4f

# 3. Evaluation datasets  →  ~/datasets/<NAME>/
#    (or export DETECTURE_DATASETS_ROOT and use that path instead)
mkdir -p ~/datasets
for DS in RWTD STLD ADE20k_Detecture CAID; do
  huggingface-cli download anon-detecture-neurips-2026/$DS \
      --repo-type dataset --local-dir ~/datasets/$DS
done
```

If any repo prompts for authentication, run `huggingface-cli login`
once with a token from <https://huggingface.co/settings/tokens>.

### Alternative: clone with `git` + `git-lfs`

If you prefer `git`, install [git-lfs](https://git-lfs.com) first, then:

```bash
git lfs install
git clone https://huggingface.co/anon-detecture-neurips-2026/Detecture-NeurIPS \
    Qwen2SAM_Detecture/checkpoints
for DS in RWTD STLD ADE20k_Detecture CAID; do
  git clone https://huggingface.co/datasets/anon-detecture-neurips-2026/$DS \
      ~/datasets/$DS
done
```

### Expected dataset layout under `~/datasets/`

After the downloads above, the tree should look like this:

```
~/datasets/
├── RWTD/
│   ├── metadata.json
│   ├── images/
│   └── textures_mask/
├── STLD/
│   ├── metadata.json
│   ├── images/
│   └── masks/
├── ADE20k_Detecture/
│   ├── metadata.json
│   ├── images/
│   └── masks/
└── CAID/
    ├── metadata.json
    ├── images/
    └── masks/
```

Override the root if your datasets live elsewhere:

```bash
export DETECTURE_DATASETS_ROOT=/mnt/fast_storage/datasets
```

All Python / YAML in this repo resolves dataset paths through this
variable (falling back to `~/datasets`). No absolute `/home/...` paths
anywhere.

---

## Quick-start: reproduce the paper's main comparison table

Once `best.pt` is in place and the four datasets are under
`$DETECTURE_DATASETS_ROOT`:

```bash
cd Qwen2SAM_Detecture_Benchmark
python master_runner.py
```

This dispatches every (method × dataset) cell of the paper's main
benchmark — 4 methods (Detecture, SAM3, Grounding_SAM3, SA2VA) × 4
datasets (RWTD, STLD, ADE20K_Detecture, CAID) = 16 cells — as fresh
subprocesses (clean GPU state per model), writes per-cell JSONs under
`results/<model>/<dataset>/zero_shot_results.json`, and prints the
final mIoU / ARI table.

Expected wall time: ~1.5–3 h on a single 40-GB GPU (Detecture's three
cells dominate; SA2VA's ADE20K cell is the longest single step).

Paper table generation:

```bash
python aggregate_results.py --csv results/summary.csv \
                            --latex results/summary.tex
```

**Expected mIoU summary** (ours in bold):

| Method | RWTD | STLD | ADE20K (multi) | CAID |
| --- | ---: | ---: | ---: | ---: |
| SAM3           | 0.6337 | 0.5042 | 0.3194 | **0.9006** |
| Grounding_SAM3 | 0.4640 | 0.4489 | 0.4518 | 0.6217 |
| SA2VA          | 0.3561 | 0.3739 | 0.7141 | 0.7986 |
| TextureSAM     | 0.4684 | 0.4677 | 0.4798 | 0.6691 |
| **Detecture**  | **0.8162** | **0.7441** | **0.7419** | 0.7450 |

Full mIoU + ARI numbers + narrative caption in
[Qwen2SAM_Detecture_Benchmark/README.md](Qwen2SAM_Detecture_Benchmark/README.md#paper-results--what-to-expect).

---

## Fairness protocol at a glance

| Dataset | Regime | What every method gets |
| --- | --- | --- |
| RWTD, STLD (K=2) | **Oracle K=2** | Detecture + SA2VA get an "exactly 2" prompt; SAM3 + Grounding_SAM3 get the mathematical `[m1, −m1]` inverse-mask trick |
| ADE20K_Detecture (K=1..6) | **Autonomous** | No method is told $K_\text{GT}$; everyone runs their natural multi-mask pathway |
| CAID (K=1) | Trivial | Single water-surface class, everyone runs their single-prompt path |

Every cell is scored through the same `metrics_utils.py` — Softmax +
static dustbin + Hungarian + ARI — so mIoU / ARI are directly
comparable across methods and datasets. Full protocol with per-method
script dispatch lives in
[`master_runner.py`](Qwen2SAM_Detecture_Benchmark/master_runner.py)'s
`DISPATCH` table; it fails loudly if any (method, regime) pair
regresses to a K-peeking default.

---

## Citation

If you use Detecture, please cite:

```bibtex
@misc{cohenzada2026detecture,
  title         = {Sub-Semantic Image Segmentation},
  author        = {Cohen Zada, Aviad and Orenstein, Nadav and Avidan, Shai and Oren, Gal},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

Replace `XXXX.XXXXX` with the arXiv ID once assigned.

## License

Released under the **MIT License** — see [LICENSE](LICENSE) for the full
text. You are free to use, modify, and redistribute this code for
research or commercial purposes, subject to attribution.
