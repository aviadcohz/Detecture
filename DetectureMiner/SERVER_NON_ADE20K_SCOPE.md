# Non-ADE20K Scope and Runtime Estimates

This document covers the next dataset scope beyond ADE20K and estimates runtime on:
- **Current iMac**: Apple M4, 8 logical CPUs, 16 GB RAM (CPU-only path)
- **Server**: i7-4930K (12 threads) + RTX 3090 24 GB, 724 GB free disk

## Assumptions used for estimates

- Pipeline mode: Stage A + Stage B + geometry scoring (no Stage C captioning, no Stage D VLM).
- Effective throughput assumptions:
  - iMac: ~1.3 images/sec
  - RTX3090 server: ~12 images/sec
- COCO-Stuff and COCO-Panoptic reuse the same RGB images; panoptic is treated as incremental pass.
- Real wall-clock includes I/O and bookkeeping overhead (not just raw compute).

## Recommended non-ADE20K scan scope

| Dataset | Approx images | iMac Stage A+B (raw hrs) | Server Stage A+B (raw hrs) |
|---|---:|---:|---:|
| COCO-Stuff | 123,000 | 26.3 | 2.9 |
| Mapillary Vistas v2 | 25,000 | 5.3 | 0.6 |
| BDD100K Segmentation | 10,000 | 2.1 | 0.2 |
| Cityscapes (fine+coarse) | 25,000 | 5.3 | 0.6 |
| Pascal Context | 10,000 | 2.1 | 0.2 |
| COCO Panoptic (incremental) | 123,000 | 6.6 | 0.7 |
| IDD Segmentation | 10,000 | 2.1 | 0.2 |
| KITTI-Materials | 2,500 | 0.5 | 0.1 |
| SUN RGB-D | 10,335 | 2.2 | 0.2 |
| NYUDv2 | 1,449 | 0.3 | 0.0 |
| CamVid | 701 | 0.2 | 0.0 |
| WildDash2 | 425 | 0.1 | 0.0 |
| **Total (equivalent)** | **249,160** | **53.2** | **5.8** |

### Realistic wall-clock range (A+B only)
- iMac: **~60-90 hours** (2.5-4 days continuous)
- Server: **~8-14 hours**

### If Stage D VLM reranking is enabled on top 15%
- Candidate volume: ~37,000 images
- Extra time:
  - iMac CPU VLM: **~100-130 hours** (not recommended)
  - Server RTX3090 VLM: **~6-10 hours**
- End-to-end with VLM:
  - iMac: **~7-10 days**
  - Server: **~18-36 hours**

## Download/storage expectations (server)

Approx RGB+annotation footprint for this scope is typically **~300-500 GB** depending chosen variants and mirrors.

Given 724 GB free on server, this is feasible with:
- deduplication of shared COCO RGB,
- staged downloads,
- periodic cache cleanup.

## Download readiness command (server only)

This repo includes a server-side planner/downloader script:

```bash
bash scripts/prepare_non_ade20k_server.sh --root /data/rwtd_datasets --mode plan
bash scripts/prepare_non_ade20k_server.sh --root /data/rwtd_datasets --mode download_public
```

Notes:
- `download_public` currently automates public COCO assets.
- Gated/manual datasets are listed with target folders and required manual steps.
- No downloads should be run on the iMac for this scope.

## Incremental publish per dataset

After each dataset run completes, publish progress immediately:

```bash
bash scripts/finalize_dataset_and_publish.sh \
  --dataset_id <dataset_id> \
  --run_dir <dataset_run_dir> \
  --public_repo /home/<user>/rwtd_miner_public_site
```

This updates:
- private repo progress logs,
- public review website bundle,
- and a completion notice file (`progress/last_notice.txt`).

## Adapter status

Current codebase has production flow for ADE20K/RWTD and SA-1B path.
For the 12 non-ADE20K datasets above, adapter-level ingestion is the next step during server migration.
