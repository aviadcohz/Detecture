# Server Start Here (VLM-First Plan)

This file is the canonical handoff context for continuing the project on the RTX3090 server.

## Mission (server phase)

Run large-scale RWTD-like mining with **CLIP + VLM enabled**, across the non-ADE20K scope, while keeping runs resumable, auditable, and reviewable in the static website.

## Current status snapshot

- Website/manual review UX is ready (pick/deselect/export) and deployed.
- ADE20K + RWTD calibration flow exists and works.
- Stage B (CLIP) and Stage D (VLM backends) are implemented in code.
- Server profiles are available:
  - `configs/profiles/server_rtx3090_fast.yaml`
  - `configs/profiles/server_rtx3090_quality.yaml`
- Non-ADE20K scope + estimated runtime documented:
  - `SERVER_NON_ADE20K_SCOPE.md`
- Server dataset planning script exists:
  - `scripts/prepare_non_ade20k_server.sh`

## What is mandatory in server runs

1. VLM must be enabled (no CLIP-only final selection).
2. Use GPU acceleration where available.
3. Keep stage manifests and checkpoints for resume safety.
4. Build/update review site bundle after each major run.

## Non-ADE20K scope for server execution

See `configs/datasets/non_ade20k_registry.yaml` for structured list, priority, access mode, and folder expectations.

## Execution strategy on server (high level)

1. Bootstrap env and verify CUDA + torch + transformers + open_clip.
2. Prepare dataset roots in `/data/rwtd_datasets`.
3. Run public downloads first (COCO) with the provided script.
4. Complete gated datasets (Mapillary/BDD/Cityscapes/IDD/etc.) manually into expected folders.
5. Implement/complete dataset adapters in priority order.
6. Run mining with server profile:
   - fast profile for first full sweep
   - quality profile for final rerank
7. After each dataset completes, publish progress immediately:
   - run `scripts/finalize_dataset_and_publish.sh` for that dataset run dir
   - commit/push private repo updates
   - sync/push public review site repo when available
8. Merge outputs and regenerate review website.
9. Export selected subsets and manifests.

## Optimization policy (server)

- Batch by dataset and stage checkpoints to avoid reruns.
- Cache CLIP embeddings to disk and reuse on reranking.
- Run VLM only on top-N shortlist from Stage B.
- Prefer mixed precision/4-bit quantized VLM when quality is acceptable.
- Use parallel dataloading for CPU-heavy precompute.

## Acceptance criteria before declaring server phase done

- All target datasets in registry either:
  - processed, or
  - explicitly skipped with reason.
- Combined manifest includes final score + stage traceability.
- Review website renders original/mask/overlay for accepted and rejected samples.
- Top subsets exported with deterministic splits.
- Progress board (`progress/datasets_progress_latest.md`) has one completion row per finished dataset.
