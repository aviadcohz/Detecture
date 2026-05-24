# Server Migration Guide (iMac -> RTX3090 Server)

This guide assumes you copy this full directory as-is to the server.

## 1) Copy the project
From iMac:

```bash
scp -r /path/to/TextureData/rwtd_miner_github_repo <user>@<server>:/home/<user>/
```

On server:

```bash
cd /home/<user>/rwtd_miner_github_repo
```

### GitHub push setup on server (required)

Run this once on server to enable GitHub push from Codex:

```bash
bash scripts/setup_server_github_auth.sh --git-name "<your name>" --git-email "<your email>"
```

Then add the printed SSH public key in GitHub (Settings -> SSH and GPG keys), and verify:

```bash
bash scripts/check_server_github_push.sh
```

## 2) Bootstrap environment
Fast profile (recommended first run):

```bash
bash scripts/bootstrap_env.sh --profile server_rtx3090_fast
```

Quality profile (stronger VLM, slower):

```bash
bash scripts/bootstrap_env.sh --profile server_rtx3090_quality
```

## 3) Run ADE20K with CLIP + VLM
If ADE20K is already local:

```bash
bash scripts/run_ade20k_rwtd.sh \
  --profile server_rtx3090_fast \
  --out /data/rwtd_runs/ade20k_fast \
  --ade_root /data/ADEChallengeData2016 \
  --skip_download
```

If ADE20K is not local, omit `--skip_download` and `--ade_root`.

## 4) Open review website output
After run completes, open:

- `<out>/batches/batch_00000/review/index.html`
- or the synced folder if you used `--sync_html_to`.

## 5) Recommended profile progression
1. `server_rtx3090_fast` for quick validation and threshold checks.
2. `server_rtx3090_quality` for better final ranking quality.

## 6) Publish after each dataset finishes

Run this after each dataset run directory is ready:

```bash
bash scripts/finalize_dataset_and_publish.sh \
  --dataset_id <dataset_id> \
  --run_dir <dataset_run_dir> \
  --public_repo /home/<user>/rwtd_miner_public_site
```

This updates git and website incrementally so progress is reviewable while long runs continue.

## Notes
- Profiles are overlays under `configs/profiles/` merged with `config.yaml`.
- Stage B (CLIP) and Stage D (VLM) are enabled in both server profiles.
- All outputs/manifests remain resumable and checkpointed.
- Non-ADE20K server scope and timing:
  - `SERVER_NON_ADE20K_SCOPE.md`
- Server mission + execution order:
  - `SERVER_START_HERE.md`
