# Detecture Request And Design

## Request Summary

The requested end state was:

- create a new GitHub repository
- keep it private
- separate the `1-100` "how texturized is the image?" work into its own repository
- rename that project to `Detecture`
- include the project files and its data
- add documentation that explains not only usage but also the scientific intent and design process

## Why Detecture Is Separate

The source workspace had become a broader umbrella for:

- texture scoring
- review-site generation
- multi-dataset mining
- server handoff and operational workflow

The Detecture request narrowed scope to the specific scoring project:

- the scalar `0-100` texture suitability score
- the ADE20K evaluation path
- the supporting review bundle
- the exported ADE20K site that presented the scoring project

This repository therefore preserves the scoring project as a standalone artifact rather than keeping it embedded in the broader mixed workspace.

## Extraction Boundary

Included in Detecture:

- the current source tree from `rwtd_miner_github_repo`
- the static review bundle under `docs/review/`
- the raw ADE20K benchmark copy
- the ADE20K-only website export snapshot
- progress diagnostics and audit files

Excluded from Detecture:

- `.git` history from the source repo
- `.venv` and machine-local dependency installs
- external run caches under `/path/to/rwtd_runs`
- unrelated existing work under `/path/to/Detecture/`

## Main Design Decisions

### 1. Keep internal module names stable

The internal package name `rwtd_miner` was kept intact.

Reason:

- a package-wide rename would add risk without improving scientific fidelity
- wrappers and top-level docs are enough to present the project as Detecture
- this keeps the extracted project runnable immediately

### 2. Rebrand only project-facing surfaces

Updated or added:

- top-level `README.md`
- `SCIENTIFIC_README.md`
- `DATA_README.md`
- `REQUEST_AND_DESIGN.md`
- Detecture-branded shell wrappers
- Detecture titles in the local HTML landing pages

### 3. Keep the repository private

Reason:

- raw benchmark images are included
- generated review bundles contain copied benchmark imagery
- third-party data terms should not be assumed to permit public redistribution

### 4. Preserve the ADE20K-only site snapshot

The separate website export under `site_exports/ade20k_texture_miner_site/` was preserved as part of the Detecture project record.

Reason:

- it is part of how the project was presented
- it contains project-specific explanatory material and assets
- it should be archived alongside the code that generated it

### 5. Prefer a feasible GitHub repository over an impossible one

The much larger external run-cache tree was not imported.

Reason:

- it is outside the narrow Detecture boundary
- it would make a GitHub-hosted repository impractical
- the included review bundles, diagnostics, and raw ADE benchmark copy capture the project more faithfully than a mixed 28G cache directory

## Snapshot Provenance

- source repo snapshot: `2da18efafe8542a5b88f9d3a15b77a9daa9f1435`
- included ADE20K-only site export snapshot: `6d73c2a67ed41a3d455cddcea0ae4ef3a80d975e`
- extraction date: `2026-03-10`

## Naming

The user explicitly preferred `Detecture` over `Texometer`, so the extracted repository and project-facing materials use `Detecture`.
