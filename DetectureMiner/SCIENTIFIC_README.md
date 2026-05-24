# Detecture Scientific README

## 1. Project Question

Detecture is built around a single operational question:

"How texturized is this image for real-world texture segmentation?"

The target is not generic image quality, not semantic salience, and not aesthetic texture richness in the artistic sense. The target is a practical score for whether an image naturally supports a segmentation task driven by texture/material transitions rather than object identity.

## 2. What the 0-100 Score Means

The Detecture score is a heuristic suitability index.

- `90-100`: very strong texture-dominant structure, usually with two to four coherent material regions and clear region boundaries
- `70-89`: good texture structure, but with weaker boundaries, minor semantic contamination, or less stable region organization
- `40-69`: mixed scenes where texture evidence exists but object content, ambiguity, clutter, or weak region separation lowers confidence
- `0-39`: object-centric, weakly textured, overly fragmented, or semantically dominant scenes that are poor fits for texture-driven segmentation

The score is not a physical measurement and not a learned calibrated probability. It is a composite ranking score designed to sort images by texture-transition suitability.

## 3. Core Modeling Idea

Detecture is geometry-first.

The system was explicitly designed to avoid collapsing "interesting image" into "good texture segmentation candidate." Many visually busy images are still poor texture examples because they are driven by semantic objects, text, or clutter. Conversely, some visually simple scenes contain excellent material transitions.

The design therefore separates:

- texture-supporting evidence
- semantic counter-evidence
- boundary quality
- region organization

This keeps the score aligned with the intended task instead of drifting toward generic semantic salience.

## 4. Data Inputs

The implementation works from dense segmentation style inputs.

For ADE20K:

- raw RGB image
- semantic annotation mask
- a keyword-based mapping from ADE classes into `TEXTURE_SURFACE`, `OBJECT`, `AMBIGUOUS`, or `IGNORE`

For the broader extracted codebase, similar logic exists for additional datasets, but Detecture as defined here is centered on the texture-score project and the ADE20K benchmark path.

## 5. Pipeline Design

### Stage A: Mask-Structure Prior

Stage A computes a weak but robust prior from region fragmentation statistics:

- number of regions
- largest region ratio
- median region ratio
- fraction of small regions
- entropy of region distribution

This stage rewards scenes that look structurally texture-rich rather than dominated by one semantic foreground object.

In the current implementation, the Stage A score is:

`stageA_rwtd_score = 100 * (0.28*n + 0.24*largest + 0.18*median + 0.18*small + 0.12*entropy)`

where each term is normalized and clipped before aggregation.

### Geometry / Boundary Stage

This is the main discriminative backbone.

The pipeline constructs coarse texture regions and measures:

- number of large texture regions
- number of strong boundaries
- normalized boundary length
- region balance
- object-fraction penalty

This stage is critical because Detecture is intended to score real texture transitions, not just texture presence.

The geometry score is:

`geom_texture_boundary_score = 100 * (0.28*region_term + 0.34*strong_term + 0.26*boundary_term + 0.12*balance) * object_penalty`

### Baseline Final Score

The baseline Detecture score is a calibrated combination that strongly favors geometry:

`review_score = 0.20*stageA + 0.60*geom + tex_bonus + region_bonus + strong_bonus - obj_penalty - amb_penalty - clutter_penalty`

Key design choices:

- geometry gets the largest weight
- texture coverage gets explicit bonuses
- object-heavy scenes are penalized hard
- ambiguous clutter is penalized
- the score prefers two to four coherent regions

### Optional CLIP Stage

CLIP is used as a retrieval-style correction layer:

`clip_score = max(positive similarities) - alpha * max(negative similarities)`

This does not replace the geometry score. It acts as a semantic contrast layer that can slightly improve ranking and filtering.

### Optional VLM Stage

A VLM can score shortlisted images against a structured rubric:

- realism
- texture dominance
- boundary clarity
- semantic subject prominence
- region count

The VLM returns:

- `score_0_100`
- `decision`
- `reason`
- `flags`

It is used conservatively. The code is intentionally designed so Detecture remains geometry-first even when multimodal correction is enabled.

### Multimodal Fusion

When CLIP and VLM are enabled, the final score remains anchored in the base texture score:

- with CLIP: `0.92*base + 0.08*clip100`
- with VLM: `0.90*base + 0.08*clip100 + 0.02*vlm`

Additional penalties and bonuses are then applied from VLM decisions and flags.

## 6. Why This Design Was Chosen

The main failure mode in this problem is semantic drift.

If the model overweights semantics:

- people, cars, buildings, signs, and mixed-object scenes score too highly
- strong texture structure becomes secondary

If the model ignores semantics completely:

- repetitive clutter and noisy fragmentation can look falsely attractive

Detecture resolves that by using geometry as the core signal and semantics only as a correction layer.

## 7. Benchmarking Logic

ADE20K is useful here because it provides dense annotations over a wide variety of natural scenes. That makes it possible to derive proxy region organization, texture occupancy, and object penalties directly from annotations without training a dedicated texture-label model.

The project uses ADE20K as a benchmark substrate, not as a perfect ground-truth texture dataset.

## 8. Outputs

The main generated outputs are:

- per-image scores and stage fields in manifest tables
- selection statuses: `selected`, `borderline`, `rejected`
- review website assets: original image, mask visualization, overlay visualization
- summaries and diagnostics for manual audit

The review bundle is meant to support human inspection of whether the score actually matches texture-segmentation intuition.

## 9. What Detecture Is Good For

- ranking large image corpora by texture-segmentation suitability
- finding candidate images for RWTD-like segmentation benchmarks
- filtering out object-centric scenes before texture-focused evaluation
- building review sets for manual curation

## 10. What Detecture Is Not

Detecture is not:

- a universal texture classifier
- a semantic segmentation model
- a perceptual image quality metric
- an aesthetic score
- a claim that "texture" has a single objective scalar truth

It is a task-driven score for one narrow problem.

## 11. Limitations

- class-to-texture mapping in ADE20K is keyword-driven and therefore imperfect
- the 0-100 scale is heuristic, not a probabilistic confidence calibration
- some scenes with valid texture transitions are still penalized if semantic objects are too prominent
- dense annotations are only proxies for true material boundaries
- the score reflects the current weighting design, not an immutable scientific law

## 12. Reproduction Path

Using the included raw ADE20K copy:

```bash
bash scripts/run_detecture_ade20k.sh \
  --out /path/to/detecture_ade20k_eval \
  --ade_root /path/to/Detecture/data/raw/ade20k/ADEChallengeData2016 \
  --skip_download
```

Generic CLI wrapper:

```bash
bash scripts/detecture_cli.sh ade20k_full \
  --out /path/to/detecture_ade20k_eval \
  --ade_root /path/to/Detecture/data/raw/ade20k/ADEChallengeData2016 \
  --skip_download \
  --enable_clip \
  --enable_vlm
```

## 13. Relevant External References

- ADE20K official benchmark page: `https://sceneparsing.csail.mit.edu/`
- ADE20K official GitHub repository: `https://github.com/CSAILVision/ADE20K`
- included raw download URL used by the code: `https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`

## 14. Scientific Status

Detecture should be treated as a research-and-engineering scoring system, not a finished benchmark standard. Its value is in the explicit design logic, the inspectable review outputs, and the fact that its failure modes are legible enough to audit.
