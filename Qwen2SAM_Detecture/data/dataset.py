"""
Multi-texture dataset and collator for Qwen2SAM-Detecture.

Each sample has K (1-5) textures with:
  - Image (PIL)
  - K texture descriptions (strings)
  - Index mask: (H, W) with values {0=dustbin, 1..K=textures}

The collator builds Qwen chat messages using the proven prompt template,
with N="1 to 5" during training to prevent train-test mismatch.
"""

import json
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# ===================================================================== #
#  SAM3 preprocessing (standalone)                                        #
# ===================================================================== #

SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_SIZE = 1008


def preprocess_image_for_sam3(image: np.ndarray, size: int = SAM3_SIZE) -> torch.Tensor:
    """Resize and normalize RGB uint8 image → (3, size, size) float32 tensor."""
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - SAM3_MEAN) / SAM3_STD
    return torch.from_numpy(np.transpose(image, (2, 0, 1)))


# ===================================================================== #
#  V7 Prompt templates — Object-aware with <|seg|> shown in format        #
# ===================================================================== #

SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)

# The {N} slot is "between 1 and 6" at training time; ablation scripts can
# substitute "exactly 2" etc. without rewriting the whole template.
USER_PROMPT_TEMPLATE = (
    "This image contains {N} main visually distinct regions separated by a "
    "boundary (for example, a prominent foreground object and its background, "
    "or contrasting materials).\n\n"
    "Write a single, highly descriptive phrase (approximately 10-15 words) "
    "for each region. Include the following precise information:\n"
    "1. Semantic Name: A natural, common name for the material or object.\n"
    "2. Distinct Visual Features: The core visual attributes like color, "
    "pattern, or texture that strongly contrast with other regions.\n"
    "3. Spatial Context: A brief note on its general position (e.g., "
    "'foreground', 'background', 'top-left').\n\n"
    "IMPORTANT: Describe the ENTIRE region as a collective group, NOT "
    "individual objects within it. Think of each region as a surface/area, "
    "not as a single object.\n\n"
    "Format your response exactly like this for the number of regions "
    "present:\n"
    "TEXTURE_1: Texture of <description> <|seg|>\n"
    "TEXTURE_2: Texture of <description> <|seg|>\n"
    "..."
)

# V7 training prompt: flexible count "between 1 and 6".
TRAIN_USER_PROMPT = USER_PROMPT_TEMPLATE.format(N="between 1 and 6")


def build_assistant_text(descriptions: list[str]) -> str:
    """
    V5: Build assistant response with <|seg|> token after each description.

    The <|seg|> token is a dedicated grounding anchor — Qwen's LoRA learns
    to pack visual-spatial information into its hidden state.

    Example output:
        "TEXTURE_1: Texture of rough mossy stone in the foreground <|seg|>
         TEXTURE_2: Texture of smooth water in the center <|seg|>"
    """
    from models.qwen2sam_detecture import SEG_TOKEN
    lines = []
    for i, desc in enumerate(descriptions):
        lines.append(f"TEXTURE_{i+1}: {desc} {SEG_TOKEN}")
    return "\n".join(lines)


# ===================================================================== #
#  Label creation (standalone — masks system/user tokens)                 #
# ===================================================================== #

def find_assistant_start(ids: torch.Tensor, tokenizer, im_start_id: int) -> int:
    """Locate the index of the first assistant-content token.

    The assistant turn is marked by the last `<|im_start|>` token in the
    sequence (Qwen chat template puts the assistant header at the end of
    the prompt). Skips past "assistant\\n" by scanning for the newline.
    """
    L = ids.shape[0]
    im_starts = (ids == im_start_id).nonzero(as_tuple=True)[0]
    if len(im_starts) == 0:
        return 0
    last_im_start = int(im_starts[-1].item())
    asst_start = last_im_start + 1
    for skip in range(last_im_start + 1, min(last_im_start + 8, L)):
        tok_str = tokenizer.decode([int(ids[skip].item())],
                                    skip_special_tokens=False)
        if "\n" in tok_str:
            asst_start = skip + 1
            break
    return asst_start


def create_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  tokenizer) -> tuple:
    """V7 Exponential LM Loss Weighting with "Shifted Zero" gradient decoupling.

    Base curve (same as before):
      Weight(d) = 1 - exp(-α * d), α = 2.0
      where d = token-wise distance to the nearest assistant-region
      <|seg|> token. A soft "freedom basin" around each SEG.

    Two overrides applied AFTER the base curve — this is the gradient-routing
    fix (V6/prior V7 bug: Qwen was given weight=0 at the SEG target position
    itself, so autoregressive loss never forced emission of <|seg|>):

      (1) FORCE EMISSION.  w[seg_positions] = 1.0
          In causal LM with shift-1 loss, shift_weights[i] = lm_weights[i+1].
          So the weight that drives the model to PREDICT <|seg|> at position
          seg_pos (from the hidden state at seg_pos-1) is lm_weights[seg_pos].
          Setting this to 1.0 gives Qwen full gradient pressure to emit the
          special token at exactly the right place.

      (2) GEOMETRIC FREEDOM (SHIFTED ZERO).  w[seg_positions + 1] = 0.0
          The HIDDEN STATE *of* the <|seg|> token is used by the Bridge to
          drive SAM (via the mask/DICE loss). In the LM path, that same
          hidden state predicts the token at position seg_pos + 1, with
          weight shift_weights[seg_pos] = lm_weights[seg_pos + 1].
          Zeroing lm_weights[seg_pos + 1] removes any LM pressure on SEG's
          hidden state, leaving it free to be shaped purely by the
          segmentation gradient. Bounds-checked against L.

    Concrete weights after overrides (assume d>=3 for "far" tokens):
      position         lm_weights[pos]     controls
      ---------------  -----------------   --------------------------
      seg_pos - 1      ~0.865 (d=1)        pressure ON Qwen to PREDICT seg_pos-1
      seg_pos          1.000 (override 1)  pressure to EMIT <|seg|> at seg_pos
      seg_pos + 1      0.000 (override 2)  ZERO pressure — frees SEG's hidden state
      seg_pos + 2      ~0.982 (d=1)        normal LM pressure resumes

    Note on V7 prompt template: the user prompt contains literal
    "<|seg|>" text which tokenises to the special SEG token. We filter
    seg positions to the assistant region only so prompt-side SEG
    tokens don't corrupt weight computation.

    Returns:
        labels:     (B, L) — token IDs for assistant text, -100 for
                    system/user/prefix/padding.
        lm_weights: (B, L) — per-token exponential + Shifted-Zero weights.
    """
    from models.qwen2sam_detecture import SEG_TOKEN

    ALPHA = 2.0
    seg_token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    B, L = input_ids.shape
    labels = input_ids.clone()
    lm_weights = torch.zeros(B, L, dtype=torch.float32)

    for b in range(B):
        ids = input_ids[b]
        asst_start = find_assistant_start(ids, tokenizer, im_start_id)

        # Only SEG tokens in the assistant region count (V7 prompt also
        # contains <|seg|> as literal text).
        all_seg = (ids == seg_token_id).nonzero(as_tuple=True)[0]
        seg_positions = all_seg[all_seg >= asst_start]
        if len(seg_positions) == 0:
            labels[b, :] = -100
            continue

        # Mask prefix (system/user/assistant header) and padding.
        labels[b, :asst_start] = -100
        labels[b, attention_mask[b] == 0] = -100

        # ---- Base exponential curve (soft freedom basin) -------------- #
        positions = torch.arange(L, dtype=torch.long)
        dists = (positions.unsqueeze(1) - seg_positions.cpu().unsqueeze(0)).abs()
        min_d = dists.min(dim=1).values.float()   # (L,)
        w = 1.0 - torch.exp(-ALPHA * min_d)        # (L,)

        # ---- Shifted-Zero override ------------------------------------ #
        # (1) FORCE EMISSION: weight at SEG's own position = 1.0 so Qwen
        #     learns to predict <|seg|> at exactly this position.
        w[seg_positions] = 1.0

        # (2) GEOMETRIC FREEDOM: weight at position (SEG + 1) = 0.0 so
        #     the hidden state AT the SEG position is never pressured by
        #     "what comes next" LM supervision. Bounds-check against L.
        post_seg = seg_positions + 1
        post_seg = post_seg[post_seg < L]           # drop any that fall off the end
        w[post_seg] = 0.0

        # ---- Standard prefix / padding masking ------------------------ #
        w[:asst_start] = 0.0
        w[attention_mask[b] == 0] = 0.0
        lm_weights[b] = w

    return labels, lm_weights


# ===================================================================== #
#  Dataset                                                                #
# ===================================================================== #

class DetectureDataset(Dataset):
    """
    Multi-texture dataset.

    Expected metadata JSON format (list of dicts):
    [
        {
            "image_path": "/path/to/image.jpg",
            "textures": [
                {"description": "Texture of rough stone...", "mask_path": "/path/to/mask1.png"},
                {"description": "Texture of smooth water...", "mask_path": "/path/to/mask2.png"},
            ]
        },
        ...
    ]

    The index mask is built from per-texture binary masks:
    - Pixel belongs to texture i → value i+1 (1-indexed)
    - Pixel belongs to no texture → value 0 (dustbin)
    """

    def __init__(
        self,
        metadata_path: str,
        image_size: int = SAM3_SIZE,
        augment: bool = False,
        qwen_gt_embeds_path: str = None,
    ):
        with open(metadata_path) as f:
            self.samples = json.load(f)
        self.image_size = image_size
        self.augment = augment

        # Load pre-computed Qwen GT embeddings for self-distillation
        self.gt_embeds = None
        if qwen_gt_embeds_path and Path(qwen_gt_embeds_path).exists():
            self.gt_embeds = torch.load(qwen_gt_embeds_path, map_location="cpu")
            print(f"Loaded {len(self.gt_embeds)} Qwen GT embeddings from {qwen_gt_embeds_path}")

    def _apply_crop(self, image_np, index_mask, descriptions, qwen_gt, k_gt):
        """
        Random crop augmentation to simulate RWTD-style close-up textures.
        Crops image + mask, filters surviving textures, remaps indices contiguously.
        Returns originals unchanged if fewer than 2 textures survive the crop.
        """
        H, W = image_np.shape[:2]
        crop_size = random.randint(300, 600)
        crop_size = min(crop_size, H, W)

        y1 = random.randint(0, H - crop_size)
        x1 = random.randint(0, W - crop_size)
        y2, x2 = y1 + crop_size, x1 + crop_size

        # Crop image and mask
        cropped_img = image_np[y1:y2, x1:x2].copy()
        cropped_mask = index_mask[y1:y2, x1:x2].copy()

        # Find surviving texture indices (non-zero, non-dustbin)
        surviving_ids = sorted(set(int(v) for v in np.unique(cropped_mask) if v > 0))

        # Need at least 2 textures for meaningful boundary signal
        if len(surviving_ids) < 2:
            return image_np, index_mask, descriptions, qwen_gt, k_gt

        # Remap surviving indices to contiguous 1..N
        new_mask = np.zeros_like(cropped_mask)
        new_descriptions = []
        from models.qwen2sam_detecture import MAX_TEXTURES
        new_qwen_gt = torch.zeros(MAX_TEXTURES, 4096)

        for new_idx, old_idx in enumerate(surviving_ids):
            new_mask[cropped_mask == old_idx] = new_idx + 1  # 1-indexed
            new_descriptions.append(descriptions[old_idx - 1])  # old_idx is 1-based
            new_qwen_gt[new_idx] = qwen_gt[old_idx - 1]

        new_k_gt = len(surviving_ids)

        # Resize crop back to image_size
        cropped_img = cv2.resize(cropped_img, (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_LINEAR)
        new_mask_full = cv2.resize(new_mask.astype(np.uint8),
                                    (self.image_size, self.image_size),
                                    interpolation=cv2.INTER_NEAREST).astype(np.int64)

        return cropped_img, new_mask_full, new_descriptions, new_qwen_gt, new_k_gt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        # Load image
        image_np = cv2.imread(meta["image_path"])
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_LINEAR)

        # Load per-texture masks and build index mask (cap at MAX_TEXTURES)
        from models.qwen2sam_detecture import MAX_TEXTURES
        textures = meta["textures"][:MAX_TEXTURES]
        k_gt = len(textures)
        descriptions = []
        index_mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        for i, tex in enumerate(textures):
            descriptions.append(tex["description"])
            mask = cv2.imread(tex["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)
                # Assign texture index (1-based, 0 is dustbin)
                index_mask[mask > 127] = i + 1

        # Load pre-computed Qwen GT embeddings if available
        from models.qwen2sam_detecture import MAX_TEXTURES
        qwen_gt = torch.zeros(MAX_TEXTURES, 4096)
        if self.gt_embeds is not None:
            key = meta["image_path"]
            if key in self.gt_embeds:
                gt = self.gt_embeds[key].float()  # (K, 4096)
                qwen_gt[:gt.shape[0]] = gt

        # Dynamic Macro-to-Micro crop augmentation (30-40% probability)
        # Simulates RWTD-style crops to prevent scene-level overfitting
        if self.augment and random.random() < 0.35:
            image_np, index_mask, descriptions, qwen_gt, k_gt = self._apply_crop(
                image_np, index_mask, descriptions, qwen_gt, k_gt,
            )

        # Build assistant text with <TEX_i> tokens
        assistant_text = build_assistant_text(descriptions)

        # Preprocess for SAM3
        sam_image = preprocess_image_for_sam3(image_np, self.image_size)

        # PIL image for Qwen
        image_pil = Image.fromarray(image_np)

        return {
            "image": image_pil,
            "assistant_text": assistant_text,
            "sam_image": sam_image,
            "index_mask": torch.from_numpy(index_mask),
            "k_gt": k_gt,
            "descriptions": descriptions,
            "qwen_gt_embeds": qwen_gt,
        }


# ===================================================================== #
#  Collator                                                               #
# ===================================================================== #

class DetectureCollator:
    """
    Collate function for Qwen2SAM-Detecture DataLoader.

    Builds Qwen chat messages, tokenizes, creates LM labels,
    and stacks SAM3 inputs + GT masks. CPU-only — no GPU operations.
    CLIP encoding is handled separately in the training loop.
    """

    def __init__(
        self,
        processor,
        inference: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.inference = inference

    def __call__(self, samples: list) -> dict:
        texts = []
        images = []

        for s in samples:
            if self.inference:
                assistant_text = ""
            else:
                assistant_text = s["assistant_text"]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": TRAIN_USER_PROMPT},
                    ],
                },
            ]
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=(not assistant_text),
                )
            )
            images.append(s["image"])

        # Tokenize (CPU only)
        qwen_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )
        qwen_inputs.pop("token_type_ids", None)

        # V6: Labels + proximity-decayed LM weights
        if not self.inference:
            labels, lm_weights = create_labels(
                qwen_inputs["input_ids"],
                qwen_inputs["attention_mask"],
                self.tokenizer,
            )
            # NOTE: labels are NOT passed to qwen_inputs — we compute the
            # weighted LM loss manually in combined_loss using the logits.
            # This allows per-token cosine-decayed weighting instead of
            # binary -100 masking.
            qwen_inputs["labels"] = labels  # still needed for Qwen's internal loss (used as fallback)
        else:
            lm_weights = None

        # Stack SAM3 inputs
        sam_images = torch.stack([s["sam_image"] for s in samples])
        index_masks = torch.stack([s["index_mask"] for s in samples])
        k_gts = torch.tensor([s["k_gt"] for s in samples], dtype=torch.long)

        # Stack Qwen GT embeddings for self-distillation
        qwen_gt_embeds = torch.stack([s["qwen_gt_embeds"] for s in samples])

        # V4: per-sample GT description lists for SAM text encoder distillation
        descriptions = [s["descriptions"] for s in samples]

        batch = {
            "qwen_inputs": qwen_inputs,
            "sam_images": sam_images,
            "index_masks": index_masks,
            "k_gts": k_gts,
            "qwen_gt_embeds": qwen_gt_embeds,
            "descriptions": descriptions,
        }
        if lm_weights is not None:
            batch["lm_weights"] = lm_weights  # (B, L) cosine-decayed per-token weights
        return batch
