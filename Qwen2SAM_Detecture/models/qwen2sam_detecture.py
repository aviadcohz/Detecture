"""
Qwen2SAM-Detecture V5: End-to-End VLM-Guided Multi-Texture Segmentation.

Architecture (V5 — [SEG] Token + Bottleneck Projector):
  Qwen3-VL-8B (LoRA r=8) → generates "TEXTURE_N: <desc> <|seg|>" lines
    → Extract hidden state at each <|seg|> token position
    → Build 7 query slots: [DUSTBIN, SEG_1, ..., SEG_K, PAD...]
    → Bottleneck Projector (4096 → 512 → 256)   [~2M params]
    → Batch Multiplexed pass through SAM3:
        Fusion Encoder (frozen + Orthogonal LoRA)
      → SegHead cross_attend_prompt (frozen + Orthogonal LoRA)
      → Pixel Decoder (frozen) → pixel_embed (B, 256, H, W)
      → Multi-texture mask head: einsum(queries, pixel_embed) → (B, 7, H, W)

Key V5 changes vs V4:
  - [SEG] token decouples visual grounding from language context
  - Qwen LoRA (r=8) trained with loss masking (-100 on text, CE only on [SEG])
  - Bottleneck projector (~2M params) prevents domain-specific overfitting
  - No Architectural Plug, no distillation loss — mask loss is the sole signal
  - Gradient flow: Mask Loss → SAM → Projector → [SEG] hidden states → Qwen LoRA

Channel layout: [0=DUSTBIN, 1..K=textures, K+1..6=PAD]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from models.bridge import BridgeProjector


# ===================================================================== #
#  Constants                                                              #
# ===================================================================== #

MAX_TEXTURES = 6
NUM_QUERY_SLOTS = MAX_TEXTURES + 1  # 6 textures + 1 dustbin = 7

SEG_TOKEN = "<|seg|>"


# ===================================================================== #
#  Multi-Texture Mask Head                                                #
# ===================================================================== #

class MultiTextureMaskHead(nn.Module):
    """
    Dot-product mask head: projects queries and pixel features into a shared
    space, then computes masks via einsum.
    """

    def __init__(self, embed_dim: int = 256, mask_dim: int = 256):
        super().__init__()
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, mask_dim),
        )
        self.pixel_head = nn.Conv2d(embed_dim, mask_dim, kernel_size=1)

    def forward(self, queries, pixel_embed):
        q = self.query_mlp(queries)          # (B, N, mask_dim)
        p = self.pixel_head(pixel_embed)     # (B, mask_dim, H, W)
        return torch.einsum("bqc, bchw -> bqhw", q, p)


# ===================================================================== #
#  Qwen utilities                                                         #
# ===================================================================== #

def load_qwen_processor(model_name: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_name)


def load_qwen_model(model_name: str, dtype=torch.bfloat16):
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype,
    )


def apply_qwen_lora(model, cfg: dict):
    """Apply lightweight LoRA to Qwen for [SEG] token training."""
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def add_seg_token(processor, model):
    """Add the <|seg|> special token to the tokenizer and resize embeddings."""
    tokenizer = processor.tokenizer
    num_added = tokenizer.add_tokens([SEG_TOKEN], special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    seg_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    print(f"  [SEG] token: '{SEG_TOKEN}' → id {seg_id}")
    return seg_id


# ===================================================================== #
#  SAM3 loading                                                           #
# ===================================================================== #

def load_sam3(model_cfg: dict, device):
    """Load SAM3 image model."""
    import sam3 as sam3_module
    from sam3.model_builder import build_sam3_image_model

    bpe_path = str(
        Path(sam3_module.__path__[0]) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    checkpoint_path = model_cfg.get("sam3_checkpoint", None)
    return build_sam3_image_model(
        bpe_path=bpe_path,
        eval_mode=False,
        checkpoint_path=checkpoint_path,
        load_from_HF=(checkpoint_path is None),
        enable_segmentation=True,
        device=device,
    )


# ===================================================================== #
#  Main Model                                                             #
# ===================================================================== #

class Qwen2SAMDetecture(nn.Module):
    """
    V5 End-to-End multi-texture segmentation model.

    Qwen3-VL-8B generates texture descriptions with <|seg|> tokens.
    LoRA (r=8) is trained with loss masking: -100 on all text tokens,
    CE only on <|seg|> tokens. Segmentation gradients flow back through
    the bottleneck projector into the <|seg|> hidden states and Qwen LoRA.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        model_cfg = cfg["model"]

        # ---- Qwen3-VL with LoRA ---------------------------------------- #
        qwen_dtype = getattr(torch, model_cfg.get("qwen_dtype", "bfloat16"))
        self.processor = load_qwen_processor(model_cfg["qwen_model"])
        self.qwen = load_qwen_model(model_cfg["qwen_model"], dtype=qwen_dtype)

        # Add [SEG] token
        self.seg_token_id = add_seg_token(self.processor, self.qwen)

        # Apply lightweight LoRA (r=8) — trained via [SEG]-only loss masking
        self.qwen = apply_qwen_lora(self.qwen, model_cfg)
        if model_cfg.get("gradient_checkpointing", True):
            self.qwen.enable_input_require_grads()
            self.qwen.gradient_checkpointing_enable()
        self.qwen.to(self.device)

        # ---- V7 fix: trainable SEG row in embed + lm_head ---------------- #
        # The <|seg|> token is appended AFTER Qwen's pretraining, so its
        # input-embedding and lm_head rows are randomly initialised. LoRA on
        # q/v never reaches those rows, so SEG can never win argmax during
        # generation — live inference silently emits <|tool_call|> / <|im_end|>
        # instead. We warm-start the SEG row from a well-trained reference
        # token (<|im_end|>) and install a per-row gradient mask so ONLY the
        # SEG row receives updates. The rest of the vocabulary stays frozen.
        self._seg_row_params = self._enable_seg_row_training(
            ref_token=model_cfg.get("seg_warmstart_token", "<|im_end|>"),
        )

        # ---- SAM3 (100% frozen in V7 — no LoRA) ------------------------ #
        self.sam3 = load_sam3(model_cfg, self.device)
        self._freeze_sam3()
        self.sam3_lora_modules = []  # retained for save/load back-compat; empty
        self.sam3.to(device=self.device)

        # ---- LLM dim --------------------------------------------------- #
        qwen_cfg = getattr(self.qwen.config, "text_config", self.qwen.config)
        self.llm_dim = qwen_cfg.hidden_size  # 4096 for Qwen3-VL-8B

        # ---- V7 Bridge Projector: 4096 → 1024 → (frozen resizer) → 256 - #
        self.bridge = BridgeProjector(
            llm_dim=self.llm_dim,
            sam_text_width=model_cfg.get("projector_hidden_dim", 1024),
            dropout=model_cfg.get("projector_dropout", 0.4),
        ).to(self.device)
        # Reuse SAM3's pretrained 1024→256 text projection; already frozen.
        self._sam_resizer = self.sam3.backbone.language_backbone.resizer

        # ---- Learnable DUSTBIN embedding -------------------------------- #
        self.dustbin_embed = nn.Parameter(
            torch.randn(1, 1, self.llm_dim, device=self.device) * 0.02
        )

        # ---- Multi-texture mask head ------------------------------------ #
        self.mask_head = MultiTextureMaskHead(
            embed_dim=256, mask_dim=256,
        ).to(self.device)

    # ------------------------------------------------------------------ #
    #  SAM3 freezing + Orthogonal LoRA                                     #
    # ------------------------------------------------------------------ #

    def _freeze_sam3(self):
        for param in self.sam3.parameters():
            param.requires_grad = False

    def _enable_seg_row_training(self, ref_token: str = "<|im_end|>") -> list:
        """Unlock the <|seg|> token's input embedding and lm_head rows for
        training, with a per-row gradient mask so only those rows update.

        Why this is needed:
          The <|seg|> token is added AFTER base Qwen training via
          resize_token_embeddings, which appends randomly-initialised rows to
          BOTH `embed_tokens.weight` and `lm_head.weight`. Qwen LoRA (on
          q_proj / v_proj) never touches those rows, so `lm_head.weight[SEG]`
          stays random forever. Result: during generation, SEG's logit is a
          random projection of the hidden state and never wins argmax —
          Qwen picks `<|tool_call|>` / `<|im_end|>` instead, so live inference
          returns 0 mIoU while teacher-forced Oracle works fine.

        The fix:
          (1) Warm-start: copy the reference token's (default `<|im_end|>`)
              row into the SEG row for both embed and lm_head — gives SEG
              a sensible statistical starting point instead of random noise.
          (2) Unfreeze the full weight tensors and register a per-row
              gradient mask that zeros out all rows except SEG. The
              optimizer thus updates only the SEG row; the rest of the
              vocabulary stays exactly at its pretrained values.

        Returns the list of two weight tensors so the outer curriculum can
        toggle their requires_grad in sync with Qwen LoRA (Stage 1 off,
        Stage 2 on). Hooks are persistent — they fire whenever a gradient
        is produced on these tensors.
        """
        tok = self.processor.tokenizer
        ref_id = tok.convert_tokens_to_ids(ref_token)
        seg_id = self.seg_token_id
        if ref_id is None or ref_id == tok.unk_token_id:
            raise ValueError(f"reference token {ref_token!r} not in vocab — "
                             f"pick a real special token for warm-start")

        embed = self.qwen.get_input_embeddings()
        lm_head = self.qwen.get_output_embeddings()

        # ---- (1) Warm-start SEG row <- reference row --------------------- #
        with torch.no_grad():
            embed.weight[seg_id].copy_(embed.weight[ref_id])
            lm_head.weight[seg_id].copy_(lm_head.weight[ref_id])

        # ---- (2) Unfreeze + per-row gradient mask ------------------------ #
        def _make_mask_hook(row_id: int):
            """Returns a grad-hook that zeros every row except `row_id`."""
            def _hook(grad):
                # Build a {0,1} mask with a 1 only on row_id, same dtype/device.
                mask = torch.zeros_like(grad)
                mask[row_id] = 1.0
                return grad * mask
            return _hook

        trainable = []
        for tensor_name, w in (("embed_tokens", embed.weight),
                                ("lm_head",       lm_head.weight)):
            w.requires_grad_(True)
            w.register_hook(_make_mask_hook(seg_id))
            trainable.append(w)

        ref_name = ref_token
        print(f"  SEG row training enabled: seg_id={seg_id}, "
              f"warm-started from {ref_name!r} (id={ref_id}). "
              f"Gradient mask restricts updates to row {seg_id} only.")
        return trainable

    # ------------------------------------------------------------------ #
    #  V5 Block-Diagonal Attention Mask (Anti-Context-Leakage)             #
    # ------------------------------------------------------------------ #

    def create_independent_texture_mask(self, input_ids):
        """
        Build a block-diagonal attention mask that prevents TEXTURE_i's tokens
        (including <|seg|>_i) from attending to any TEXTURE_j tokens (j < i).

        All texture blocks CAN attend to the shared prefix (system + image +
        user + assistant header). Within each block, standard causal attention
        applies. Cross-block attention is strictly blocked.

        This eliminates Context Leakage where <|seg|>_2's hidden state absorbs
        semantic noise from TEXTURE_1's description, which was proven to inflate
        cosine similarity from ~0.16 to ~0.74 and cause Directional Drift.

        Args:
            input_ids: (B, L) token IDs (teacher-forced or generated).

        Returns:
            (B, 1, L, L) float attention mask. 0.0 = attend, -65504 = block.
            Compatible with HuggingFace's 4D mask convention (returned as-is
            by create_causal_mask when it detects a 4D input).
        """
        B, L = input_ids.shape
        device = input_ids.device
        # Use bf16-safe min value (not -inf, avoids NaN in softmax edge cases)
        BLOCK_VAL = torch.finfo(torch.bfloat16).min  # -65504

        # Start with standard causal mask: upper triangle blocked
        mask = torch.zeros(B, 1, L, L, device=device, dtype=torch.bfloat16)
        causal_block = torch.triu(
            torch.full((L, L), BLOCK_VAL, device=device, dtype=torch.bfloat16),
            diagonal=1,
        )
        mask += causal_block.unsqueeze(0).unsqueeze(0)  # broadcast to (B, 1, L, L)

        # Cache the <|im_start|> token ID for finding assistant boundary
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

        for b in range(B):
            ids = input_ids[b]

            # V7: locate assistant content start FIRST so we can filter
            # prompt-template <|seg|> occurrences out before building blocks.
            asst_content_start = self._find_asst_start(ids, im_start_id)

            all_seg = (ids == self.seg_token_id).nonzero(as_tuple=True)[0]
            seg_positions = all_seg[all_seg >= asst_content_start]
            K = seg_positions.shape[0]
            if K < 2:
                continue  # 0-1 textures → no cross-block leakage possible

            # Build texture block boundaries:
            # Block k spans from block_start to seg_positions[k] (inclusive).
            # Block 0: asst_content_start to seg_positions[0]
            # Block k (k>=1): seg_positions[k-1]+1 to seg_positions[k]
            blocks = []
            for k in range(K):
                start = asst_content_start if k == 0 else seg_positions[k - 1].item() + 1
                end = seg_positions[k].item()  # inclusive
                if start <= end:
                    blocks.append((start, end))

            # Apply the block-diagonal constraint:
            # For block k (k >= 1), BLOCK attention to all tokens in blocks 0..k-1.
            for k in range(1, len(blocks)):
                row_start, row_end = blocks[k]
                for prev in range(k):
                    col_start, col_end = blocks[prev]
                    mask[b, 0, row_start:row_end + 1, col_start:col_end + 1] = BLOCK_VAL

            # Also block padding tokens (where input_ids == pad_token_id)
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                pad_positions = (ids == pad_token_id)
                if pad_positions.any():
                    mask[b, 0, :, pad_positions] = BLOCK_VAL
                    mask[b, 0, pad_positions, :] = BLOCK_VAL

        return mask

    # ------------------------------------------------------------------ #
    #  V5 [SEG] token extraction                                           #
    # ------------------------------------------------------------------ #

    def extract_seg_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple:
        """
        Extract hidden states at <|seg|> token positions.

        Clean and deterministic — no regex, no line parsing, no truncation
        edge cases. Works identically at training and inference time.

        Args:
            hidden_states: (B, seq_len, llm_dim) from Qwen last layer.
            input_ids: (B, seq_len) token IDs.

        Returns:
            seg_embeds: (B, MAX_TEXTURES, llm_dim) hidden states at [SEG] tokens.
            k_preds: (B,) number of [SEG] tokens found per sample.
        """
        B = hidden_states.shape[0]
        seg_embeds = torch.zeros(B, MAX_TEXTURES, self.llm_dim,
                                 device=hidden_states.device,
                                 dtype=hidden_states.dtype)
        k_preds = torch.zeros(B, dtype=torch.long, device=hidden_states.device)
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

        for b in range(B):
            ids = input_ids[b]
            # V7: filter SEG positions to the assistant region. Prompt template
            # contains literal "<|seg|>" which tokenises to the SEG special
            # token — those occurrences must not be mistaken for real outputs.
            asst_start = self._find_asst_start(ids, im_start_id)
            all_positions = (ids == self.seg_token_id).nonzero(as_tuple=True)[0]
            positions = all_positions[all_positions >= asst_start]
            k = min(len(positions), MAX_TEXTURES)
            for i in range(k):
                seg_embeds[b, i] = hidden_states[b, positions[i]]
            k_preds[b] = k

        return seg_embeds, k_preds

    def _find_asst_start(self, ids: torch.Tensor, im_start_id: int) -> int:
        """Index of the first assistant-content token (last <|im_start|> + header)."""
        L = ids.shape[0]
        im_starts = (ids == im_start_id).nonzero(as_tuple=True)[0]
        if len(im_starts) == 0:
            return 0
        last = int(im_starts[-1].item())
        asst_start = last + 1
        for skip in range(last + 1, min(last + 8, L)):
            tok_str = self.processor.tokenizer.decode(
                [int(ids[skip].item())], skip_special_tokens=False,
            )
            if "\n" in tok_str:
                asst_start = skip + 1
                break
        return asst_start

    # ------------------------------------------------------------------ #
    #  Query slot assembly                                                 #
    # ------------------------------------------------------------------ #

    def build_query_slots(self, seg_embeds, k_preds):
        """
        Build the 7-slot query sequence: [DUSTBIN, SEG_1, ..., SEG_K, PAD...]
        """
        B = seg_embeds.shape[0]
        device = seg_embeds.device

        dustbin = self.dustbin_embed.expand(B, 1, self.llm_dim)
        query_embeds = torch.cat([dustbin, seg_embeds], dim=1)  # (B, 7, llm_dim)

        pad_mask = torch.zeros(B, NUM_QUERY_SLOTS, dtype=torch.bool, device=device)
        for b in range(B):
            kp = int(k_preds[b].item())
            pad_mask[b, kp + 1:] = True

        return query_embeds, pad_mask

    # ------------------------------------------------------------------ #
    #  SAM3 pipeline (Batch Multiplexing — unchanged from V3/V4)           #
    # ------------------------------------------------------------------ #

    def _get_img_feats(self, backbone_out, img_ids):
        n_levels = self.sam3.num_feature_levels
        vis_feats = backbone_out["backbone_fpn"][-n_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-n_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]

        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return img_feats, img_pos_embeds, vis_feat_sizes

    def run_sam3_semantic(self, backbone_out, query_256, pad_mask):
        """
        Batch Multiplexed SAM3 pass. Each query gets its own "slot 1"
        position by flattening into the batch dimension.
        """
        B, N, D = query_256.shape
        device = query_256.device

        image_ids = torch.arange(B, device=device).repeat_interleave(N)
        img_feats, img_pos_embeds, vis_feat_sizes = self._get_img_feats(
            backbone_out, image_ids,
        )

        query_flat = query_256.reshape(B * N, 1, D)
        prompt = query_flat.transpose(0, 1)
        prompt_pos = torch.zeros_like(prompt)

        memory_dict = self.sam3.transformer.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=None,
            feat_sizes=vis_feat_sizes,
        )
        enc_hs = memory_dict["memory"]

        seg_head = self.sam3.segmentation_head
        if seg_head.cross_attend_prompt is not None:
            tgt2 = seg_head.cross_attn_norm(enc_hs)
            tgt2 = seg_head.cross_attend_prompt(
                query=tgt2, key=prompt, value=prompt,
                key_padding_mask=None,
            )[0]
            enc_hs = tgt2 + enc_hs

        pixel_embed = seg_head._embed_pixels(
            backbone_feats=backbone_out["backbone_fpn"],
            image_ids=image_ids,
            encoder_hidden_states=enc_hs,
        )

        logits_flat = self.mask_head(query_flat, pixel_embed)
        mask_logits = logits_flat.reshape(B, N, *logits_flat.shape[-2:])
        return mask_logits

    # ------------------------------------------------------------------ #
    #  Full forward pass (V5 — live Qwen with [SEG] extraction)            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        qwen_inputs: dict,
        sam_images: torch.Tensor,
        seg_grad_to_lm: bool = True,
    ) -> dict:
        """
        V5 forward pass: Qwen generates with teacher forcing, hidden states
        extracted at <|seg|> positions, projected through bottleneck, fed to SAM.

        Gradients from mask loss flow: SAM → projector → [SEG] hidden states
        → Qwen LoRA (when seg_grad_to_lm=True).

        Args:
            qwen_inputs: dict from Qwen processor (input_ids, attention_mask,
                         labels with -100 on text and real IDs on [SEG]).
            sam_images: (B, 3, 1008, 1008) SAM3-preprocessed images.
            seg_grad_to_lm: whether seg gradients flow to Qwen LoRA.

        Returns:
            dict with mask_logits, seg_embeds, k_preds, pad_mask, lm_loss, query_256.
        """
        # Build block-diagonal attention mask to prevent Context Leakage.
        # Each TEXTURE block's tokens (including <|seg|>) can only attend to
        # the shared prefix and their own block — NOT to earlier texture blocks.
        input_ids = qwen_inputs["input_ids"]
        attention_mask_2d = qwen_inputs.get("attention_mask")

        # Step 1: Pre-compute position_ids using the standard 2D attention mask.
        # Qwen3-VL's get_rope_index needs a 2D mask for rope position encoding.
        # We compute this BEFORE replacing the mask with our 4D block-diagonal.
        qwen_vl_model = self.qwen.base_model.model.model  # PeftModel → LoraModel → Qwen3VLForCausalLM → Qwen3VLModel
        with torch.no_grad():
            position_ids, rope_deltas = qwen_vl_model.get_rope_index(
                input_ids,
                qwen_inputs.get("image_grid_thw"),
                qwen_inputs.get("video_grid_thw"),
                attention_mask_2d,
            )

        # Step 2: Build the 4D block-diagonal mask
        custom_mask = self.create_independent_texture_mask(input_ids)

        # Step 3: Qwen forward with pre-computed position_ids + custom 4D mask.
        # By passing position_ids explicitly, the model skips get_rope_index
        # (which would crash on the 4D mask).
        qwen_fwd_inputs = {k: v for k, v in qwen_inputs.items()
                           if k != "attention_mask"}
        qwen_fwd_inputs["attention_mask"] = custom_mask
        qwen_fwd_inputs["position_ids"] = position_ids
        qwen_out = self.qwen(**qwen_fwd_inputs, output_hidden_states=True)
        hidden_states = qwen_out.hidden_states[-1]  # (B, seq, llm_dim)
        lm_loss = qwen_out.loss if qwen_out.loss is not None else \
            torch.tensor(0.0, device=self.device)

        # Extract hidden states at <|seg|> positions
        seg_embeds, k_preds = self.extract_seg_hidden_states(
            hidden_states, input_ids,
        )

        if not seg_grad_to_lm:
            seg_embeds = seg_embeds.detach()

        # Build query slots [DUSTBIN, SEG_1..MAX]
        query_embeds, pad_mask = self.build_query_slots(seg_embeds, k_preds)

        # V7: Bridge (4096 → 1024) then SAM's frozen resizer (1024 → 256)
        query_256 = self._sam_resizer(self.bridge(query_embeds))

        # SAM3 backbone (frozen)
        with torch.no_grad():
            backbone_out = self.sam3.backbone.forward_image(sam_images)
            backbone_out["img_batch_all_stages"] = sam_images

        # SAM3 semantic path → (B, 7, H, W)
        mask_logits = self.run_sam3_semantic(backbone_out, query_256, pad_mask)

        return {
            "mask_logits": mask_logits,
            "seg_embeds": seg_embeds,
            "query_256": query_256,
            "k_preds": k_preds,
            "pad_mask": pad_mask,
            "lm_loss": lm_loss,
            "qwen_logits": qwen_out.logits,  # V6: raw logits for weighted LM loss
        }

    # ------------------------------------------------------------------ #
    #  Inference forward (autoregressive generation with [SEG])             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def inference_forward(
        self,
        qwen_inputs: dict,
        sam_images: torch.Tensor,
        max_new_tokens: int = 300,
    ) -> dict:
        """
        V5 Inference with adaptive extraction strategy.

        Pass 1 (Generation): Run qwen.generate() with STANDARD causal mask
          and output_hidden_states=True.

        Extraction (adaptive):
          - If <|seg|> tokens found in generated text → Pass 2 with block-
            diagonal mask for clean, decoupled extraction (mature LoRA).
          - If NO <|seg|> tokens → regex fallback on Pass 1 hidden states
            (early training, LoRA hasn't learned to emit <|seg|> yet).

        This makes inference robust during the entire training lifecycle.
        """
        import re

        # ---- Pass 1: Generation + hidden states -------------------------- #
        gen_out = self.qwen.generate(
            **qwen_inputs,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
        generated_ids = gen_out.sequences       # (B, prompt_len + gen_len)
        prompt_len = qwen_inputs["input_ids"].shape[1]
        B = generated_ids.shape[0]

        # Collect Pass 1 hidden states (needed for regex fallback)
        gen_hidden_list = []
        for step_hidden in gen_out.hidden_states:
            last_layer = step_hidden[-1]
            if last_layer.shape[1] > 1:
                last_layer = last_layer[:, -1:, :]
            gen_hidden_list.append(last_layer)
        gen_hidden = torch.cat(gen_hidden_list, dim=1)

        full_hidden = torch.zeros(
            B, generated_ids.shape[1], self.llm_dim,
            device=generated_ids.device, dtype=gen_hidden.dtype,
        )
        full_hidden[:, prompt_len:prompt_len + gen_hidden.shape[1]] = gen_hidden

        # Decode generated text for logging
        generated_text = []
        for b in range(B):
            gen_tokens = generated_ids[b, prompt_len:]
            text = self.processor.tokenizer.decode(
                gen_tokens, skip_special_tokens=False,
            )
            generated_text.append(text)

        # ---- Check if <|seg|> tokens were generated ---------------------- #
        has_seg = (generated_ids == self.seg_token_id).any(dim=1)  # (B,)

        if has_seg.all():
            # ---- Path A: All samples have <|seg|> → Pass 2 (decoupled) --- #
            custom_mask = self.create_independent_texture_mask(generated_ids)
            attention_mask_2d = torch.ones(
                B, generated_ids.shape[1],
                dtype=torch.long, device=generated_ids.device,
            )
            qwen_vl_model = self.qwen.base_model.model.model
            position_ids, _ = qwen_vl_model.get_rope_index(
                generated_ids,
                qwen_inputs.get("image_grid_thw"),
                qwen_inputs.get("video_grid_thw"),
                attention_mask_2d,
            )
            pass2_inputs = {
                "input_ids": generated_ids,
                "attention_mask": custom_mask,
                "position_ids": position_ids,
            }
            for key in ("pixel_values", "image_grid_thw",
                         "pixel_values_videos", "video_grid_thw"):
                if key in qwen_inputs:
                    pass2_inputs[key] = qwen_inputs[key]

            pass2_out = self.qwen(**pass2_inputs, output_hidden_states=True)
            hidden_states = pass2_out.hidden_states[-1]
            seg_embeds, k_preds = self.extract_seg_hidden_states(
                hidden_states, generated_ids,
            )
        else:
            # ---- Path B: No <|seg|> → regex fallback on Pass 1 hidden ---- #
            # (Early training: LoRA hasn't learned to emit <|seg|> yet)
            seg_embeds = torch.zeros(
                B, MAX_TEXTURES, self.llm_dim,
                device=full_hidden.device, dtype=full_hidden.dtype,
            )
            k_preds = torch.zeros(B, dtype=torch.long, device=full_hidden.device)

            for b in range(B):
                # V7: filter to generated-region seg tokens only; prompt
                # template contains literal <|seg|> which tokenises to the
                # special SEG token and would otherwise be counted here.
                all_seg = (generated_ids[b] == self.seg_token_id).nonzero(
                    as_tuple=True)[0]
                seg_pos = all_seg[all_seg >= prompt_len]
                if len(seg_pos) > 0:
                    k = min(len(seg_pos), MAX_TEXTURES)
                    for i in range(k):
                        seg_embeds[b, i] = full_hidden[b, seg_pos[i]]
                    k_preds[b] = k
                    continue

                # Regex fallback: parse TEXTURE_N: lines from generated text
                text = generated_text[b]
                if "TEXTURE" not in text.upper():
                    continue

                was_truncated = "<|im_end|>" not in text
                lines = text.strip().split("\n")
                tex_count = 0

                for line_idx, line in enumerate(lines):
                    match = re.match(
                        r"(?:<\s*)?TEXTURE[_\s]*(\d+)\s*:", line.strip(),
                        re.IGNORECASE,
                    )
                    if not match or tex_count >= MAX_TEXTURES:
                        continue
                    if was_truncated and line_idx == len(lines) - 1:
                        continue
                    desc = line.strip().split(":", 1)[-1].strip()
                    if len(desc.split()) < 3:
                        continue

                    text_up_to = "\n".join(lines[: line_idx + 1])
                    tokens_up_to = self.processor.tokenizer.encode(
                        text_up_to, add_special_tokens=False,
                    )
                    pos = prompt_len + min(
                        len(tokens_up_to) - 1,
                        gen_hidden.shape[1] - 1,
                    )
                    if 0 <= pos < full_hidden.shape[1]:
                        seg_embeds[b, tex_count] = full_hidden[b, pos]
                        tex_count += 1

                k_preds[b] = tex_count

        # ---- SAM3 pipeline ----------------------------------------------- #
        query_embeds, pad_mask = self.build_query_slots(seg_embeds, k_preds)
        # V7 Bridge (4096 → 1024) then SAM's frozen resizer (1024 → 256)
        query_256 = self._sam_resizer(self.bridge(query_embeds))

        backbone_out = self.sam3.backbone.forward_image(sam_images)
        backbone_out["img_batch_all_stages"] = sam_images
        mask_logits = self.run_sam3_semantic(backbone_out, query_256, pad_mask)

        return {
            "mask_logits": mask_logits,
            "seg_embeds": seg_embeds,
            "k_preds": k_preds,
            "pad_mask": pad_mask,
            "lm_loss": torch.tensor(0.0, device=self.device),
            "generated_text": generated_text,
        }

    # ------------------------------------------------------------------ #
    #  Parameter groups                                                    #
    # ------------------------------------------------------------------ #

    def get_parameter_groups(self, base_lr: float) -> list:
        """V7 parameter groups.

        - Qwen LoRA:  conservative LR (base * qwen_lr_scale = 1e-6).
                       Frozen in Stage 1; unfrozen at Stage 2.
        - Bridge:     full base LR (1e-4). Decayed 10× at Stage 2 entry.
        - Mask head + dustbin: full base LR.
        - SEG rows (embed + lm_head): base LR. Frozen in Stage 1; unfrozen
          at Stage 2. Full tensors are in the group but a gradient mask
          restricts effective updates to the SEG row only (see
          `_enable_seg_row_training`).
        - SAM3:       100% frozen permanently — no LoRA group.
        """
        qwen_lr_scale = self.cfg["model"].get("qwen_lr_scale", 0.01)
        seg_row_lr_scale = self.cfg["model"].get("seg_row_lr_scale", 1.0)

        qwen_lora_params = [
            p for n, p in self.qwen.named_parameters()
            if "lora" in n.lower()
        ]

        bridge_params = list(self.bridge.parameters())
        mask_head_params = list(self.mask_head.parameters())
        dustbin_params = [self.dustbin_embed]
        seg_row_params = list(getattr(self, "_seg_row_params", []))

        groups = [
            {"params": qwen_lora_params,
             "lr": base_lr * qwen_lr_scale, "name": "qwen_lora"},
            {"params": bridge_params, "lr": base_lr, "name": "bridge"},
            {"params": mask_head_params, "lr": base_lr, "name": "mask_head"},
            {"params": dustbin_params, "lr": base_lr, "name": "dustbin"},
        ]
        if seg_row_params:
            groups.append({
                "params": seg_row_params,
                "lr": base_lr * seg_row_lr_scale,
                "name": "seg_rows",
            })
        return groups

    def num_trainable_params(self) -> dict:
        qwen_lora_n = sum(
            p.numel() for n, p in self.qwen.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        )
        bridge_n = sum(p.numel() for p in self.bridge.parameters())
        mask_head_n = sum(p.numel() for p in self.mask_head.parameters())
        dustbin_n = self.dustbin_embed.numel()

        # SEG-row effective count (one row of embed + one row of lm_head).
        # The tensors themselves are full vocab×hidden, but the gradient
        # mask zeros all rows except SEG — so EFFECTIVELY only 2 × hidden
        # parameters learn. We report the effective count for clarity.
        if getattr(self, "_seg_row_params", None):
            hidden = self._seg_row_params[0].shape[-1]
            seg_rows_effective = 2 * hidden
            seg_rows_raw = sum(p.numel() for p in self._seg_row_params)
        else:
            seg_rows_effective = 0
            seg_rows_raw = 0

        total = (qwen_lora_n + bridge_n + mask_head_n + dustbin_n
                 + seg_rows_effective)
        return {
            "qwen_lora": qwen_lora_n,
            "bridge": bridge_n,
            "mask_head": mask_head_n,
            "dustbin": dustbin_n,
            "seg_rows_effective": seg_rows_effective,
            "seg_rows_raw_tensor_size": seg_rows_raw,
            "total_trainable": total,
        }
