"""V7 Bridge Projector: Qwen [SEG] (4096) → SAM3 text-encoder native space (1024).

Output passes through SAM3's *frozen* `resizer` (Linear 1024 → 256) to reach
the SAM query space. We train only the Bridge; SAM's pretrained 1024→256
projection contributes its learned semantic understanding for free.

Architecture:
    Linear(4096, 1024) + LayerNorm(1024) + GELU + Dropout(0.4)

Parameter count: ~4.20M trainable (vs V6 bottleneck's 2.23M).
The 4x-wider output feeds into SAM's native text space instead of V6's
256-dim bottleneck — more geometric capacity, mitigated by heavy dropout.
"""

import torch.nn as nn


class BridgeProjector(nn.Module):
    """Qwen [SEG] hidden (4096) → SAM3 text-encoder native width (1024)."""

    def __init__(
        self,
        llm_dim: int = 4096,
        sam_text_width: int = 1024,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, sam_text_width),
            nn.LayerNorm(sam_text_width),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        """(B, N, llm_dim) → (B, N, sam_text_width)."""
        return self.proj(x)
