"""Compatibility imports for the original H3 feasibility scripts.

The runtime implementation now ships in the standalone ComfyUI package.
"""
from freefuse_comfyui.freefuse_core.minimax_h3_support import (
    Collection, H3Attention, H3Block, augmented_attention, concept_positions,
    make_mask_volumes, patch_model, similarity_logits, video_segment,
)
