"""Exercise the native ComfyUI MOSS block and the FreeFuse node dispatch."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

COMFY = Path(__file__).resolve().parents[2] / "ComfyUI"
if not COMFY.is_dir():
    pytest.skip("Place ComfyUI beside the FreeFuse repository for native tests.", allow_module_level=True)
sys.path.insert(0, str(COMFY))

import comfy.model_management
import comfy.ops
from comfy.ldm.moss_image.model import MOSSImage
from freefuse_comfyui.freefuse_core.attention_replace import FreeFuseState
from freefuse_comfyui.freefuse_core.attention_bias import AttentionBiasConfig
from freefuse_comfyui.freefuse_core.bypass_lora_loader import MultiAdapterBypassForwardHook
from freefuse_comfyui.freefuse_core.moss_image_support import apply_moss_collect_patches, apply_moss_bias_patches
from freefuse_comfyui.freefuse_core.token_utils import detect_model_type


class Patcher:
    def __init__(self, model):
        self.model = SimpleNamespace(diffusion_model=model)
        self.patches = {}

    def set_model_patch_replace(self, patch, group, kind, index):
        self.patches[(kind, index)] = patch


def native():
    torch.manual_seed(10)
    model = MOSSImage(dim=32, ffn_dim=48, num_heads=2, num_layers=3, text_dim=24,
                      latent_channels=8, dtype=torch.float32, device="cpu",
                      operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.normal_(std=0.05)
    return model


@pytest.mark.parametrize("fused", [False, True])
def test_native_collect_and_zero_bias_parity(fused):
    # Comfy's differentiable path is also usable on a CPU-only test host.
    old_training = comfy.model_management.in_training
    comfy.model_management.in_training = True
    try:
        model = native()
        if fused:
            for block in model.layers:
                qkv = torch.nn.Linear(32, 96, bias=False)
                with torch.no_grad():
                    qkv.weight.copy_(torch.cat([block.proj[name].weight for name in ("q", "k", "v")]))
                for name in ("q", "k", "v"):
                    del block.proj[name]
                block.proj["qkv"] = qkv
                block.fused_qkv = True
        patcher = Patcher(model)
        assert detect_model_type(model=patcher) == "moss_image"
        x, text = torch.randn(1, 8, 2, 3), torch.randn(1, 7, 24)
        with torch.no_grad():
            expected = model(x.clone(), torch.tensor([500.]), text)
            state = FreeFuseState()
            state.collect_step = 0
            state.collect_block = state.collect_block_end = 1
            state.token_pos_maps = {"a": [[1]], "b": [[4]]}
            state.collected_outputs["img_seq_len"] = 6
            apply_moss_collect_patches(patcher, state)
            options = {"patches_replace": {"dit": patcher.patches}, "cond_or_uncond": [0]}
            actual = model(x.clone(), torch.tensor([500.]), text, options)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert set(state.similarity_maps) == {"a", "b"}
            assert state.similarity_maps["a"].shape == (1, 6, 1)
            # Negative CFG text has a different length and no subject positions.
            options["cond_or_uncond"] = [1]
            short_text = text[:, :2]
            actual_negative = model(x.clone(), torch.tensor([500.]), short_text, options)
            expected_negative = model(x.clone(), torch.tensor([500.]), short_text)
            torch.testing.assert_close(actual_negative, expected_negative, rtol=0, atol=0)

            patcher.patches = {}
            masks = {"a": torch.tensor([[1, 1, 1, 0, 0, 0.]]),
                     "b": torch.tensor([[0, 0, 0, 1, 1, 1.]])}
            config = AttentionBiasConfig(enabled=True, bias_scale=0, positive_bias_scale=0)
            apply_moss_bias_patches(patcher, masks, state.token_pos_maps, config)
            options = {"patches_replace": {"dit": patcher.patches}, "cond_or_uncond": [0]}
            actual = model(x.clone(), torch.tensor([500.]), text, options)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            # An unconditional call receives no concept bias at all.
            options["cond_or_uncond"] = [1]
            actual = model(x.clone(), torch.tensor([500.]), text, options)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        comfy.model_management.in_training = old_training


def test_native_moss_spatial_mask_image_prefix_and_text_exclusion():
    layer = torch.nn.Linear(32, 32)
    hook = MultiAdapterBypassForwardHook(layer, "diffusion_model.layers.0.proj.qkv")
    hook.mask_enabled = True
    hook.masks = {"a": torch.tensor([[1., 0., 1.], [0., 1., 0.]])}
    mask = hook._get_mask_for_sequence("a", 9, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(mask, torch.tensor([1., 0., 1., 0., 1., 0., 1., 1., 1.]))
    text = MultiAdapterBypassForwardHook(layer, "diffusion_model.text_embedder.blocks.0.qkv")
    assert not text._should_apply_spatial_mask()


def test_native_moss_token_positions_match_release_template_for_unicode():
    from comfy.text_encoders.moss_image import MOSSImageTokenizer
    from freefuse_comfyui.freefuse_core.token_utils import (
        find_concept_positions, _flatten_chunked_token_ids, get_tokenizer_for_model)
    from src.pipeline.freefuse_moss_image_pipeline import find_moss_concept_positions
    tokenizer = MOSSImageTokenizer()
    clip = SimpleNamespace(tokenizer=tokenizer, tokenize=tokenizer.tokenize_with_weights)
    prompt = "金发精灵与黑发女性，站在花园里。"
    concepts = {"fairy": "金发精灵", "demon": "黑发女性"}
    actual = find_concept_positions(clip, prompt, concepts, model_type="moss_image")
    hf = get_tokenizer_for_model(clip, "moss_image")
    expected = find_moss_concept_positions(hf, prompt, concepts)
    assert actual == expected
    ids = _flatten_chunked_token_ids(clip.tokenize(prompt))
    assert "<think>\n" in hf.decode(ids)

def test_native_bias_patches_stay_scoped_to_clone_options():
    model = native()
    patcher = Patcher(model)
    original_forward = model.layers[0].forward
    masks = {"a": torch.ones(1, 6)}
    apply_moss_bias_patches(patcher, masks, {"a": [[1]]}, AttentionBiasConfig(), "last_half")
    assert ("single_block", 0) not in patcher.patches
    assert ("single_block", 1) in patcher.patches
    assert model.layers[0].forward == original_forward
    assert not model.layers[1]._forward_hooks and not model.layers[1]._forward_pre_hooks
