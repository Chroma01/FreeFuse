"""Small real MOSS blocks exercise routing without downloading model weights."""
import copy

import pytest
import torch

from src.models.freefuse_transformer_moss_image import FreeFuseMossImageTransformer
from src.pipeline.freefuse_moss_image_pipeline import FreeFuseMossImagePipeline
from src.tuner.moss_image_lora import MossFreeFuseLinear, load_moss_lora
from src.vendor.moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


def tiny_model():
    torch.manual_seed(13)
    return MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=2, num_layers=3,
                                text_dim=24, freq_dim=16, latent_channels=8)


def inputs():
    torch.manual_seed(42)
    return torch.randn(1, 8, 2, 3), torch.randn(1, 7, 24), torch.ones(1, 7, dtype=torch.bool)


def test_disabled_transformer_parity_and_padding():
    original = tiny_model()
    model = FreeFuseMossImageTransformer.init_from_transformer(copy.deepcopy(original))
    x, text, mask = inputs()
    mask[:, -2:] = False
    with torch.no_grad():
        expected = original(x, 700, text, mask).sample
        actual = model(x, 700, text, mask).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_full_cfg_slg_sampler_parity_with_routing_but_no_bias():
    original = tiny_model()
    base = MOSSImagePipeline(original, MOSSImageScheduler())
    model = FreeFuseMossImageTransformer.init_from_transformer(copy.deepcopy(original))
    pipe = FreeFuseMossImagePipeline(model, MOSSImageScheduler())
    x, text, mask = inputs()
    options = dict(prompt_embeds=text, prompt_attention_mask=mask,
                   negative_prompt_embeds=text[:, :3], negative_prompt_attention_mask=mask[:, :3],
                   height=32, width=48, num_inference_steps=4, guidance_scale=3,
                   uncond_skip_layers=[1], slg_layers=[2], slg_scale=2,
                   slg_start=0.25, slg_stop=0.75, latents=x, output_type="latent")
    expected = base(**options).images
    actual = pipe(**options, freefuse_token_pos_maps={"a": [[1]], "b": [[4]]},
                  collect_block=1, collect_step=1, attention_bias_scale=0).images
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(pipe.last_freefuse_debug["masks"]) == {"a", "b"}
    masks = pipe.last_freefuse_debug["masks"]
    torch.testing.assert_close(masks["a"] + masks["b"], torch.ones(1, 6, 1))
    assert not model._freefuse_state.routing


def test_spatial_and_text_routing_and_negative_length():
    model = FreeFuseMossImageTransformer.init_from_transformer(tiny_model())
    down, up = torch.ones(2, 32), torch.ones(32, 2)
    for name in ("a", "b"):
        load_moss_lora(model, {"layers.0.proj.q.lora_A.weight": down,
                               "layers.0.proj.q.lora_B.weight": up}, name)
    layer = model.layers[0].proj["q"]
    state = model._freefuse_state
    state.routing = True
    state.image_length, state.text_length = 3, 4
    state.token_positions = {"a": [[0]], "b": [[2]]}
    state.masks = {"a": torch.tensor([[[1.], [0.], [0.]]]),
                   "b": torch.tensor([[[0.], [1.], [0.]]])}
    x = torch.ones(1, 7, 32)
    delta = layer(x) - layer.base_layer(x)
    # Both LoRAs are zero on background; one remains active on each subject token.
    assert torch.all(delta[:, 2] == 0)
    torch.testing.assert_close(delta[:, 3], delta[:, 0])
    torch.testing.assert_close(delta[:, 5], delta[:, 1])
    state.conditional = False
    negative = torch.ones(1, 4, 32)  # one negative text token, not four
    delta = layer(negative) - layer.base_layer(negative)
    assert torch.all(delta[:, 2] == 0)
    torch.testing.assert_close(delta[:, 3], 2 * delta[:, 0])


def test_lora_fused_qkv_alpha_and_transactional_rejection():
    model = FreeFuseMossImageTransformer.init_from_transformer(tiny_model())
    state = {"lora_unet_layers_0_proj_qkv.lora_down.weight": torch.ones(2, 32),
             "lora_unet_layers_0_proj_qkv.lora_up.weight": torch.ones(96, 2),
             "lora_unet_layers_0_proj_qkv.alpha": torch.tensor(1.)}
    assert load_moss_lora(model, state, "char", weight=0.8) == 3
    assert model.layers[0].proj["q"].scales["char"] == pytest.approx(0.4)
    invalid = {**state, "unknown.weight": torch.ones(1)}
    with pytest.raises(ValueError, match="Unsupported"):
        load_moss_lora(model, invalid, "bad")
    assert all("bad" not in m.adapters for m in model.modules() if isinstance(m, MossFreeFuseLinear))


def test_exception_cleanup_and_repeated_baseline():
    model = FreeFuseMossImageTransformer.init_from_transformer(tiny_model())
    pipe = FreeFuseMossImagePipeline(model, MOSSImageScheduler())
    x, text, mask = inputs()
    options = dict(prompt_embeds=text, height=32, width=48, num_inference_steps=3,
                   guidance_scale=1, uncond_skip_layers=[], output_type="latent", latents=x)
    with pytest.raises(ValueError, match="invalid positions"):
        pipe(**options, freefuse_token_pos_maps={"a": [[99]]}, collect_block=1, collect_step=1)
    state = model._freefuse_state
    assert not state.routing and not state.collect and state.loras_enabled
    first = pipe(**options, use_freefuse=False).images
    second = pipe(**options, use_freefuse=False).images
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_attention_bias_quadrants_and_negative_branch():
    model = FreeFuseMossImageTransformer.init_from_transformer(tiny_model())
    state = model._freefuse_state
    state.routing = True
    state.image_length, state.text_length = 2, 3
    state.token_positions = {"a": [[1]]}
    state.masks = {"a": torch.tensor([[[1.], [0.]]])}
    x = torch.zeros(1, 5, 32)
    bias = state.attention_bias(0, x)[0, 0]
    assert bias[0, 3] == 2 and bias[1, 3] == -4
    assert bias[3, 0] == 2 and bias[3, 1] == -4
    assert torch.all(bias[:2, :2] == 0)
    state.conditional = False
    assert state.attention_bias(0, x) is None

def test_exporter_normalizes_qwen35_keys_without_changing_tensors():
    from scripts.export_moss_to_comfyui import normalize_text_keys
    tensor = torch.randn(3)
    state = {"language_model.layers.0.linear_attn.A_log": tensor,
             "visual.patch_embed.proj.weight": tensor}
    normalized = normalize_text_keys(state)
    assert normalized["model.language_model.layers.0.linear_attn.A_log"] is tensor
    assert normalized["model.visual.patch_embed.proj.weight"] is tensor
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_text_keys({**state, "model.visual.patch_embed.proj.weight": tensor})

def test_transformer_constructor_and_pretrained_round_trip(tmp_path):
    model = FreeFuseMossImageTransformer(dim=32, ffn_dim=48, num_heads=2, num_layers=3,
                                         text_dim=24, freq_dim=16, latent_channels=8)
    model.save_pretrained(tmp_path)
    restored = FreeFuseMossImageTransformer.from_pretrained(tmp_path)
    x, text, mask = inputs()
    with torch.no_grad():
        expected = model(x, 700, text, mask).sample
        actual = restored(x, 700, text, mask).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
