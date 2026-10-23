"""Native Anima/Cosmos checks: T5 alignment, CFG isolation and spatial routing."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

COMFY = Path(__file__).resolve().parents[2] / "ComfyUI"
if not COMFY.is_dir():
    pytest.skip("ComfyUI sibling checkout is required", allow_module_level=True)
sys.path.insert(0, str(COMFY))
import comfy.ops
import comfy.model_management
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from comfy.text_encoders.anima import AnimaTokenizer
from freefuse_comfyui.freefuse_core.anima_support import apply_anima_collect_patches, apply_anima_bias_patches
from freefuse_comfyui.freefuse_core.attention_replace import FreeFuseState
from freefuse_comfyui.freefuse_core.attention_bias import AttentionBiasConfig
from freefuse_comfyui.freefuse_core.bypass_lora_loader import MultiAdapterBypassForwardHook
from freefuse_comfyui.freefuse_core.token_utils import find_concept_positions, detect_model_type


class Patcher:
    def __init__(self, model):
        self.model = SimpleNamespace(diffusion_model=model)
        self.model_options = {"transformer_options": {}}
    def clone(self):
        out = Patcher(self.model.diffusion_model)
        out.model_options = copy.deepcopy(self.model_options)
        return out
    def set_model_patch(self, patch, name):
        patches = self.model_options["transformer_options"].setdefault("patches", {})
        patches.setdefault(name, []).append(patch)
    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")
    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")


def native():
    model = MiniTrainDIT(max_img_h=8, max_img_w=8, max_frames=1, in_channels=2, out_channels=2,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=False, model_channels=128,
        num_blocks=3, num_heads=2, mlp_ratio=2, crossattn_emb_channels=24,
        pos_emb_cls="rope3d", rope_enable_fps_modulation=False, dtype=torch.float32,
        operations=comfy.ops.disable_weight_init)
    model.llm_adapter = torch.nn.Identity()
    torch.manual_seed(4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    return model


def forward(model, x, text, patcher, branches):
    options = dict(patcher.model_options["transformer_options"], cond_or_uncond=branches)
    return model(x.clone(), torch.ones(x.shape[0])*0.5, text, transformer_options=options)


def test_native_collect_parity_and_negative_text():
    model = native()
    patcher = Patcher(model)
    assert detect_model_type(model=patcher) == "anima"
    x, text = torch.randn(1, 2, 1, 4, 6), torch.randn(1, 7, 24)
    expected = forward(model, x, text, patcher, [0])
    state = FreeFuseState()
    state.collect_step = state.current_step = 0
    state.collect_block = state.collect_block_end = 1
    state.token_pos_maps = {"a": [[1]], "b": [[4]]}
    apply_anima_collect_patches(patcher, state)
    actual = forward(model, x, text, patcher, [0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert state.collected_blocks == {1}
    assert state.collected_outputs["anima_grid"] == (2, 3)
    assert state.similarity_maps["a"].shape == (1, 6, 1)
    negative = forward(model, x, text[:, :2], Patcher(model), [1])
    torch.testing.assert_close(forward(model, x, text[:, :2], patcher, [1]), negative, rtol=0, atol=0)


def test_bias_cfg_isolation_zero_parity_and_clone_scope():
    model = native()
    base = Patcher(model); patched = base.clone()
    x, text = torch.randn(2, 2, 1, 4, 6), torch.randn(2, 7, 24)
    masks = {"a": torch.tensor([[1.,1.,0.],[1.,0.,0.]]), "b": torch.tensor([[0.,0.,1.],[0.,1.,1.]])}
    positions = {"a": [[1]], "b": [[4]]}
    expected = forward(model, x, text, base, [0,1])
    zero = AttentionBiasConfig(enabled=True, bias_scale=0, positive_bias_scale=0)
    apply_anima_bias_patches(patched, masks, positions, zero)
    torch.testing.assert_close(forward(model, x, text, patched, [0,1]), expected, rtol=1e-5, atol=1e-6)
    config = AttentionBiasConfig(enabled=True, bias_scale=5, positive_bias_scale=2)
    apply_anima_bias_patches(patched, masks, positions, config)
    actual = forward(model, x, text, patched, [0,1])
    assert not torch.allclose(actual[:1], expected[:1], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)
    assert base.model_options["transformer_options"] == {}
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    # Native attention markers remain one-per-kind across repeated application.
    assert all(len(v)==1 for v in patched.model_options["transformer_options"]["patches"].values())


def test_t5_positions_weighted_unicode_and_prompt_batch():
    tokenizer = AnimaTokenizer()
    clip = SimpleNamespace(tokenizer=tokenizer, tokenize=tokenizer.tokenize_with_weights)
    assert detect_model_type(clip=clip) == "anima"
    prompts = ["(red hair:1.3), blue eyes, 黑发女性", "blue eyes, (red hair:0.8), 黑发女性"]
    concepts = {"red": "red hair", "eyes": "blue eyes"}
    positions = find_concept_positions(clip, prompts, concepts, model_type="anima", filter_meaningless=False)
    for row, prompt in enumerate(prompts):
        ids = [t[0] for t in clip.tokenize(prompt)["t5xxl"][0]]
        for name, concept in concepts.items():
            selected = [ids[i] for i in positions[name][row]]
            assert tokenizer.t5xxl.tokenizer.decode(selected).strip() == concept
    assert positions["red"][0] != positions["red"][1]
    missing = find_concept_positions(clip, prompts[0], {"missing": "green shoes"}, model_type="anima")
    assert missing["missing"] == [[]]


@pytest.mark.parametrize("suffix,expected", [
    ("blocks.0.self_attn.q_proj", True), ("blocks.0.cross_attn.q_proj", True),
    ("blocks.0.cross_attn.output_proj", True), ("blocks.0.mlp.layer1", True),
    ("blocks.0.cross_attn.k_proj", False), ("blocks.0.cross_attn.v_proj", False),
    ("blocks.0.adaln_modulation_mlp.1", False), ("llm_adapter.blocks.0.cross_attn.q_proj", False),
])
def test_anima_mask_domains(suffix, expected):
    hook = MultiAdapterBypassForwardHook(torch.nn.Linear(3,3), "diffusion_model."+suffix)
    assert hook._should_apply_spatial_mask() is expected


def test_anima_mlp_five_dimensional_spatial_routing():
    layer = torch.nn.Linear(3,3, bias=False)
    layer.weight.data.zero_()
    hook = MultiAdapterBypassForwardHook(layer, "diffusion_model.blocks.0.mlp.layer1")
    hook.original_forward = layer.forward
    hook.mask_enabled = True
    hook.masks = {"a": torch.tensor([[1.,0.,1.],[0.,1.,0.]])}
    adapter = SimpleNamespace(h=lambda x, base: torch.ones_like(base), g=lambda x: x)
    hook.adapters = [{"adapter":adapter,"offset":None,"adapter_name":"a"}]
    actual = hook._bypass_forward(torch.zeros(1,1,2,3,3))
    torch.testing.assert_close(actual, hook.masks["a"][None,None,:,:,None].expand_as(actual))


def test_rejects_bad_block_and_positive_positions():
    model=native();patcher=Patcher(model);state=FreeFuseState()
    state.collect_block=99
    with pytest.raises(ValueError, match="collection blocks"):
        apply_anima_collect_patches(patcher,state)
    state.collect_block=state.collect_block_end=0;state.collect_step=0
    state.token_pos_maps={"bad":[[100]]}
    apply_anima_collect_patches(patcher,state)
    with pytest.raises(ValueError, match="T5 sequence"):
        forward(model,torch.randn(1,2,1,4,6),torch.randn(1,7,24),patcher,[0])



def test_anima_similarity_uses_float32_and_keeps_spatial_signal():
    from freefuse_comfyui.freefuse_core.anima_support import compute_anima_similarity_maps
    # Distinct normalized queries/keys and spatial features in native head layout.
    q = torch.tensor([[[[2., 0.], [2., 0.], [0., 2.], [0., 2.]]]], dtype=torch.bfloat16)
    k = torch.tensor([[[[2., 0.], [0., 2.]]]], dtype=torch.bfloat16)
    hidden = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]], dtype=torch.bfloat16)
    maps = compute_anima_similarity_maps(q, k, hidden, {"a":[[0]], "b":[[1]]}, 0.5, 1.0)
    assert maps["a"].dtype == torch.float32
    assert maps["a"][0,:2].min() > maps["a"][0,2:].max()
    assert maps["b"][0,2:].min() > maps["b"][0,:2].max()
    torch.testing.assert_close(maps["a"].sum(), torch.tensor(1.))

@pytest.mark.parametrize("class_name", ["WAN22_Animate", "WAN_Animate2"])
def test_video_animate_models_are_not_anima(class_name):
    model = type(class_name, (), {})()
    assert detect_model_type(model=SimpleNamespace(model=model)) != "anima"


def test_native_anima_ui_workflow_uses_available_loader_and_still_latent():
    import json
    from nodes import CLIPLoader
    workflow = json.loads((Path(__file__).resolve().parents[1] /
        "freefuse_comfyui/workflows/anima_freefuse_complete.json").read_text())
    nodes = {n["id"]:n for n in workflow["nodes"]}
    assert nodes[2]["widgets_values"][1] in CLIPLoader.INPUT_TYPES()["required"]["type"][0]
    assert nodes[10]["type"] == "EmptyCosmosLatentVideo"
    assert nodes[10]["widgets_values"] == [1024,1024,1,1]
    assert nodes[12]["widgets_values"][4] == nodes[16]["widgets_values"][3] == 4.5
    assert nodes[4]["widgets_values"][1] == "starfire"


@pytest.mark.parametrize("prefix", ["single_blocks", "single_transformer_blocks"])
def test_anima_routing_does_not_capture_flux_joint_mlp(prefix):
    hook = MultiAdapterBypassForwardHook(torch.nn.Linear(3,3),
        f"diffusion_model.{prefix}.0.mlp.proj")
    assert hook._should_apply_spatial_mask()
    assert hook._get_mask_type() == "img_with_text"
