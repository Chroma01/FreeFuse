"""Native H3 parity and explicit packed-row routing checks (no downloaded weights)."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[2]
if not (BASE/"ComfyUI").is_dir():
    pytest.skip("Place ComfyUI beside FreeFuse for native H3 tests.",allow_module_level=True)
if not (BASE/"ComfyUI/comfy/ldm/minimax/model.py").is_file() or not torch.cuda.is_available():
    pytest.skip("Native H3 parity tests require current ComfyUI and CUDA.",allow_module_level=True)
sys.path[:0] = [str(BASE/"ComfyUI"),str(BASE/"FreeFuse")]
sys.argv = [sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
import comfy.ops
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
from freefuse_comfyui.freefuse_core.minimax_h3_support import Collection,patch_model,concept_positions,augmented_attention,video_segment
from freefuse_comfyui.freefuse_core.bypass_lora_loader import MultiAdapterBypassForwardHook

class Patcher:
    def __init__(self,model):
        self.model=SimpleNamespace(diffusion_model=model)
        self.model_options={"transformer_options":{}}
    def clone(self):
        out=Patcher(self.model.diffusion_model)
        out.model_options=copy.deepcopy(self.model_options)
        return out
    def set_model_patch_replace(self,patch,name,kind,index):
        options=self.model_options["transformer_options"]
        options.setdefault("patches_replace",{}).setdefault(name,{})[(kind,index)]=patch

def native():
    torch.manual_seed(42)
    device=torch.device("cuda")
    model=MiniMaxH3Model(hidden_size=128,num_layers=2,token_refiner_num_layers=1,
        num_attention_heads=2,attention_head_dim=64,ffn_hidden_size=192,text_dim=96,
        timestep_input_dim=32,time_embed_hidden_size=128,time_embed_dim=64,rope_inv_freq_len=8,
        dtype=torch.float32,device=device,operations=comfy.ops.disable_weight_init)
    model.requires_grad_(False)
    with torch.no_grad():
        for p in model.parameters():p.normal_(std=0.03)
    return model

def forward(model,patcher):
    torch.manual_seed(52)
    x=[torch.randn(1,24,2,4,6,device="cuda"),torch.randn(1,32,2,9,device="cuda")]
    ctx=torch.randn(1,7,96,device="cuda")
    with torch.inference_mode():
        return model(x,torch.tensor([500.],device="cuda"),ctx,
            transformer_options=dict(patcher.model_options["transformer_options"]))

def test_native_collection_and_zero_bias_preserve_both_streams():
    model=native();base=Patcher(model)
    expected=forward(model,base)
    state=Collection({"a":[1],"b":[4]},steps=(0,),blocks=(0,1))
    collecting=patch_model(base,collection=state)
    actual=forward(model,collecting)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert len(state.captures)==2
    assert state.captures[(0,1)]["grid"]==(2,2,3)
    assert state.captures[(0,1)]["logits"]["a"].shape==(1,12)
    masks={"a":torch.ones(2,2,3),"b":torch.zeros(2,2,3)}
    zero=patch_model(base,masks=masks,positions=state.positions,bias=True,negative=0,positive=0)
    for a,b in zip(forward(model,zero),expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert base.model_options["transformer_options"]=={}

def test_native_negative_cfg_receives_no_concept_bias():
    model=native();base=Patcher(model)
    masks={"a":torch.ones(2,2,3)}
    negative=patch_model(base,masks=masks,positions={"a":[100]},bias=True)
    negative.model_options["transformer_options"]["cond_or_uncond"]=[1]
    for a,b in zip(forward(model,negative),forward(model,base)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)

def sdpa(q,k,v,heads,**kwargs):
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=kwargs.get("mask"))
    return out.transpose(1,2).flatten(-2)

@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
@pytest.mark.parametrize("head_dim",[64,128])
def test_augmented_attention_matches_explicit_dense_bias(dtype,head_dim):
    torch.manual_seed(123)
    q,k,v=[torch.randn(1,2,20,head_dim,device="cuda",dtype=dtype) for _ in range(3)]
    masks={"a":torch.tensor([1.,1.,0.,0.,1.,0.],device="cuda"),
           "b":torch.tensor([0.,0.,1.,1.,0.,1.],device="cuda")}
    positions={"a":[1,2],"b":[4,5]};video=(14,20)
    bias=torch.zeros(1,1,20,20,device="cuda",dtype=dtype)
    for name,mask in masks.items():
        other="b" if name=="a" else "a"
        for index in positions[name]:bias[0,0,14:20,index]+=2*mask
        for index in positions[other]:bias[0,0,14:20,index]-=4*mask
    expected=sdpa(q,k,v,2,mask=bias)
    actual=augmented_attention(sdpa,q,k,v,2,masks,positions,video)
    tolerance=2e-5 if dtype==torch.float32 else 0.025
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    # Rows outside the video segment get no mathematical bias.
    torch.testing.assert_close(actual[:,:14],sdpa(q,k,v,2)[:,:14],rtol=tolerance,atol=tolerance)

def test_2d_packed_lora_mask_leaves_all_other_rows_neutral():
    layer=torch.nn.Linear(3,3,bias=False)
    layer.weight.data.zero_()
    hook=MultiAdapterBypassForwardHook(layer,"diffusion_model.blocks.0.mlp.fc1")
    hook.original_forward=layer.forward
    hook.mask_enabled=True
    mask=torch.tensor([[[1.,0.,1.],[0.,1.,0.]],[[0.,1.,0.],[1.,0.,1.]]])
    hook.masks={"a":mask};hook.packed_mask_slice=(7,19);hook.txt_len=0
    hook.adapters=[{"adapter":SimpleNamespace(h=lambda x,base:torch.ones_like(base),g=lambda x:x),
                   "offset":None,"adapter_name":"a"}]
    output=hook._bypass_forward(torch.zeros(23,3))
    torch.testing.assert_close(output[:7],torch.ones(7,3))
    torch.testing.assert_close(output[19:],torch.ones(4,3))
    torch.testing.assert_close(output[7:19],mask.reshape(-1,1).expand(-1,3))
    clone=copy.deepcopy(hook)
    assert clone.packed_mask_slice==(7,19)
    hook.packed_mask_slice=None
    torch.testing.assert_close(hook._bypass_forward(torch.zeros(23,3)),torch.ones(23,3))
    hook.packed_mask_slice=(7,20)
    with pytest.raises(ValueError,match="explicit sequence segment"):
        hook._bypass_forward(torch.zeros(23,3))

def test_raw_native_token_positions_with_unicode_and_special_tokens():
    tokenizer=MiniMaxH3Tokenizer()
    clip=SimpleNamespace(tokenizer=tokenizer,tokenize=tokenizer.tokenize_with_weights)
    prompt="Two women: Starfire with red hair; 黑发女性. <d>Hello!</d>"
    concepts={"a":"Starfire with red hair","b":"黑发女性"}
    positions=concept_positions(clip,prompt,concepts)
    ids=[t[0] for t in clip.tokenize(prompt)["qwen3vl_32b"][0]]
    hf=tokenizer.qwen3vl_32b.tokenizer
    for name,concept in concepts.items():
        assert hf.decode([ids[i] for i in positions[name]]).strip()==concept
    with pytest.raises(ValueError,match="missing"):
        concept_positions(clip,prompt,{"missing":"blue shoes"})

def test_layout_tracks_reference_and_audio_rows_separately():
    layout=PackedLayout(7,2,4,6,9,keyframes=[{"resolved_frame_index":0,
        "latent":torch.zeros(1,24,1,4,6)}])
    text,video,grid=video_segment(layout)
    assert text==(0,7) and grid==(2,2,3)
    assert any(kind=="cond" for _,_,kind in layout.segments)
    assert video[1]==layout.seq_len and video[1]-video[0]==12


def test_h3_native_tokenizer_dispatch_is_not_krea2():
    from freefuse_comfyui.freefuse_core.token_utils import detect_model_type,find_concept_positions
    tokenizer=MiniMaxH3Tokenizer()
    clip=SimpleNamespace(tokenizer=tokenizer,tokenize=tokenizer.tokenize_with_weights)
    assert detect_model_type(clip=clip)=="minimax_h3"
    assert detect_model_type(model=Patcher(native()))=="minimax_h3"
    prompt="Starfire greets 黑发女性."
    expected=concept_positions(clip,prompt,{"a":"Starfire","b":"黑发女性"})
    actual=find_concept_positions(clip,prompt,{"a":"Starfire","b":"黑发女性"})
    assert actual=={name:[positions] for name,positions in expected.items()}


def test_lora_disable_survives_rebuilding_injection_hooks():
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import OffsetBypassInjectionManager
    base=torch.nn.Module();base.proj=torch.nn.Linear(3,3,bias=False)
    base.proj.weight.data.zero_()
    adapter=SimpleNamespace(h=lambda x,y:torch.ones_like(y),g=lambda y:y)
    manager=OffsetBypassInjectionManager()
    manager.add_adapter("proj.weight",adapter,adapter_name="a")
    injection=manager.create_injections(base)[0]
    manager.disable_lora()
    patcher=SimpleNamespace(model=copy.deepcopy(base),model_options={
        "transformer_options":{"freefuse_bypass_manager":manager}})
    injection.inject(patcher)
    try:
        assert not manager.hooks[0].lora_enabled
        torch.testing.assert_close(patcher.model.proj(torch.zeros(2,3)),torch.zeros(2,3))
    finally:injection.eject(patcher)
    manager.enable_lora();injection.inject(patcher)
    try:torch.testing.assert_close(patcher.model.proj(torch.zeros(2,3)),torch.ones(2,3))
    finally:injection.eject(patcher)


def test_phase1_preserves_joint_latent_and_returns_clean_branch(monkeypatch):
    import comfy.sample
    from comfy.nested_tensor import NestedTensor
    from freefuse_comfyui.nodes.sampler import FreeFusePhase1Sampler
    import freefuse_comfyui.freefuse_core.minimax_h3_support as h3
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import OffsetBypassInjectionManager
    diffusion=native();base=Patcher(diffusion)
    manager=OffsetBypassInjectionManager()
    base.model_options["transformer_options"]["freefuse_bypass_manager"]=manager
    latent={"samples":NestedTensor((torch.zeros(1,24,2,4,6),torch.zeros(1,32,2,9)))}
    data={"concepts":{"a":"a","b":"b"},"token_pos_maps":{"a":[[1]],"b":[[4]]},"model_type":"minimax_h3"}
    seen=[]
    def sampling(clone,*args,callback=None,**kwargs):
        assert args[7] is latent["samples"]
        assert not clone.model_options["transformer_options"]["freefuse_bypass_manager"]._lora_enabled
        for step in range(10):
            forward(diffusion,clone);seen.append(step);callback(step,None,None,10)
    monkeypatch.setattr(comfy.sample,"sample",sampling)
    split=torch.zeros(2,2,3);split[:,:,:1]=1
    monkeypatch.setattr(h3,"make_mask_volumes",lambda *a,**k:{"a":split,"b":1-split})
    output,payload,preview=FreeFusePhase1Sampler().collect_masks(
        base,[[torch.empty(1,7,96),{"minimax_token_tags":[]}]],[],latent,data,
        42,10,4,1.,"res_multistep","simple",collect_block=1,collect_block_end=1)
    assert seen==[0,1,2,3]
    assert output is base and manager._lora_enabled
    assert "patches_replace" not in base.model_options["transformer_options"]
    assert payload["grid"]==(2,2,3) and preview.shape==(2,256,256,3)


def test_phase1_rejects_unsupported_cfg_and_references_before_sampling():
    from comfy.nested_tensor import NestedTensor
    from freefuse_comfyui.freefuse_core.minimax_h3_support import collect_masks
    latent={"samples":NestedTensor((torch.zeros(1,24,2,4,6),torch.zeros(1,32,2,9)))}
    params=[None,[[torch.empty(1,7,96),{"minimax_token_tags":[]}]],[],latent,{},42,20,4,2.,
            "res_multistep","simple",1,1,100000.,.1,True]
    with pytest.raises(ValueError,match="CFG=1"):collect_masks(*params)
    params[8]=1.;params[1][0][1]["minimax_keyframes"]=[{}]
    with pytest.raises(ValueError,match="reference or guide"):collect_masks(*params)


def test_video_mask_preview_keeps_temporal_frames():
    from freefuse_comfyui.nodes.preview import FreeFuseMaskPreview
    masks={"a":torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]])}
    combined,individual=FreeFuseMaskPreview().preview(
        {"model_type":"minimax_h3","masks":masks},width=32,height=32,
        show_labels=False,show_coverage=False)
    assert combined.shape==(2,32,32,3) and individual.shape[0]==2
    assert not torch.equal(combined[0],combined[1])


def test_standalone_bias_keeps_existing_video_lora_routing():
    from freefuse_comfyui.nodes.mask_applicator import FreeFuseMaskApplicator
    from freefuse_comfyui.nodes.attention_bias import FreeFuseAttentionBias
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import OffsetBypassInjectionManager
    base=Patcher(native())
    base.model_options["transformer_options"]["freefuse_bypass_manager"]=OffsetBypassInjectionManager()
    split=torch.zeros(2,2,3);split[:,:,:1]=1
    payload={"masks":{"a":split,"b":1-split},"model_type":"minimax_h3"}
    data={"adapters":[{"name":"a"},{"name":"b"}],"token_pos_maps":{"a":[[1]],"b":[[4]]}}
    routed,=FreeFuseMaskApplicator().apply_masks(base,payload,data,enable_attention_bias=False)
    biased,_=FreeFuseAttentionBias().apply_bias(routed,payload,data,bias_blocks="last_half")
    patches=biased.model_options["transformer_options"]["patches_replace"]["dit"]
    assert patches[("double_block",0)].route and not patches[("double_block",0)].bias
    assert patches[("double_block",1)].route and patches[("double_block",1)].bias
    assert not routed.model_options["transformer_options"]["patches_replace"]["dit"][("double_block",1)].bias
    assert "patches_replace" not in base.model_options["transformer_options"]


def test_h3_rejects_masks_from_another_video_length():
    masks={"a":torch.ones(3,2,3)}
    model=native();patched=patch_model(Patcher(model),masks=masks,positions={"a":[1]},bias=True)
    with pytest.raises(ValueError,match="sampled video grid"):
        forward(model,patched)


@pytest.mark.parametrize("initial_flag",[None,False,True])
def test_native_fused_activation_cannot_skip_bypass_lora(monkeypatch,initial_flag):
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import OffsetBypassInjectionManager
    core=torch.nn.Module();core.proj=torch.nn.Linear(3,3,bias=False)
    core.proj.weight.data.zero_()
    if initial_flag is not None:core.proj.comfy_force_forward=initial_flag
    adapter=SimpleNamespace(h=lambda x,y:torch.ones_like(y),g=lambda y:y)
    manager=OffsetBypassInjectionManager();manager.add_adapter("proj.weight",adapter,adapter_name="a")
    injection=manager.create_injections(core)[0]
    patcher=SimpleNamespace(model=core,model_options={"transformer_options":{"freefuse_bypass_manager":manager}})
    called=[]
    monkeypatch.setattr(comfy.ops,"_fp16_linear_wanted",lambda x:True)
    monkeypatch.setattr(comfy.ops,"linear_input_act_",lambda x,*a,**k:called.append(True) or torch.zeros_like(x))
    injection.inject(patcher)
    try:
        actual=comfy.ops.linear_input_act(core.proj,torch.zeros(2,3),None)
        torch.testing.assert_close(actual,torch.ones(2,3))
        assert not called
    finally:injection.eject(patcher)
    assert ("comfy_force_forward" in vars(core.proj))==(initial_flag is not None)
    assert getattr(core.proj,"comfy_force_forward",None)==initial_flag


def test_native_all_ones_temporal_routing_preserves_both_streams():
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import OffsetBypassInjectionManager
    model=native();base=Patcher(model)
    manager=OffsetBypassInjectionManager()
    manager.add_adapter("diffusion_model.blocks.0.mlp.fc2.weight",
        SimpleNamespace(h=lambda x,y:torch.ones_like(y)*.01,g=lambda y:y),adapter_name="a")
    injection=manager.create_injections(base.model)[0]
    base.model_options["transformer_options"]["freefuse_bypass_manager"]=manager
    routed=patch_model(base,masks={"a":torch.ones(2,2,3)},positions={"a":[1]},route=True)
    results=[]
    for branch in (base,routed):
        injection.inject(branch)
        try:results.append(forward(model,branch))
        finally:injection.eject(branch)
    for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert manager._pending_masks is None


def test_new_collection_does_not_reuse_old_video_geometry():
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import OffsetBypassInjectionManager
    model=native();base=Patcher(model)
    base.model_options["transformer_options"]["freefuse_bypass_manager"]=OffsetBypassInjectionManager()
    routed=patch_model(base,masks={"a":torch.ones(3,2,3)},positions={"a":[1]},route=True)
    state=Collection({"a":[1]},steps=(0,),blocks=(0,))
    collecting=patch_model(routed,collection=state)
    forward(model,collecting)
    assert state.captures[(0,0)]["grid"]==(2,2,3)


def test_h3_missing_background_is_optional_but_subjects_are_strict():
    from freefuse_comfyui.freefuse_core.token_utils import find_background_positions,find_concept_positions
    tokenizer=MiniMaxH3Tokenizer()
    clip=SimpleNamespace(tokenizer=tokenizer,tokenize=tokenizer.tokenize_with_weights)
    assert find_background_positions(clip,"Starfire waves.","a missing garden")==[]
    with pytest.raises(ValueError,match="missing"):
        find_concept_positions(clip,"Starfire waves.",{"albedo":"Albedo"})
