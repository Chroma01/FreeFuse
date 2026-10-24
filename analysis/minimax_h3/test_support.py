"""Native H3 parity and explicit packed-row routing checks (no downloaded weights)."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(BASE/"ComfyUI"),str(BASE/"FreeFuse")]
sys.argv = [sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
import comfy.ops
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
from analysis.minimax_h3.support import Collection,patch_model,concept_positions,augmented_attention,video_segment
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
