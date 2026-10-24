"""Experimental FreeFuse integration for native ComfyUI MiniMax H3.

All geometry comes from H3 PackedLayout. No production H3 support is claimed.
"""
from dataclasses import dataclass, field
import math
import torch
import torch.nn.functional as F
from freefuse_comfyui.freefuse_core.mask_utils import generate_masks


def concept_positions(clip, prompt, concepts):
    """Index raw T2VA text with the actual native Qwen token stream."""
    token_batches = clip.tokenize(prompt)["qwen3vl_32b"]
    if len(token_batches) != 1 or any(not isinstance(t[0], int) for t in token_batches[0]):
        raise ValueError("This experiment accepts text-only H3 conditioning in one chunk.")
    ids = [t[0] for t in token_batches[0]]
    tokenizer = clip.tokenizer.qwen3vl_32b.tokenizer
    raw = prompt.encode("utf-8")
    spans = []
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        if ids != encoded["input_ids"]:
            raise ValueError("Native H3 conditioning tokens differ from prompt offsets.")
        offsets = [(len(prompt[:a].encode("utf-8")), len(prompt[:b].encode("utf-8")))
                   for a, b in encoded["offset_mapping"]]
    else:
        # Byte-level BPE offsets avoid U+FFFD at split Unicode characters.
        joined = b""
        offsets = []
        for token_id, piece in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
            value = (piece.encode("utf-8") if token_id in tokenizer.all_special_ids else
                     bytes(tokenizer.byte_decoder[c] for c in piece))
            offsets.append((len(joined), len(joined) + len(value)))
            joined += value
        if not joined.startswith(raw):
            raise ValueError("Native H3 byte tokens do not reproduce the prompt.")
    result = {}
    for name, concept in concepts.items():
        needle = concept.encode("utf-8")
        if not needle:
            raise ValueError("Concept text must not be empty.")
        matched = set()
        start = 0
        while True:
            start = raw.find(needle, start)
            if start < 0:
                break
            stop = start + len(needle)
            matched.update(i for i, (a,b) in enumerate(offsets) if b > start and a < stop)
            start += 1
        if not matched:
            raise ValueError(f"Concept {name!r} is missing from the H3 prompt.")
        result[name] = sorted(matched)
    return result


def video_segment(layout):
    text = next((a,b) for a,b,kind in layout.segments if kind == "text")
    video = next((a,b) for a,b,kind in layout.segments if kind == "video")
    _, t, h, w, _ = layout.signature
    grid = (t, h // 2, w // 2)
    if video[1] - video[0] != math.prod(grid):
        raise ValueError("H3 target-video span does not match its patch grid.")
    return text, video, grid


def similarity_logits(q, k, output, positions, video, top_k_ratio=0.1):
    """Competitive native-QK cores followed by FreeFuse feature similarity."""
    start, stop = video
    queries = q[:, :, start:stop].float()
    keys = k.float()
    hidden = output[:, start:stop].float()
    scores = {}
    for name, indices in positions.items():
        logits = torch.matmul(queries, keys[:, :, indices].transpose(-1,-2))
        scores[name] = (logits / math.sqrt(q.shape[-1])).softmax(dim=2).mean(1).mean(-1)
    subjects = [name for name in scores if not name.startswith("__")]
    count = max(1, int((stop-start) * top_k_ratio))
    result = {}
    for name, score in scores.items():
        if name in subjects:
            score = score * len(subjects)
            for other in subjects:
                if other != name:
                    score = score - scores[other]
        indices = score.topk(count,dim=-1).indices
        core = hidden.gather(1, indices[...,None].expand(-1,-1,hidden.shape[-1])).mean(1,keepdim=True)
        result[name] = torch.bmm(core, hidden.transpose(-1,-2)).squeeze(1)
    return result


def make_mask_volumes(logits, grid, temperature=1.0, temporal_window=3):
    """Apply existing image mask balancing framewise on the actual video grid."""
    if temperature <= 0 or temporal_window not in (1,3,5):
        raise ValueError("Use a positive temperature and temporal_window 1, 3 or 5.")
    names = list(logits)
    t,h,w = grid
    probabilities = torch.stack([logits[name].float().reshape(t,h*w).softmax(-1)
                                if temperature == 1 else
                                (logits[name].float().reshape(t,h*w)/temperature).softmax(-1)
                                for name in names]).reshape(len(names),t,h,w)
    if temporal_window > 1:
        probabilities = F.avg_pool3d(probabilities[None],(temporal_window,1,1),
            stride=1,padding=(temporal_window//2,0,0),count_include_pad=False)[0]
    per_frame = []
    for frame in range(t):
        maps = {name:probabilities[i,frame].reshape(1,h*w,1) for i,name in enumerate(names)}
        per_frame.append(generate_masks(maps,include_background=True,method="stabilized",
            bg_scale=0.95,use_morphological_cleaning=True,debug=False,max_iter=15,
            latent_h=h,latent_w=w))
    masks = {name:torch.stack([m[name] for m in per_frame]) for name in per_frame[0]}
    for name in names:
        if not name.startswith("__") and (name not in masks or not bool(masks[name].any())):
            raise ValueError(f"Automatic H3 mask is empty for {name}; no fallback was substituted.")
    return masks


def augmented_attention(call, q, k, v, heads, masks, positions, video, negative=4.0, positive=2.0, **kwargs):
    """Factor image-to-text bias into auxiliary Q/K channels, without an S² mask.

    In real arithmetic this equals an additive bias. Quantized activations add
    rounding error, measured against explicit dense attention in the tests.
    """
    if negative == 0 and positive == 0:
        return call(q,k,v,heads,**kwargs)
    start, stop = video
    names = list(masks)
    dim = q.shape[-1]
    new_dim = ((dim + len(names) + 7) // 8) * 8
    q_aug = F.pad(q, (0,new_dim-dim))
    k_aug = F.pad(k, (0,new_dim-dim))
    v_aug = F.pad(v, (0,new_dim-dim))
    q_aug[...,:dim] *= math.sqrt(new_dim/dim)
    owners = torch.full((k.shape[2],),-1,device=k.device,dtype=torch.long)
    for owner,name in enumerate(names):
        for index in positions[name]:
            owners[index] = owner if owners[index] == -1 else -2
    for owner,name in enumerate(names):
        mask = masks[name].reshape(-1).to(q)
        if mask.numel() != stop-start:
            raise ValueError("Attention-bias mask does not match target video rows.")
        q_aug[:,:,start:stop,dim+owner] = mask * math.sqrt(new_dim)
        values = (owners == owner).to(k.dtype)*positive
        values -= ((owners >= 0) & (owners != owner)).to(k.dtype)*negative
        k_aug[:,:,:,dim+owner] = values
    result = call(q_aug,k_aug,v_aug,heads,**kwargs)
    return result.reshape(q.shape[0],q.shape[2],heads,new_dim)[...,:dim].flatten(-2).contiguous()


@dataclass
class Collection:
    positions: dict
    steps: tuple = (3,6,10)
    blocks: tuple = (10,20,30,40)
    top_k_ratio: float = 0.1
    current_step: int = 0
    captures: dict = field(default_factory=dict)


class H3Attention:
    def __init__(self, layout, collection=None, masks=None, positions=None,
                 bias=False, negative=4.0,positive=2.0,previous=None):
        self.layout, self.collection, self.masks = layout, collection, masks
        self.positions, self.bias = positions, bias
        self.negative, self.positive, self.previous = negative, positive, previous

    def __call__(self,original,q,k,v,heads,*args,**kwargs):
        if args:
            raise ValueError("H3 attention parameters must be passed by keyword.")
        call = original if self.previous is None else lambda *a,**kw:self.previous(original,*a,**kw)
        options = kwargs.get("transformer_options",{})
        if options.get("cond_or_uncond",[0]) != [0]:
            return call(q,k,v,heads,**kwargs)
        text, video, grid = video_segment(self.layout)
        state = self.collection
        step = state.current_step if state is not None else None
        block = options.get("block_index")
        collecting = (state is not None and step in state.steps and block in state.blocks
                      and (step,block) not in state.captures)
        positions = state.positions if collecting else self.positions
        if collecting or self.bias:
            if not kwargs.get("skip_reshape") or q.ndim != 4 or q.shape[2] != self.layout.seq_len:
                raise ValueError("H3 attention did not expose native packed Q/K/V.")
            for name,indices in positions.items():
                if not indices or min(indices)<text[0] or max(indices)>=text[1]:
                    raise ValueError(f"H3 concept {name!r} lies outside the native text segment.")
        if self.bias:
            output = augmented_attention(call,q,k,v,heads,self.masks,positions,video,
                self.negative,self.positive,**kwargs)
        else:
            output = call(q,k,v,heads,**kwargs)
        if collecting:
            maps = similarity_logits(q,k,output,positions,video,state.top_k_ratio)
            state.captures[(step,block)] = {
                "grid":grid,"segments":self.layout.segments,
                "logits":{name:values.detach().cpu() for name,values in maps.items()},
            }
        return output


class H3Block:
    def __init__(self, collection=None,masks=None,positions=None,route=False,bias=False,
                 negative=4.0,positive=2.0):
        self.collection,self.masks,self.positions = collection,masks,positions
        self.route,self.bias,self.negative,self.positive = route,bias,negative,positive

    def __call__(self,args,extra):
        options = dict(args["transformer_options"])
        index = options["block_index"]
        layout = args["layout"]
        _,video,_ = video_segment(layout)
        state = self.collection
        collect = state is not None and index in state.blocks and state.current_step in state.steps
        if collect or self.bias:
            previous = options.get("optimized_attention_override")
            options["optimized_attention_override"] = H3Attention(layout,state if collect else None,
                self.masks,self.positions,self.bias,self.negative,self.positive,previous)
        manager = options.get("freefuse_bypass_manager")
        configured = []
        if self.route and manager is not None:
            prefix = f"diffusion_model.blocks.{index}."
            for hook in manager.hooks:
                if hook.module_key.startswith(prefix) and any(part in hook.module_key for part in (".attn.",".mlp.")):
                    old = hook.packed_mask_slice
                    hook.packed_mask_slice = video
                    configured.append((hook,old))
        try:
            return extra["original_block"](dict(args,transformer_options=options))
        finally:
            for hook,old in configured:
                hook.packed_mask_slice = old


def patch_model(model, collection=None,masks=None,positions=None,route=False,bias=False,
                negative=4.0,positive=2.0):
    clone = model.clone()
    if route:
        manager = clone.model_options["transformer_options"]["freefuse_bypass_manager"]
        subjects = {n:m for n,m in masks.items() if not n.startswith("_")}
        manager.set_masks(subjects,tuple(next(iter(subjects.values())).shape[-2:]),txt_len=0)
    subjects = None if masks is None else {n:m for n,m in masks.items() if not n.startswith("_")}
    block = H3Block(collection,subjects,positions,route,bias,negative,positive)
    indices = range(len(clone.model.diffusion_model.blocks)) if route or bias else collection.blocks
    for index in indices:
        clone.set_model_patch_replace(block,"dit","double_block",index)
    return clone
