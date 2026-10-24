"""Native MiniMax H3 text-to-audio-video collection and packed video routing.

Geometry always comes from H3 PackedLayout. Masks are [time,height,width]
volumes; text, reference and audio rows retain their global LoRA contribution.
"""
from dataclasses import dataclass, field
import math
import torch
import torch.nn.functional as F
from .mask_utils import generate_masks
from .token_offsets import native_byte_offsets


def concept_positions(clip, prompt, concepts):
    """Index raw T2VA text with the actual native Qwen token stream."""
    token_batches = clip.tokenize(prompt)["qwen3vl_32b"]
    if len(token_batches) != 1 or any(not isinstance(t[0], int) for t in token_batches[0]):
        raise ValueError("This experiment accepts text-only H3 conditioning in one chunk.")
    ids = [t[0] for t in token_batches[0]]
    tokenizer = clip.tokenizer.qwen3vl_32b.tokenizer
    raw = prompt.encode("utf-8")
    offsets = native_byte_offsets(tokenizer, ids, prompt)
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
    hidden = output[:, start:stop].float()
    scores = {}
    for name, indices in positions.items():
        logits = torch.matmul(queries, k[:, :, indices].float().transpose(-1,-2))
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


def make_mask_volumes(logits, grid, temperature=1.0, temporal_window=3,
                      include_background=True, **mask_options):
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
        options = {"bg_scale":0.95,"use_morphological_cleaning":True,"max_iter":15}
        options.update(mask_options)
        per_frame.append(generate_masks(maps,include_background=include_background,
            method="stabilized",debug=False,latent_h=h,latent_w=w,**options))
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
    # Build on CPU to avoid a CUDA synchronization for every text token.
    ownership = [-1] * k.shape[2]
    for owner,name in enumerate(names):
        for index in positions[name]:
            ownership[index] = owner if ownership[index] == -1 else -2
    owners = torch.tensor(ownership,device=k.device,dtype=torch.long)
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


def single_prompt_positions(positions):
    """Accept FreeFuse's per-prompt payload or the flat native experiment form."""
    result = {}
    for name, indices in positions.items():
        if indices and isinstance(indices[0], (list, tuple)):
            if len(indices) != 1:
                raise ValueError("MiniMax H3 FreeFuse supports one prompt/video at a time.")
            indices = indices[0]
        if not indices or any(not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError(f"MiniMax H3 requires valid token positions for {name!r}.")
        result[name] = list(indices)
    return result


def validate_masks(masks):
    subjects = {name: mask for name, mask in masks.items() if not name.startswith("_")}
    if not subjects:
        raise ValueError("MiniMax H3 needs at least one subject mask volume.")
    shape = None
    for name, mask in subjects.items():
        if mask.ndim != 3 or min(mask.shape) < 1:
            raise ValueError(f"MiniMax H3 mask {name!r} must have shape [time,height,width].")
        if not bool(torch.isfinite(mask).all()) or bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError(f"MiniMax H3 mask {name!r} must contain finite values in [0,1].")
        if shape is not None and tuple(mask.shape) != shape:
            raise ValueError("MiniMax H3 mask volumes must share the same grid.")
        shape = tuple(mask.shape)
    return subjects, shape


class H3Block:
    """Clone-local block replacement; never mutate native block forward methods."""

    def __init__(self, collection=None, masks=None, positions=None, route=False,
                 bias=False, negative=4.0, positive=2.0, previous=None):
        self.collection, self.masks, self.positions = collection, masks, positions
        self.route, self.bias = route, bias
        self.negative, self.positive, self.previous = negative, positive, previous

    def __call__(self, args, extra):
        options = dict(args["transformer_options"])
        index, layout = options["block_index"], args["layout"]
        _, video, grid = video_segment(layout)
        if self.masks is not None and tuple(next(iter(self.masks.values())).shape) != grid:
            raise ValueError(f"MiniMax H3 masks do not match the sampled video grid {grid}.")
        state = self.collection
        collect = state is not None and index in state.blocks and state.current_step in state.steps
        if collect or self.bias:
            options["optimized_attention_override"] = H3Attention(
                layout, state if collect else None, self.masks, self.positions,
                self.bias, self.negative, self.positive,
                options.get("optimized_attention_override"),
            )
        manager = options.get("freefuse_bypass_manager")
        configured = []
        if self.route:
            if manager is None:
                raise ValueError("Load H3 character LoRAs with FreeFuse LoRA Loader (Bypass).")
            prefix = f"diffusion_model.blocks.{index}."
            for hook in manager.hooks:
                if hook.module_key.startswith(prefix) and any(
                    part in hook.module_key for part in (".attn.", ".mlp.")
                ):
                    configured.append((hook, hook.packed_mask_slice))
                    hook.packed_mask_slice = video
        try:
            patched = dict(args, transformer_options=options)
            if self.previous is not None:
                return self.previous(patched, extra)
            return extra["original_block"](patched)
        finally:
            for hook, previous in configured:
                hook.packed_mask_slice = previous


def install_patches(model, collection=None, masks=None, positions=None, route=None,
                    bias=False, negative=4.0, positive=2.0, bias_blocks="all"):
    """Install on an already cloned patcher, composing with existing native patches."""
    if bias_blocks not in {"all", "last_half", "none", "double_stream_only"}:
        raise ValueError("For MiniMax H3, bias_blocks must be all, last_half or none.")
    subjects = None
    if masks is not None:
        subjects, _ = validate_masks(masks)
    positions = single_prompt_positions(positions) if positions is not None else None
    if bias and (subjects is None or positions is None or set(subjects) - set(positions)):
        raise ValueError("MiniMax H3 attention bias needs matching subject masks and token positions.")
    options = model.model_options["transformer_options"]
    if route:
        manager = options.get("freefuse_bypass_manager")
        if manager is None:
            raise ValueError("MiniMax H3 spatial routing requires FreeFuse bypass LoRA loaders.")
        registered = {a["adapter_name"] for values in manager.adapters.values() for a in values}
        if registered - set(subjects):
            raise ValueError(f"Missing H3 masks for adapters: {sorted(registered-set(subjects))}")
        manager.set_masks(subjects, tuple(next(iter(subjects.values())).shape[-2:]), txt_len=0)
    count = len(model.model.diffusion_model.blocks)
    if collection is not None and (not collection.blocks or any(
        i < 0 or i >= count for i in collection.blocks
    )):
        raise ValueError(f"MiniMax H3 collection block must be in 0..{count-1}.")
    replacements = options.get("patches_replace", {}).get("dit", {})
    for index in range(count):
        previous = replacements.get(("double_block", index))
        existing = previous if isinstance(previous, H3Block) else None
        active_route = existing.route if route is None and existing else bool(route)
        active_bias = bias and bias_blocks != "none" and (
            bias_blocks != "last_half" or index >= count // 2
        )
        if collection is None and not active_route and not active_bias and existing is None:
            continue
        # Replacing our own patch changes its configuration instead of stacking bias twice.
        if existing:
            previous = existing.previous
        model.set_model_patch_replace(H3Block(
            collection, subjects if subjects is not None else (existing.masks if existing and active_route else None),
            positions, active_route, active_bias, negative, positive, previous,
        ), "dit", "double_block", index)


def patch_model(model, collection=None, masks=None, positions=None, route=False,
                bias=False, negative=4.0, positive=2.0):
    """Compatibility entry point for the reproducible H3 experiments."""
    clone = model.clone()
    install_patches(clone, collection, masks, positions, route, bias, negative, positive)
    return clone


def video_latent(samples):
    """Validate the native joint AV latent without flattening or dropping audio."""
    if not getattr(samples, "is_nested", False):
        raise ValueError("Use Empty MiniMax H3 AV Latent: FreeFuse requires a native joint AV latent.")
    streams = samples.unbind()
    if len(streams) != 2:
        raise ValueError("MiniMax H3 expects exactly one video and one audio stream.")
    video, audio = streams
    if (video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24
            or audio.ndim != 4 or audio.shape[:3] != (1, 32, 2)):
        raise ValueError("MiniMax H3 FreeFuse requires a single-video batch with native video/audio channels.")
    if video.shape[-2] % 2 or video.shape[-1] % 2:
        raise ValueError("MiniMax H3 video latent dimensions must be divisible by two.")
    return video


def mask_preview(masks, width=256, height=256):
    """One RGB IMAGE per latent time position, preserving the entire mask volume."""
    colors = ((1., .25, .25), (.25, 1., .5), (.35, .55, 1.), (1., .8, .2))
    first = next(iter(masks.values()))
    result = torch.zeros(first.shape[0], 3, height, width)
    subject = 0
    for name, volume in masks.items():
        if name.startswith("_"):
            color = (.3, .3, .3)
        else:
            color = colors[subject % len(colors)]
            subject += 1
        resized = F.interpolate(volume.detach().float().cpu()[:, None],
                                (height, width), mode="nearest")
        result += resized * torch.tensor(color).view(1, 3, 1, 1)
    return result.clamp(0, 1).movedim(1, -1)


class _CollectionComplete(Exception):
    pass


def collect_masks(model, conditioning, neg_conditioning, latent, freefuse_data,
                  seed, steps, collect_step, cfg, sampler_name, scheduler,
                  collect_block, collect_block_end, temperature, top_k_ratio,
                  disable_lora_phase1, include_background=True, temporal_window=3,
                  sigmas=None, **mask_options):
    """Run the native partial sigma schedule, then balance each temporal mask frame."""
    import comfy.sample
    video = video_latent(latent["samples"])
    if cfg != 1.0:
        raise ValueError("MiniMax H3 FreeFuse currently requires CFG=1 (native distilled model).")
    if not 1 <= collect_step <= steps:
        raise ValueError("MiniMax H3 collect_step must be between 1 and total steps.")
    if collect_block_end != collect_block:
        raise ValueError("MiniMax H3 currently collects one block; set collect_block_end = collect_block.")
    if not conditioning or any(
        any(key.startswith("minimax_") and key != "minimax_token_tags" for key in meta)
        for _, meta in conditioning
    ):
        raise ValueError("MiniMax H3 FreeFuse currently supports text-to-audio-video without reference or guide conditioning.")
    if any("minimax_token_tags" not in meta for _, meta in conditioning):
        raise ValueError("Use the native MiniMax H3 text encoder and preserve minimax_token_tags.")
    positions = single_prompt_positions(freefuse_data.get("token_pos_maps", {}))
    concepts = freefuse_data.get("concepts", {})
    if not concepts or set(concepts) - set(positions):
        raise ValueError("Compute token positions for every H3 subject before Phase 1.")
    clone = model.clone()
    state = Collection(positions, steps=(collect_step-1,), blocks=(collect_block,),
                       top_k_ratio=top_k_ratio)
    install_patches(clone, collection=state, route=False)
    manager = clone.model_options["transformer_options"].get("freefuse_bypass_manager")
    if manager is None:
        raise ValueError("Load H3 character LoRAs with FreeFuse LoRA Loader (Bypass).")
    if disable_lora_phase1:
        manager.disable_lora()

    def callback(step, x0, x, total):
        state.current_step = step + 1
        if (collect_step-1, collect_block) in state.captures:
            raise _CollectionComplete()

    try:
        noise = comfy.sample.prepare_noise(latent["samples"], seed, None)
        comfy.sample.sample(clone, noise, steps, cfg, sampler_name, scheduler,
                            conditioning, neg_conditioning, latent["samples"],
                            seed=seed, callback=callback, sigmas=sigmas,
                            force_full_denoise=False)
    except _CollectionComplete:
        pass
    finally:
        if disable_lora_phase1:
            manager.enable_lora()
    capture = state.captures.get((collect_step-1, collect_block))
    if capture is None:
        raise RuntimeError("MiniMax H3 did not expose collection attention; update ComfyUI to a version with native H3 block patches.")
    grid = (video.shape[2], video.shape[3] // 2, video.shape[4] // 2)
    if tuple(capture["grid"]) != grid:
        raise ValueError("Collected MiniMax H3 grid differs from the input video latent.")
    masks = make_mask_volumes(capture["logits"], grid, temperature or 100000.,
                              temporal_window, include_background, **mask_options)
    payload = {"masks": masks, "model_type": "minimax_h3", "grid": grid,
               "collection": {"step": collect_step, "block": collect_block,
                              "temperature": temperature or 100000.,
                              "temporal_window": temporal_window}}
    # Return the input branch, never the temporary collector or a partial AV latent.
    return model, payload, mask_preview(masks)
