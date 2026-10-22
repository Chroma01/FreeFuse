"""Native ComfyUI MOSS Image block patches; all state belongs to a model clone."""
import torch
from torch.nn import functional as F

from .attention_replace import compute_z_image_similarity_maps_with_qkv
from .attention_bias import construct_attention_bias


def _conditional_rows(options, batch, device):
    branches = options.get("cond_or_uncond", [0])
    if not branches or batch % len(branches):
        raise ValueError("MOSS FreeFuse received an invalid CFG batch.")
    size = batch // len(branches)
    rows = [i * size + j for i, branch in enumerate(branches) if branch == 0 for j in range(size)]
    return torch.tensor(rows, device=device, dtype=torch.long)


def _qkv(block, x, mod, pe):
    from comfy.ldm.moss_image.model import modulated_rms_norm, rope_split_half
    shift, scale = mod[0], mod[1]
    y = modulated_rms_norm(x, block.norm1, scale, shift)
    if getattr(block, "fused_qkv", False):
        q, k, v = block.proj["qkv"](y).chunk(3, -1)
    else:
        q, k, v = (block.proj[name](y) for name in ("q", "k", "v"))
    heads = block.num_heads
    q = rope_split_half(block.q_norm(q.unflatten(-1, (heads, -1))), pe).transpose(1, 2)
    k = rope_split_half(block.k_norm(k.unflatten(-1, (heads, -1))), pe).transpose(1, 2)
    v = v.unflatten(-1, (heads, -1)).transpose(1, 2)
    return q, k, v


def _validate_positions(positions, text_length):
    for name, rows in positions.items():
        if not rows or not rows[0] or min(rows[0]) < 0 or max(rows[0]) >= text_length:
            raise ValueError(f"MOSS concept {name!r} is missing or outside the encoded text.")


def apply_moss_collect_patches(model, state, block_indices=None):
    dm = model.model.diffusion_model
    indices = [state.collect_block] if block_indices is None else list(block_indices)
    if not indices or any(not 0 <= i < len(dm.layers) for i in indices):
        raise ValueError("MOSS collection block is outside the transformer.")
    for index in indices:
        block = dm.layers[index]
        def replace(args, extra, block=block, index=index):
            options = args.get("transformer_options", {})
            step = options.get("sigmas_index", state.current_step)
            if not state.is_collect_step(step, index):
                return extra["original_block"](args)
            x = args["img"]
            n = state.collected_outputs["img_seq_len"]
            text_length = x.shape[1] - n
            rows = _conditional_rows(options, x.shape[0], x.device)
            if not rows.numel():
                return extra["original_block"](args)
            _validate_positions(state.token_pos_maps, text_length)
            q, k, v = _qkv(block, x, args["mod"], args["pe"])
            out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
            maps = compute_z_image_similarity_maps_with_qkv(
                q[rows, :, :n].transpose(1, 2), k[rows, :, n:].transpose(1, 2), out[rows, :n],
                text_length, n, state.token_pos_maps, state.top_k_ratio, state.temperature, block.num_heads)
            state.similarity_maps.update(maps)
            state.collected_outputs.setdefault("block_similarity_maps", {})[index] = maps
            state.collected_blocks.add(index)
            return extra["original_block"](args)
        model.set_model_patch_replace(replace, "dit", "single_block", index)


def apply_moss_bias_patches(model, masks, positions, config, bias_blocks="all"):
    dm = model.model.diffusion_model
    count = len(dm.layers)
    if bias_blocks == "none":
        return
    if bias_blocks not in {"all", "last_half"}:
        raise ValueError("MOSS bias_blocks must be 'all', 'last_half', or 'none'.")
    indices = range(count) if bias_blocks == "all" else range(count // 2, count)
    n = next(iter(masks.values())).numel()
    cache = {}
    for index in indices:
        block = dm.layers[index]
        def replace(args, extra, block=block):
            import comfy.ops
            from comfy.ldm.moss_image.model import gated_residual, modulated_rms_norm
            x, mod, pe = args["img"], args["mod"], args["pe"]
            options = args.get("transformer_options", {})
            rows = _conditional_rows(options, x.shape[0], x.device)
            if not rows.numel() or not config.enabled:
                return extra["original_block"](args)
            text_length = x.shape[1] - n
            _validate_positions(positions, text_length)
            key = (text_length, x.device, x.dtype)
            if key not in cache:
                text_first = construct_attention_bias(
                    masks, positions, text_length, n, bias_scale=config.bias_scale,
                    positive_bias_scale=config.positive_bias_scale, bidirectional=config.bidirectional,
                    use_positive_bias=config.use_positive_bias, device=x.device, dtype=x.dtype)
                # The shared builder uses [text,image]; MOSS uses [image,text].
                order = torch.cat((torch.arange(text_length, text_length + n, device=x.device),
                                   torch.arange(text_length, device=x.device)))
                cache[key] = text_first[:, order][:, :, order]
            bias = x.new_zeros(x.shape[0], 1, x.shape[1], x.shape[1])
            bias[rows] = cache[key][:1, None]
            q, k, v = _qkv(block, x, mod, pe)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).flatten(2)
            x = gated_residual(x, block.proj["o"](out), mod[2])
            y = modulated_rms_norm(x, block.norm2, mod[4], mod[3])
            x = gated_residual(x, comfy.ops.linear_input_act(
                block.proj["ffn_down"], block.proj["ffn_up_gate"](y), "situ"), mod[5])
            return {"img": x}
        model.set_model_patch_replace(replace, "dit", "single_block", index)
