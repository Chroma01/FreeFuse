"""Anima/Cosmos attention integration using clone-local ComfyUI patch options.

Cross-attention consumes T5-aligned adapter outputs, not raw Qwen tokens.
No shared model module or forward method is mutated.
"""
import torch


def _rows(options, batch):
    branches = options.get("cond_or_uncond", [0])
    if not branches or batch % len(branches):
        raise ValueError("Anima received an invalid CFG batch layout.")
    size = batch // len(branches)
    return [(i * size + j, j) for i, branch in enumerate(branches) if branch == 0 for j in range(size)]


def _positions(mapping, sample, text_length):
    result = {}
    for name, per_prompt in mapping.items():
        if not per_prompt:
            raise ValueError(f"Anima concept {name!r} has no positions.")
        if len(per_prompt) != 1 and sample >= len(per_prompt):
            raise ValueError("Anima token positions do not match the prompt batch.")
        positions = per_prompt[0 if len(per_prompt) == 1 else sample]
        if not positions or min(positions) < 0 or max(positions) >= text_length:
            raise ValueError(f"Anima concept {name!r} is outside the adapted T5 sequence.")
        result[name] = [positions]
    return result


def _post_input(args):
    image = args["img"]
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError("Anima FreeFuse currently supports still images only.")
    data = args["transformer_options"]["model_patch_data"]
    data["freefuse_anima_shape"] = tuple(image.shape[2:4])
    data["freefuse_anima_attention"] = None
    return args


def _mark_self(q, k, v, extra_options, **kwargs):
    extra_options["model_patch_data"]["freefuse_anima_attention"] = "self"
    return {}


def _mark_cross(q, k, v, extra_options, **kwargs):
    extra_options["model_patch_data"]["freefuse_anima_attention"] = "cross"
    return {}


def compute_anima_similarity_maps(q, k, image_hidden, positions, top_k_ratio, temperature):
    """FreeFuse's competitive cores and feature similarity in native Anima units.

    Q/K are head-shaped and already normalized by Cosmos. Keep score/probability
    math in float32: bfloat16 rounds the small per-pixel probabilities to ties.
    """
    import math
    if not 0 < top_k_ratio <= 1 or temperature <= 0:
        raise ValueError("Anima requires top_k_ratio in (0,1] and a positive temperature.")
    q, k, hidden = q.float(), k.float(), image_hidden.float()
    scores = {}
    for name, per_prompt in positions.items():
        logits = torch.matmul(q, k[:, :, per_prompt[0]].transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores[name] = logits.softmax(dim=2).mean(dim=1).mean(dim=-1)
    subject_names = [name for name in scores if not name.startswith("__")]
    result = {}
    count = max(1, int(q.shape[2] * top_k_ratio))
    for name, score in scores.items():
        if name in subject_names:
            score = score * len(subject_names)
            for other in subject_names:
                if other != name:
                    score = score - scores[other]
        indices = score.topk(count, dim=-1).indices
        core = hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1]))
        # Average the core first, avoiding a [core_tokens,image_tokens] matrix.
        logits = torch.bmm(core.mean(dim=1, keepdim=True), hidden.transpose(-1, -2))
        result[name] = (logits / temperature).softmax(dim=-1).transpose(1, 2)
    return result


class AnimaAttentionOverride:
    def __init__(self, previous=None, state=None, indices=None, masks=None, positions=None, config=None):
        self.previous = previous
        self.state = state
        self.indices = set(indices or [])
        self.masks = masks
        self.positions = positions
        self.config = config

    def __call__(self, original, q, k, v, heads, *args, **kwargs):
        options = kwargs.get("transformer_options", {})
        data = options.get("model_patch_data", {})
        cross = data.get("freefuse_anima_attention") == "cross"
        index = options.get("block_index")
        collecting = (cross and self.state is not None and index in self.indices
                      and self.state.is_collect_step(options.get("sigmas_index", self.state.current_step), index))
        applying = cross and self.masks is not None and index in self.indices and self.config.enabled
        call = original if self.previous is None else lambda *a, **kw: self.previous(original, *a, **kw)
        if not collecting and not applying:
            return call(q, k, v, heads, *args, **kwargs)
        if not kwargs.get("skip_reshape", False) or q.ndim != 4 or k.ndim != 4:
            raise ValueError("Anima attention backend must expose normalized head-shaped Q/K/V.")
        rows = _rows(options, q.shape[0])
        if not rows:
            return call(q, k, v, heads, *args, **kwargs)
        height, width = data["freefuse_anima_shape"]
        if height * width != q.shape[2]:
            raise ValueError("Anima image token grid does not match the attention query length.")
        mapping = self.state.token_pos_maps if collecting else self.positions
        positions = {row: _positions(mapping, sample, k.shape[2]) for row, sample in rows}
        if applying:
            bias = q.new_zeros(q.shape[0], 1, q.shape[2], k.shape[2])
            for row, sample in rows:
                owners = torch.full((k.shape[2],), -1, device=q.device, dtype=torch.long)
                names = list(self.masks)
                # Shared subject tokens remain neutral rather than being owned by the last LoRA.
                for owner, name in enumerate(names):
                    for pos in positions[row][name][0]:
                        owners[pos] = owner if owners[pos] == -1 else -2
                for owner, name in enumerate(names):
                    mask = self.masks[name]
                    if mask.ndim != 2:
                        raise ValueError("Anima masks must be [height,width], without a batch axis.")
                    mask = torch.nn.functional.interpolate(mask[None, None].float(), size=(height, width),
                                                         mode="nearest").reshape(-1).to(q)
                    other = (owners >= 0) & (owners != owner)
                    bias[row, 0] -= self.config.bias_scale * mask[:, None] * other[None, :]
                    if self.config.use_positive_bias:
                        bias[row, 0] += self.config.positive_bias_scale * mask[:, None] * (owners == owner)[None, :]
            old_mask = kwargs.get("mask")
            if args:
                raise ValueError("Anima attention mask must be passed by keyword.")
            if old_mask is not None:
                if old_mask.ndim == 2:
                    old_mask = old_mask[None, None]
                elif old_mask.ndim == 3:
                    old_mask = old_mask[:, None]
                if old_mask.dtype == torch.bool:
                    bias = bias.masked_fill(~old_mask, float("-inf"))
                else:
                    bias = bias + old_mask
            kwargs = dict(kwargs, mask=bias)
        result = call(q, k, v, heads, *args, **kwargs)
        if collecting:
            if result.ndim != 3:
                raise ValueError("Anima collection requires an image-sequence attention output.")
            maps = {}
            for row, sample in rows:
                row_maps = compute_anima_similarity_maps(
                    q[row:row+1], k[row:row+1], result[row:row+1],
                    positions[row], self.state.top_k_ratio, self.state.temperature)
                for name, values in row_maps.items():
                    maps.setdefault(name, []).append(values)
            maps = {name: torch.cat(values) for name, values in maps.items()}
            self.state.similarity_maps.update(maps)
            self.state.collected_outputs.setdefault("block_similarity_maps", {})[index] = maps
            self.state.collected_outputs["anima_grid"] = (height, width)
            self.state.collected_blocks.add(index)
        return result


def _install(model, **kwargs):
    options = model.model_options["transformer_options"]
    previous = options.get("optimized_attention_override")
    if isinstance(previous, AnimaAttentionOverride):
        previous = previous.previous
    options["optimized_attention_override"] = AnimaAttentionOverride(previous=previous, **kwargs)
    if not options.get("freefuse_anima_markers"):
        model.set_model_patch(_post_input, "post_input")
        model.set_model_attn1_patch(_mark_self)
        model.set_model_attn2_patch(_mark_cross)
        options["freefuse_anima_markers"] = True


def apply_anima_collect_patches(model, state, block_indices=None):
    count = len(model.model.diffusion_model.blocks)
    indices = [state.collect_block] if block_indices is None else list(block_indices)
    if not indices or any(i < 0 or i >= count for i in indices):
        raise ValueError(f"Anima collection blocks must be in 0..{count-1}.")
    _install(model, state=state, indices=indices)


def apply_anima_bias_patches(model, masks, positions, config, bias_blocks="all"):
    count = len(model.model.diffusion_model.blocks)
    if bias_blocks == "none":
        return
    if bias_blocks not in {"all", "last_half"}:
        raise ValueError("Anima bias_blocks must be 'all', 'last_half', or 'none'.")
    if not masks or set(masks) - set(positions):
        raise ValueError("Anima attention bias requires masks and matching subject positions.")
    indices = range(count) if bias_blocks == "all" else range(count // 2, count)
    _install(model, masks=masks, positions=positions, config=config, indices=indices)
