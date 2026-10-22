"""MOSS joint attention: image tokens precede the unpadded text tokens."""
import torch
from torch.nn import functional as F

from src.vendor.moss_image import _rotate


class FreeFuseMossImageAttnProcessor:
    def __call__(self, block, x, rope, modulation):
        state = block._freefuse_state
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = block.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        y = y.to(block.proj["q"].weight.dtype)
        q = block.q_norm(block.proj["q"](y).unflatten(-1, (block.heads, -1)))
        k = block.k_norm(block.proj["k"](y).unflatten(-1, (block.heads, -1)))
        q = _rotate(q, rope).transpose(1, 2)
        k = _rotate(k, rope).transpose(1, 2)
        v = block.proj["v"](y).unflatten(-1, (block.heads, -1)).transpose(1, 2)
        bias = state.attention_bias(block._freefuse_index, x)
        out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=bias)
        out = out.transpose(1, 2).flatten(2)
        if state.collect and block._freefuse_index == state.collect_block:
            self.collect_maps(state, q, k, out)
        x = x + block.proj["o"](out) * gate_a[:, None]
        y = block.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = block.proj["ffn_up_gate"](y.to(block.proj["q"].weight.dtype)).chunk(2, -1)
        from src.vendor.moss_image import _situ
        return x + block.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]

    @staticmethod
    def collect_maps(state, q, k, out):
        n = state.image_length
        img = out[:, :n].float()
        cross = {}
        positions = state.positions()
        if state.background_positions:
            positions = {**positions, "__background__": state.background_positions}
        for name, indices in positions.items():
            keys = k[:, :, [n + i for i in indices]].float()
            logits = torch.matmul(q[:, :, :n].float(), keys.transpose(-1, -2)) / state.cross_temperature
            cross[name] = logits.softmax(dim=-2).mean(dim=(1, 3))
        maps = {}
        for name, scores in cross.items():
            # Background does not compete with subject anchors.
            others = [value for key, value in cross.items() if key != name and key != "__background__"]
            contrast = scores * (1 + len(others)) - sum(others) if others else scores
            count = max(1, min(n, int(n * state.top_k_ratio)))
            indices = contrast.topk(count, dim=-1).indices
            core = img.gather(1, indices[..., None].expand(-1, -1, img.shape[-1]))
            similarity = (core @ img.transpose(1, 2)).mean(dim=1)
            maps[name] = (similarity / state.temperature).softmax(-1)[..., None]
        state.maps_by_sample[state.sample_index] = maps
