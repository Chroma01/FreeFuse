"""Runtime controls for the upstream MOSS transformer, preserving checkpoint keys."""
from dataclasses import dataclass, field
from types import MethodType

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from src.attn_processor.freefuse_moss_image_attn_processor import FreeFuseMossImageAttnProcessor
from src.vendor.moss_image import MOSSImageTransformer


@dataclass
class MossFreeFuseState:
    token_positions: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    maps_by_sample: dict = field(default_factory=dict)
    background_positions: list = field(default_factory=list)
    routing: bool = False
    collect: bool = False
    loras_enabled: bool = True
    conditional: bool = True
    sample_index: int = 0
    image_length: int = 0
    text_length: int = 0
    collect_block: int = 20
    top_k_ratio: float = 0.1
    temperature: float = 4000.0
    cross_temperature: float = 1000.0
    bias_scale: float = 4.0
    positive_bias_scale: float = 2.0
    bias_blocks: object = None
    bias_cache: dict = field(default_factory=dict)

    def positions(self):
        result = {}
        if self.conditional:
            for name, rows in self.token_positions.items():
                row = rows[self.sample_index]
                if not row or min(row) < 0 or max(row) >= self.text_length:
                    raise ValueError(f"Concept {name!r} has invalid positions for {self.text_length} text tokens.")
                result[name] = row
        return result

    def attention_bias(self, index, x):
        if not self.routing or not self.conditional or not self.bias_scale:
            return None
        if self.bias_blocks is not None and index not in self.bias_blocks:
            return None
        cache_key = (self.sample_index, x.shape[1], x.device, x.dtype)
        if cache_key in self.bias_cache:
            return self.bias_cache[cache_key]
        n, size = self.image_length, x.shape[1]
        bias = x.new_zeros(1, 1, size, size)
        for name, positions in self.positions().items():
            mask = self.masks[name][self.sample_index, :, 0].to(x)
            values = mask * self.positive_bias_scale - (1 - mask) * self.bias_scale
            for pos in positions:
                bias[0, 0, :n, n + pos] = values
                bias[0, 0, n + pos, :n] = values
        self.bias_cache[cache_key] = bias
        return bias

    def lora_mask(self, name, scope, x):
        if not self.routing or x.ndim != 3:
            return None
        if scope not in {"joint", "text", "image"}:
            return None
        mask = x.new_ones(1, x.shape[1], 1)
        if scope in {"joint", "image"} and name in self.masks:
            spatial = self.masks[name][self.sample_index:self.sample_index + 1].to(x)
            mask[:, :self.image_length] = spatial
        if scope in {"joint", "text"} and self.conditional:
            start = self.image_length if scope == "joint" else 0
            for other, positions in self.positions().items():
                if other != name:
                    mask[:, [start + i for i in positions]] = 0
        return mask


def _block_forward(block, x, rope, modulation):
    state = block._freefuse_state
    if not state.routing and not state.collect:
        return block._freefuse_original_forward(x, rope, modulation)
    return block._freefuse_processor(block, x, rope, modulation)


class FreeFuseMossImageTransformer(MOSSImageTransformer):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, num_layers=num_layers,
                         text_dim=text_dim, freq_dim=freq_dim, latent_channels=latent_channels,
                         rope_base=rope_base)
        self._install_freefuse()

    def _install_freefuse(self):
        self._freefuse_state = MossFreeFuseState()
        for index, block in enumerate(self.layers):
            block._freefuse_original_forward = block.forward
            block._freefuse_state = self._freefuse_state
            block._freefuse_index = index
            block._freefuse_processor = FreeFuseMossImageAttnProcessor()
            block.forward = MethodType(_block_forward, block)

    @classmethod
    def init_from_transformer(cls, model):
        if not isinstance(model, MOSSImageTransformer):
            raise TypeError("Expected the bundled MOSSImageTransformer.")
        if isinstance(model, cls):
            return model
        model.__class__ = cls
        model._install_freefuse()
        return model

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        state = self._freefuse_state
        outputs = []
        batch = hidden_states.shape[0]
        times = torch.as_tensor(timestep, device=hidden_states.device).reshape(-1).expand(batch)
        for index in range(batch):
            state.sample_index = index
            state.image_length = hidden_states.shape[-2] * hidden_states.shape[-1]
            mask = None if encoder_attention_mask is None else encoder_attention_mask[index:index + 1]
            state.text_length = encoder_hidden_states.shape[1] if mask is None else int(mask.sum().item())
            sample = super().forward(hidden_states[index:index + 1], times[index:index + 1],
                                     encoder_hidden_states[index:index + 1], mask,
                                     return_dict=False, skip_layers=skip_layers)[0]
            outputs.append(sample)
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)

    def clear_freefuse_state(self):
        state = self._freefuse_state
        state.routing = state.collect = False
        state.loras_enabled = True
        state.token_positions = {}
        state.masks = {}
        state.maps_by_sample = {}
        state.background_positions = []
        state.bias_cache = {}
        state.conditional = True
