"""Strict MOSS LoRA loading with per-adapter spatial routing.

Accepts PEFT A/B and Kohya/Comfy down/up linear LoRAs. Unsupported tensors
raise before installing any weights, rather than silently losing identity.
"""
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


class MossFreeFuseLinear(nn.Module):
    def __init__(self, base_layer, state, scope):
        super().__init__()
        self.base_layer = base_layer
        self.adapters = nn.ModuleDict()
        self.scales = {}
        self.state = state
        self.scope = scope

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        result = self.base_layer(x)
        if self.state.loras_enabled:
            for name, adapter in self.adapters.items():
                delta = adapter(x.to(adapter[0].weight.dtype)) * self.scales[name]
                mask = self.state.lora_mask(name, self.scope, x)
                if mask is not None:
                    delta = delta * mask
                result = result + delta.to(result.dtype)
        return result


def _target_aliases(model):
    aliases = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, MossFreeFuseLinear)) and ".adapters." not in name and not name.endswith(".base_layer"):
            aliases[name] = name
            aliases[name.replace(".", "_")] = name
    # Training tools may export the three attention projections as one QKV pair.
    for i in range(len(model.layers)):
        for spelling in (f"layers.{i}.proj.qkv", f"layers_{i}_proj_qkv"):
            aliases[spelling] = [f"layers.{i}.proj.{key}" for key in ("q", "k", "v")]
    return aliases


def load_moss_lora(model, source, adapter_name, weight=1.0):
    if not adapter_name or "." in adapter_name or adapter_name.startswith("__"):
        raise ValueError("adapter_name must be nonempty, contain no dots, and not start with '__'.")
    if not math.isfinite(weight):
        raise ValueError("LoRA weight must be finite.")
    if any(isinstance(m, MossFreeFuseLinear) and adapter_name in m.adapters for m in model.modules()):
        raise ValueError(f"Adapter {adapter_name!r} is already loaded.")
    tensors = load_file(str(Path(source)), device="cpu") if not isinstance(source, dict) else dict(source)
    aliases = _target_aliases(model)
    pairs, alphas = {}, {}
    for key, value in tensors.items():
        original = key
        for prefix in ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "lora_unet_", "lora_transformer_"):
            if key.startswith(prefix):
                key = key[len(prefix):]
        match = re.fullmatch(r"(.+)\.(lora_A|lora_B)(?:\.default)?\.weight", key)
        if match:
            target, side = match.groups()
            side = "down" if side == "lora_A" else "up"
        else:
            match = re.fullmatch(r"(.+)\.lora_(down|up)\.weight", key)
            if match:
                target, side = match.groups()
            elif key.endswith(".alpha"):
                alphas[key[:-6]] = value
                continue
            else:
                raise ValueError(f"Unsupported MOSS LoRA tensor: {original}")
        if target not in aliases:
            raise ValueError(f"LoRA target {target!r} is absent from this MOSS transformer.")
        if side in pairs.setdefault(target, {}):
            raise ValueError(f"Duplicate {side} tensor for {target}.")
        pairs[target][side] = value
    if not pairs or set(alphas) - set(pairs):
        raise ValueError("LoRA contains no complete pairs or has unmatched alpha values.")
    installs = []
    for target, pair in pairs.items():
        if set(pair) != {"down", "up"}:
            raise ValueError(f"Incomplete LoRA pair: {target}")
        down, up = pair["down"], pair["up"]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] == 0 or up.shape[1] != down.shape[0]:
            raise ValueError(f"Invalid linear LoRA shapes: {target}")
        rank = down.shape[0]
        alpha = float(alphas.get(target, rank))
        if not math.isfinite(alpha):
            raise ValueError(f"Invalid alpha: {target}")
        targets = aliases[target]
        targets = targets if isinstance(targets, list) else [targets]
        if len(targets) > 1 and up.shape[0] % len(targets):
            raise ValueError(f"Invalid fused QKV LoRA: {target}")
        chunks = up.chunk(len(targets), 0)
        for name, chunk in zip(targets, chunks):
            module = model.get_submodule(name)
            if tuple(down.shape) != (rank, module.weight.shape[1]) or tuple(chunk.shape) != (module.weight.shape[0], rank):
                raise ValueError(f"LoRA dimensions do not match {name}.")
            if any(item[0] == name for item in installs):
                raise ValueError(f"Overlapping fused/split targets: {name}")
            installs.append((name, down, chunk, weight * alpha / rank))
    for name, down, up, scale in installs:
        module = model.get_submodule(name)
        if not isinstance(module, MossFreeFuseLinear):
            scope = "joint" if name.startswith("layers.") else "text" if name.startswith("text_embedder.") else "image" if name == "video_embedder" else "none"
            wrapped = MossFreeFuseLinear(module, model._freefuse_state, scope)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, wrapped)
            module = wrapped
        adapter = nn.Sequential(nn.Linear(down.shape[1], down.shape[0], bias=False),
                                nn.Linear(up.shape[1], up.shape[0], bias=False)).to(module.weight)
        with torch.no_grad():
            adapter[0].weight.copy_(down)
            adapter[1].weight.copy_(up)
        module.adapters[adapter_name] = adapter
        module.scales[adapter_name] = scale
    return len(installs)
