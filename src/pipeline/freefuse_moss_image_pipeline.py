"""Two-pass FreeFuse for MOSS-Image Base/SFT using the release sampler."""
import json
import math
import re
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from src.models.freefuse_transformer_moss_image import FreeFuseMossImageTransformer
from src.tuner.moss_image_lora import load_moss_lora
from src.utils.moss_image_masks import process_moss_maps
from src.vendor.moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer, _validate_skip_layers


def find_moss_concept_positions(tokenizer, prompt, concepts, max_sequence_length=4096):
    """Find literal concept occurrences in exactly the template used by encode_prompt."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    start = formatted.find(prompt)
    if not prompt or start < 0:
        raise ValueError("A nonempty prompt must be preserved by the MOSS chat template.")
    encoded = tokenizer(formatted, return_offsets_mapping=True, truncation=True,
                        max_length=max_sequence_length)
    result = {}
    for name, concept in concepts.items():
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError(f"Concept {name!r} must be nonempty.")
        spans = [(start + m.start(), start + m.end()) for m in re.finditer(re.escape(concept), prompt)]
        offsets = encoded["offset_mapping"]
        spans = [(begin, end) for begin, end in spans
                 if any(a <= begin < b for a, b in offsets) and any(a < end <= b for a, b in offsets)]
        positions = [i for i, (a, b) in enumerate(offsets)
                     if b > a and any(a < end and b > begin for begin, end in spans)]
        if not positions:
            raise ValueError(f"Concept {concept!r} is absent from the prompt or was truncated.")
        result[name] = [positions]
    return result


class FreeFuseMossImagePipeline(MOSSImagePipeline):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, torch_dtype=torch.bfloat16,
                        local_files_only=False, revision=None):
        source = Path(pretrained_model_name_or_path).expanduser()
        if not source.is_dir():
            source = Path(snapshot_download(
                str(pretrained_model_name_or_path), revision=revision,
                local_files_only=local_files_only,
                allow_patterns=["transformer/*", "text_encoder/*", "tokenizer/*", "vae/*", "scheduler/*"]))
        dtype = torch_dtype.get("default", torch.bfloat16) if isinstance(torch_dtype, dict) else torch_dtype
        vae_dtype = torch_dtype.get("vae", torch.float32) if isinstance(torch_dtype, dict) else torch.float32
        transformer = MOSSImageTransformer.from_pretrained(source / "transformer", torch_dtype=dtype)
        transformer = FreeFuseMossImageTransformer.init_from_transformer(transformer)
        scheduler = MOSSImageScheduler.from_pretrained(source / "scheduler")
        tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer")
        text_encoder = AutoModel.from_pretrained(source / "text_encoder", dtype=dtype).eval()
        vae = AutoencoderKLFlux2.from_pretrained(source / "vae", torch_dtype=vae_dtype).eval()
        return cls(transformer, scheduler, text_encoder, tokenizer, vae)

    def load_lora_weights(self, source, *, adapter_name, weight=1.0):
        return load_moss_lora(self.transformer, source, adapter_name, weight)

    @torch.no_grad()
    def __call__(self, prompt=None, *, concepts=None, freefuse_token_pos_maps=None,
                 use_freefuse=True, collect_block=20, collect_step=4, top_k_ratio=0.1,
                 temperature=4000.0, cross_temperature=1000.0, background_concept=None,
                 attention_bias_scale=4.0, attention_bias_positive_scale=2.0,
                 attention_bias_blocks=None, debug_save_path=None, bg_scale=0.95,
                 use_morphological_cleaning=True, **kwargs):
        self.transformer = FreeFuseMossImageTransformer.init_from_transformer(self.transformer)
        state = self.transformer._freefuse_state
        self.transformer.clear_freefuse_state()
        self.last_freefuse_debug = {}
        try:
            if not use_freefuse:
                return super().__call__(prompt, **kwargs)
            if not 0 <= collect_block < len(self.transformer.layers):
                raise ValueError("collect_block is outside the MOSS transformer.")
            steps = kwargs.get("num_inference_steps", 50)
            if type(collect_step) is not int or not 0 <= collect_step < steps:
                raise ValueError("collect_step is a zero-based index smaller than num_inference_steps.")
            if not 0 < top_k_ratio <= 1 or not all(math.isfinite(v) and v > 0 for v in (temperature, cross_temperature)):
                raise ValueError("Require 0 < top_k_ratio <= 1 and positive finite temperatures.")
            if not math.isfinite(bg_scale) or bg_scale <= 0:
                raise ValueError("bg_scale must be positive and finite.")
            if not all(math.isfinite(v) and v >= 0 for v in (attention_bias_scale, attention_bias_positive_scale)):
                raise ValueError("Attention bias scales must be finite and nonnegative.")
            if kwargs.get("num_images_per_prompt", 1) != 1 or (prompt is not None and not isinstance(prompt, str)):
                raise ValueError("MOSS FreeFuse currently supports one prompt and one image per call.")
            max_length = kwargs.get("max_sequence_length", 4096)
            if not 1 <= max_length <= 4096:
                raise ValueError("max_sequence_length must be in [1, 4096].")
            if freefuse_token_pos_maps is None:
                if not concepts or prompt is None:
                    raise ValueError("Provide a prompt with concepts, or explicit token positions with embeddings.")
                freefuse_token_pos_maps = find_moss_concept_positions(self.tokenizer, prompt, concepts, max_length)
            if not freefuse_token_pos_maps or any(len(rows) != 1 for rows in freefuse_token_pos_maps.values()):
                raise ValueError("Each concept must have one nonempty token-position row.")
            state.token_positions = freefuse_token_pos_maps
            state.collect_block, state.top_k_ratio = collect_block, top_k_ratio
            state.temperature, state.cross_temperature = temperature, cross_temperature
            state.bias_scale, state.positive_bias_scale = attention_bias_scale, attention_bias_positive_scale
            if attention_bias_blocks is not None:
                state.bias_blocks = _validate_skip_layers(attention_bias_blocks, len(self.transformer.layers), "attention_bias_blocks")
            else:
                state.bias_blocks = None
            if background_concept is not None:
                if prompt is None:
                    raise ValueError("background_concept requires a string prompt.")
                state.background_positions = find_moss_concept_positions(
                    self.tokenizer, prompt, {"background": background_concept}, max_length)["background"][0]
            prompt_embeds = kwargs.pop("prompt_embeds", None)
            prompt_mask = kwargs.pop("prompt_attention_mask", None)
            if (prompt is None) == (prompt_embeds is None):
                raise ValueError("Provide exactly one of prompt and prompt_embeds.")
            if prompt_embeds is None:
                prompt_embeds, prompt_mask = self.encode_prompt(prompt, max_length)
            if prompt_embeds.shape[0] != 1:
                raise ValueError("MOSS FreeFuse currently supports batch size one.")
            if prompt_mask is None:
                prompt_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.bool, device=prompt_embeds.device)
            prompt_embeds = prompt_embeds.to(self._execution_device)
            prompt_mask = prompt_mask.to(self._execution_device)
            # Validate positions before any denoising.
            state.text_length = int(prompt_mask.sum().item())
            state.positions()
            height, width = kwargs.get("height", 1024), kwargs.get("width", 1024)
            if height <= 0 or width <= 0 or height % 16 or width % 16:
                raise ValueError("height and width must be positive multiples of 16.")
            shape = (1, self.transformer.config.latent_channels, height // 16, width // 16)
            latents = kwargs.pop("latents", None)
            if latents is None:
                latents = randn_tensor(shape, generator=kwargs.get("generator"),
                                       device=self._execution_device, dtype=torch.float32)
            elif tuple(latents.shape) != shape:
                raise ValueError(f"latents must have shape {shape}.")
            latents = latents.to(device=self._execution_device, dtype=torch.float32)
            # Collection uses the base model with no LoRA, as in other FreeFuse backends.
            state.loras_enabled = False
            phase1 = latents.clone()
            self.scheduler.set_timesteps(steps, device=latents.device)
            if collect_step >= len(self.scheduler.timesteps):
                raise ValueError("collect_step exceeds the scheduler's deduplicated timestep count.")
            for index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps[:collect_step + 1])):
                state.collect = index == collect_step
                prediction = self.transformer(phase1, timestep, prompt_embeds, prompt_mask).sample.float()
                phase1 = self.scheduler.step(prediction, timestep, phase1, return_dict=False)[0]
            maps = state.maps_by_sample.get(0)
            if maps is None or any(name not in maps for name in state.token_positions):
                raise RuntimeError("The selected block did not collect every concept.")
            state.masks = process_moss_maps(maps, height // 16, width // 16,
                                            bg_scale=bg_scale, clean=use_morphological_cleaning)
            state.collect = False
            state.routing = state.loras_enabled = True
            self.last_freefuse_debug = {
                "similarity_maps": {k: v.detach().cpu() for k, v in maps.items()},
                "masks": {k: v.detach().cpu() for k, v in state.masks.items()},
                "token_positions": state.token_positions, "collect_block": collect_block,
                "collect_step": collect_step, "height": height, "width": width,
            }
            if debug_save_path is not None:
                self._save_debug(debug_save_path)
            return super().__call__(
                prompt=None, prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_mask,
                latents=latents.clone(), **kwargs)
        finally:
            self.transformer.clear_freefuse_state()

    def _predict(self, latents, timestep, embeds, mask, *, conditional=True, **kwargs):
        self.transformer._freefuse_state.conditional = conditional
        return super()._predict(latents, timestep, embeds, mask, conditional=conditional, **kwargs)

    def _save_debug(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        debug = self.last_freefuse_debug
        torch.save(debug, path / "routing.pt")
        metadata = {key: value for key, value in debug.items() if key not in {"similarity_maps", "masks"}}
        (path / "routing.json").write_text(json.dumps(metadata, indent=2))
        h, w = debug["height"] // 16, debug["width"] // 16
        for index, (name, mask) in enumerate(debug["masks"].items()):
            image = (mask[0, :, 0].reshape(h, w).numpy() * 255).astype("uint8")
            Image.fromarray(image).resize((debug["width"], debug["height"]), Image.Resampling.NEAREST).save(path / f"mask_{index}.png")
