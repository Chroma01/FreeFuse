#!/usr/bin/env python
"""Generate isolated, single-LoRA references for the existing comparison examples.

Run in the same environment as ComfyUI. Outputs include exact prompts, sampling
settings and model/LoRA paths. No FreeFuse masking or attention bias is enabled.
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

def reference_prompt(kind, name, description, background):
    if kind == "z_image" and name == "skeleton":
        # Franchise names can induce a title card instead of a scenic character reference.
        description = description.replace("Masters of the Universe character, ", "")
        return (
            "A picture of one character, " + description + ", " + background + ". "
            "The character stands in an uninterrupted landscape extending to every edge "
            "of the image. Full character illustration without lettering, captions or logos."
        )
    prefix = "Realistic photography, portrait of " if kind in ("flux", "sdxl") else "A picture of one character, "
    return prefix + description + ", " + background + "."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="flux,z_image,klein9b,klein4b,sdxl")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--character", choices=["harry", "daiyu", "jinx", "skeleton", "harry_klein", "shalnark_klein", "starfire", "albedo"], help="Generate only this adapter")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.size < 256 or args.size % 64:
        parser.error("--size must be at least 256 and divisible by 64")
    # ComfyUI reads argv during import; keep its runtime options separate.
    sys.argv = [sys.argv[0], "--disable-dynamic-vram"]
    import test_freefuse_compare as compare
    import torch
    import comfy.sd
    import comfy.utils
    import folder_paths
    from nodes import UNETLoader, CLIPLoader, DualCLIPLoader, VAELoader, CheckpointLoaderSimple
    from comfy_extras.nodes_flux import CLIPTextEncodeFlux

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"reference_type": "single LoRA generation, FreeFuse disabled", "runs": []}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    def load_base(kind):
        if kind == "anima":
            from anima_test_support import load_anima_models
            return (*load_anima_models(), {"diffusion_model": "anima-base-v1.0.safetensors", "text_encoders": ["qwen_3_06b_base.safetensors"], "vae": "qwen_image_vae.safetensors"})
        if kind == "sdxl":
            names = {"checkpoint": "sd_xl_base_1.0.safetensors"}
            return (*CheckpointLoaderSimple().load_checkpoint(names["checkpoint"]), names)
        unets = {"flux": "flux1-dev-fp8.safetensors",
                 "z_image": "z_image_turbo_bf16.safetensors",
                 "klein4b": "flux-2-klein-4b-fp8.safetensors",
                 "klein9b": "flux-2-klein-9b-fp8.safetensors"}
        unet = unets[kind]
        model, = UNETLoader().load_unet(unet, "default")
        if kind == "flux":
            encoders = ["clip_l.safetensors", "t5xxl_fp8_e4m3fn.safetensors"]
            clip, = DualCLIPLoader().load_clip(*encoders, "flux")
        else:
            encoders = ["qwen_3_8b_fp8mixed.safetensors" if kind == "klein9b" else "qwen_3_4b.safetensors"]
            clip, = CLIPLoader().load_clip(encoders[0], type="lumina2" if kind == "z_image" else "flux2")
        vae_name = "flux2-vae.safetensors" if kind.startswith("klein") else "ae.safetensors"
        vae, = VAELoader().load_vae(vae_name)
        return model, clip, vae, {"diffusion_model": unet, "text_encoders": encoders, "vae": vae_name}

    def encode(kind, clip, prompt, negative):
        if kind == "flux":
            positive, = CLIPTextEncodeFlux().encode(clip, prompt, prompt, 3.5)
            negative, = CLIPTextEncodeFlux().encode(clip, "", "", 3.5)
            return positive, negative
        if kind == "z_image":
            system = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts."
            prompt = f"{system} <Prompt Start> {prompt}"
        kwargs = {"llama_template": compare.KLEIN_NO_THINK_TEMPLATE} if kind.startswith("klein") else {}
        extra = {"guidance": None} if kind.startswith("klein") else {}
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, **kwargs), add_dict=extra)
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative, **kwargs), add_dict=extra)
        return positive, negative

    for kind in args.models.split(","):
        if kind not in {"anima", "flux", "z_image", "klein9b", "klein4b", "sdxl"}:
            raise ValueError(f"Unsupported model: {kind}")
        settings = compare.MODEL_DEFAULTS[kind]
        _, negative, concepts, background, a, b = compare._default_prompt_config(kind)
        lora_a, lora_b, _, _, strength = compare._find_loras_for_model(kind)
        for name, lora_file in [(a, lora_a), (b, lora_b)]:
            if args.character and name != args.character:
                continue
            prompt = reference_prompt(kind, name, concepts[name], background)
            target = out / kind / (name + ".png")
            existing = next((r for r in manifest["runs"] if r["model"] == kind and r["character"] == name), None)
            if target.exists() and existing is not None:
                if existing["width"] != args.size or existing["height"] != args.size or existing["prompt"] != prompt:
                    raise ValueError(f"Existing reference has different settings; choose a new output directory: {target}")
                print("EXISTS", target, flush=True)
                continue
            compare._reset_comfy_memory(kind + "/" + name)
            model, clip, vae, names = load_base(kind)
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_file)
            weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_single, clip_single = comfy.sd.load_lora_for_models(model, clip, weights, strength, strength)
            del weights
            conditioning, neg_conditioning = encode(kind, clip_single, prompt, negative)
            channels, h, w = compare._get_latent_shape(model_single, args.size, args.size)
            latent = torch.zeros([1, channels, h, w])
            if kind == "anima":
                latent = latent.unsqueeze(2)
            sigmas = compare._build_flux2_sigmas(settings["steps"], args.size, args.size) if kind.startswith("klein") else None
            noise = compare.comfy.sample.prepare_noise(latent, settings["seed"], None)
            with torch.inference_mode():
                samples = compare.comfy.sample.sample(
                    model_single, noise, settings["steps"], settings["cfg"],
                    settings["sampler"], settings["scheduler"], conditioning,
                    neg_conditioning, latent, denoise=1.0, sigmas=sigmas, seed=settings["seed"])
                image = compare.decode_to_pil(vae, samples)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            manifest["runs"].append(dict(model=kind, character=name, prompt=prompt,
                negative_prompt=negative, seed=settings["seed"], steps=settings["steps"],
                cfg=settings["cfg"], sampler=settings["sampler"], scheduler=settings["scheduler"],
                width=args.size, height=args.size, lora=lora_file, lora_sha256=hashlib.sha256(Path(lora_path).read_bytes()).hexdigest(), strength_model=strength,
                strength_clip=strength, model_files=names, output=str(target), freefuse=False))
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print("SAVED", target, flush=True)
            del model_single, clip_single, model, clip, vae, conditioning, neg_conditioning, samples
            gc.collect()
            compare._reset_comfy_memory("after " + name)

if __name__ == "__main__":
    main()
