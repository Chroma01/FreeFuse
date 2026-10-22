"""Shared real-weight MOSS fixtures for ComfyUI comparison/parameter runs."""
import folder_paths


def load_moss_models():
    from nodes import UNETLoader, CLIPLoader, VAELoader
    def pick(category, filename):
        if filename not in folder_paths.get_filename_list(category):
            raise FileNotFoundError(f"Install {filename} in ComfyUI/models/{category}. See docs/moss_image.md.")
        return filename
    model, = UNETLoader().load_unet(pick("diffusion_models", "moss-image-sft-dit.safetensors"), "default")
    clip, = CLIPLoader().load_clip(pick("text_encoders", "moss-image-sft-qwen35.safetensors"), type="moss_image")
    vae, = VAELoader().load_vae(pick("vae", "moss-image-sft-vae.safetensors"))
    return model, clip, vae


def moss_lora_config():
    names = folder_paths.get_filename_list("loras")
    expected = ("Tinkerbell_MOSS_v1.safetensors", "AlbedoOverlord_MOSS_v1.safetensors")
    if any(name not in names for name in expected):
        raise FileNotFoundError("Install the two MOSS character LoRAs listed in docs/moss_image.md.")
    return (*expected, "tinkerbell", "albedo", 1.0)


def moss_prompt_config():
    concepts = {
        "tinkerbell": "t1nk3r, a blonde fairy with blue eyes, a green dress and translucent wings",
        "albedo": "lbd0c1tr0n, a black-haired woman with golden eyes, curved white horns, a white dress and black feathered wings",
    }
    prompt = ("An anime illustration of two adult characters in a sunny garden: "
              + concepts["tinkerbell"] + ", on the left; " + concepts["albedo"]
              + ", on the right. Both are smiling and holding flowers, full body.")
    return prompt, "", concepts, "a sunny garden", "tinkerbell", "albedo"
