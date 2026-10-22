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
    concepts = {'tinkerbell': 't1nk3r, Tinkerbell, a blonde adult fairy with blue eyes, hair in a high bun, a short green dress and translucent fairy wings, in classic Disney animation style', 'albedo': 'lbd0c1tr0n, Albedo from Overlord, a black-haired adult woman with golden eyes, curved white horns, a long white dress with gold ornaments and black feathered wings, in Japanese anime style'}
    prompt = 'A close portrait of two adult characters standing shoulder to shoulder in a sunny garden with flowers: t1nk3r, Tinkerbell, a blonde adult fairy with blue eyes, hair in a high bun, a short green dress and translucent fairy wings, in classic Disney animation style on the left, and lbd0c1tr0n, Albedo from Overlord, a black-haired adult woman with golden eyes, curved white horns, a long white dress with gold ornaments and black feathered wings, in Japanese anime style on the right. Both look at the viewer with a gentle smile, upper bodies, detailed faces.'
    return prompt, "", concepts, 'a sunny garden with flowers', "tinkerbell", "albedo"
