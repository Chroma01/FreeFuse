"""Native Anima fixtures; preserve the T5 adapter metadata in conditioning."""
import folder_paths

def load_anima_models():
    from nodes import UNETLoader, CLIPLoader, VAELoader
    def pick(category, name):
        if name not in folder_paths.get_filename_list(category):
            raise FileNotFoundError(f"Install {name} in ComfyUI/models/{category}. See docs/anima.md.")
        return name
    model, = UNETLoader().load_unet(pick("diffusion_models", "anima-base-v1.0.safetensors"), "default")
    clip, = CLIPLoader().load_clip(pick("text_encoders", "qwen_3_06b_base.safetensors"), type="stable_diffusion")
    vae, = VAELoader().load_vae(pick("vae", "qwen_image_vae.safetensors"))
    return model, clip, vae

def anima_lora_config():
    names = ("StarfireTT_ANIMAv1_v3.safetensors", "AlbedoOverlord_ANIMAv1_v1.safetensors")
    if any(name not in folder_paths.get_filename_list("loras") for name in names):
        raise FileNotFoundError("Install the Anima character LoRAs listed in docs/anima.md.")
    return (*names, "starfire", "albedo", 1.0)

def anima_prompt_config():
    concepts = {
        "starfire": "strfr, Starfire, green eyes, green sclera, long red hair, purple crop top, purple skirt, silver gorget, cartoon style",
        "albedo": "lbd0c1tr0n, Albedo, yellow eyes, slit pupils, long black hair, white horns, white dress, gold ornaments, black feathered wings, anime style",
    }
    background = "a sunny garden with flowers"
    prompt = "masterpiece, best quality, 2girls, upper body, " + concepts["starfire"] + " on the left, and " + concepts["albedo"] + " on the right, standing shoulder to shoulder in " + background + ", looking at viewer, smiling"
    return prompt, "worst quality, low quality, bad anatomy, text, watermark", concepts, background, "starfire", "albedo"
