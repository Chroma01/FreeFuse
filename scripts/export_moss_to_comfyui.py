"""Package an official MOSS Diffusers bundle for native ComfyUI loaders."""
import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def load_shards(directory, index_name):
    index = json.loads((directory / index_name).read_text())["weight_map"]
    state = {}
    for name in sorted(set(index.values())):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Shard path escapes the component directory.")
        state.update(load_file(path))
    if set(state) != set(index):
        raise ValueError("Shard tensors do not match the component index.")
    return state


def normalize_text_keys(state):
    # AutoModel.save_pretrained writes language_model.* / visual.* whereas
    # Comfy's Qwen3.5 detector expects model.language_model.* / model.visual.*.
    result = {}
    for key, value in state.items():
        name = "model." + key if key.startswith(("language_model.", "visual.")) else key
        if name in result:
            raise ValueError(f"Duplicate normalized text tensor: {name}")
        result[name] = value
    if "model.language_model.layers.0.linear_attn.A_log" not in result:
        raise ValueError("Expected the official Qwen3.5-9B text encoder.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("--name", default="moss-image-sft")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a filename prefix.")
    config = json.loads((args.source / "transformer/config.json").read_text())
    if config.get("_class_name") != "MOSSImageTransformer" or config.get("latent_channels") != 128:
        raise ValueError("Expected an official MOSS Image Base/SFT bundle.")
    outputs = {kind: args.models / folder / (args.name + suffix) for kind, folder, suffix in [
        ("transformer", "diffusion_models", "-dit.safetensors"),
        ("text_encoder", "text_encoders", "-qwen35.safetensors"),
        ("vae", "vae", "-vae.safetensors")]}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("One or more output files already exist; choose another --name.")
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for component, index in [
        ("transformer", "diffusion_pytorch_model.safetensors.index.json"),
        ("text_encoder", "model.safetensors.index.json")]:
        state = load_shards(args.source / component, index)
        if component == "text_encoder":
            state = normalize_text_keys(state)
        path = outputs[component]
        temporary = path.with_suffix(".partial")
        save_file(state, temporary, metadata={"format": "pt", "model": "moss-image"})
        temporary.rename(path)
        del state
    path = outputs["vae"]
    temporary = path.with_suffix(".partial")
    shutil.copyfile(args.source / "vae/diffusion_pytorch_model.safetensors", temporary)
    temporary.rename(path)


if __name__ == "__main__":
    main()
