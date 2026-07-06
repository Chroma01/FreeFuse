#!/usr/bin/env python3
"""Run a Krea2 baseline vs FreeFuse comparison inside a local ComfyUI checkout.

This script intentionally calls ComfyUI node classes directly instead of posting
to the server. It is meant for repeatable local or remote testing and prints
memory/progress checkpoints for long Krea2 runs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
from pathlib import Path
import sys
import time
from typing import Dict, Tuple


PROMPT = (
    "On the left stands kimpossible, Kim Possible, a teenage cartoon girl with long red-orange hair, "
    "green eyes, wearing a fitted green cropped shirt and black pants. On the right stands violetparr, "
    "Violet Parr, a superhero girl with short black hair with purple highlights, purple eyes, wearing a "
    "red and black superhero bodysuit and black gloves. They are standing on a cinematic city rooftop at "
    "sunset with glass buildings in the background, full body, two clearly separated characters, clean "
    "composition, no merged faces, no costume swapping."
)

NEGATIVE_PROMPT = (
    "low quality, blurry, merged characters, duplicate face, wrong costume, extra limbs, deformed hands"
)

KIM_CONCEPT = (
    "kimpossible, Kim Possible, a teenage cartoon girl with long red-orange hair, green eyes, "
    "wearing a fitted green cropped shirt and black pants"
)

VIOLET_CONCEPT = (
    "violetparr, Violet Parr, a superhero girl with short black hair with purple highlights, "
    "purple eyes, wearing a red and black superhero bodysuit and black gloves"
)

BACKGROUND_CONCEPT = "a cinematic city rooftop at sunset with glass buildings in the background"


def parse_args() -> argparse.Namespace:
    default_comfyui = Path.home() / ".cache" / "freefuse-comfyui-test" / "ComfyUI"
    plugin_dir = Path(__file__).resolve().parents[1]
    downloads = Path.home() / "Downloads"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-dir", type=Path, default=default_comfyui)
    parser.add_argument("--plugin-dir", type=Path, default=plugin_dir)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "FreeFuse" / "Krea2_compare")
    parser.add_argument("--mode", choices=("baseline", "freefuse", "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--phase1-steps", type=int, default=None)
    parser.add_argument("--collect-step", type=int, default=3)
    parser.add_argument("--collect-block", type=int, default=10)
    parser.add_argument("--collect-block-end", type=int, default=10)
    parser.add_argument("--seed", type=int, default=354347915735006)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--unet-name", default="krea2_turbo_fp8_scaled.safetensors")
    parser.add_argument("--clip-name", default="qwen3vl_4b_fp8_scaled.safetensors")
    parser.add_argument("--vae-name", default="qwen_image_vae.safetensors")
    parser.add_argument("--kim-lora-name", default="FreeFuse/Krea2/Krea 2 - Kim Possible.safetensors")
    parser.add_argument("--violet-lora-name", default="FreeFuse/Krea2/Krea 2 - Violet Parr.safetensors")
    parser.add_argument("--kim-lora-file", type=Path, default=downloads / "Krea 2 - Kim Possible.safetensors")
    parser.add_argument("--violet-lora-file", type=Path, default=downloads / "Krea 2 - Violet Parr.safetensors")
    parser.add_argument("--lora-strength", type=float, default=0.85)
    parser.add_argument("--bias-scale", type=float, default=4.0)
    parser.add_argument("--positive-bias-scale", type=float, default=2.0)
    parser.add_argument("--bias-blocks", choices=("all", "last_half", "none"), default="all")
    parser.add_argument("--balance-iterations", type=int, default=15)
    parser.add_argument("--max-rss-gb", type=float, default=42.0)
    parser.add_argument("--max-memory-percent", type=float, default=92.0)
    parser.add_argument("--no-decode", action="store_true", help="Skip VAE decode and PNG output.")
    return parser.parse_args()


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
            return
        if link_path.exists():
            return
        raise RuntimeError(f"Broken symlink exists: {link_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Required source file does not exist: {target_path}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] linking {link_path} -> {target_path}", flush=True)
    try:
        link_path.symlink_to(target_path)
    except FileExistsError:
        if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
            return
        raise


def ensure_test_layout(args: argparse.Namespace) -> None:
    if not args.comfyui_dir.exists():
        raise FileNotFoundError(f"ComfyUI directory does not exist: {args.comfyui_dir}")
    if not args.plugin_dir.exists():
        raise FileNotFoundError(f"FreeFuse plugin directory does not exist: {args.plugin_dir}")

    custom_node_link = args.comfyui_dir / "custom_nodes" / "freefuse_comfyui"
    if custom_node_link.exists() or custom_node_link.is_symlink():
        if custom_node_link.resolve() != args.plugin_dir.resolve():
            raise RuntimeError(
                f"{custom_node_link} points to {custom_node_link.resolve()}, "
                f"expected {args.plugin_dir.resolve()}"
            )
    else:
        custom_node_link.parent.mkdir(parents=True, exist_ok=True)
        print(f"[setup] linking {custom_node_link} -> {args.plugin_dir}", flush=True)
        try:
            custom_node_link.symlink_to(args.plugin_dir)
        except FileExistsError:
            if custom_node_link.is_symlink() and custom_node_link.resolve() == args.plugin_dir.resolve():
                pass
            else:
                raise

    lora_root = args.comfyui_dir / "models" / "loras"
    ensure_symlink(lora_root / args.kim_lora_name, args.kim_lora_file)
    ensure_symlink(lora_root / args.violet_lora_name, args.violet_lora_file)


def validate_device_model_pair(args: argparse.Namespace) -> None:
    fp8_names = [name for name in (args.unet_name, args.clip_name) if "fp8" in name.lower()]
    if args.device == "auto" and platform.system() == "Darwin" and fp8_names:
        raise RuntimeError(
            "ComfyUI auto device selection on macOS uses Apple MPS, which does not support "
            "the fp8 Krea2 weights used here "
            f"({', '.join(fp8_names)}). Use --device cpu for a local compatibility smoke test "
            "or run the formal image comparison on a CUDA GPU host."
        )


class Progress:
    def __init__(self, max_rss_gb: float, max_memory_percent: float):
        import psutil

        self.psutil = psutil
        self.process = psutil.Process(os.getpid())
        self.start = time.time()
        self.max_rss_gb = max_rss_gb
        self.max_memory_percent = max_memory_percent

    def report(self, stage: str) -> None:
        rss_gb = self.process.memory_info().rss / (1024 ** 3)
        memory = self.psutil.virtual_memory()
        print(
            f"[krea2-compare] {stage}: rss={rss_gb:.2f} GiB, "
            f"system_used={memory.percent:.1f}%, elapsed={time.time() - self.start:.1f}s",
            flush=True,
        )
        if rss_gb > self.max_rss_gb:
            raise MemoryError(f"RSS {rss_gb:.2f} GiB exceeded --max-rss-gb={self.max_rss_gb}")
        if memory.percent > self.max_memory_percent:
            raise MemoryError(
                f"System memory {memory.percent:.1f}% exceeded "
                f"--max-memory-percent={self.max_memory_percent}"
            )


def initialize_comfyui(args: argparse.Namespace):
    os.chdir(args.comfyui_dir)
    sys.path.insert(0, str(args.comfyui_dir))

    comfy_argv = [
        "run_krea2_compare.py",
        "--disable-auto-launch",
        "--preview-method",
        "none",
        "--cache-none",
        "--disable-api-nodes",
        "--log-stdout",
    ]
    if args.device == "cpu":
        comfy_argv.append("--cpu")
    sys.argv = comfy_argv

    import main
    import nodes

    main.apply_custom_paths()
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    return nodes


def save_tensor_images(images, output_dir: Path, prefix: str) -> Path:
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, image in enumerate(images):
        arr = image.detach().cpu().clamp(0, 1).numpy()
        arr = (arr * 255.0).round().astype(np.uint8)
        path = output_dir / f"{prefix}_{index:02d}.png"
        Image.fromarray(arr).save(path)
        saved_paths.append(path)
    return saved_paths[0]


def make_contact_sheet(paths: Dict[str, Path], output_dir: Path) -> Path | None:
    from PIL import Image, ImageDraw

    if "baseline" not in paths or "freefuse" not in paths:
        return None

    baseline = Image.open(paths["baseline"]).convert("RGB")
    freefuse = Image.open(paths["freefuse"]).convert("RGB")
    width = baseline.width + freefuse.width
    label_h = 34
    sheet = Image.new("RGB", (width, baseline.height + label_h), "white")
    sheet.paste(baseline, (0, label_h))
    sheet.paste(freefuse, (baseline.width, label_h))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "Baseline", fill=(0, 0, 0))
    draw.text((baseline.width + 12, 10), "FreeFuse", fill=(0, 0, 0))
    path = output_dir / "krea2_baseline_vs_freefuse.png"
    sheet.save(path)
    return path


def load_base_graph(nodes, args: argparse.Namespace, progress: Progress):
    N = nodes.NODE_CLASS_MAPPINGS
    model = N["UNETLoader"]().load_unet(args.unet_name, "default")[0]
    progress.report("UNET loaded")
    clip_device = "cpu" if args.device == "cpu" else "default"
    clip = N["CLIPLoader"]().load_clip(args.clip_name, "krea2", clip_device)[0]
    progress.report("CLIP loaded")
    vae = None
    if not args.no_decode:
        vae = N["VAELoader"]().load_vae(args.vae_name)[0]
        progress.report("VAE loaded")

    model, clip, data = N["FreeFuseLoRALoader"]().load_lora(
        model,
        clip,
        args.kim_lora_name,
        "kimpossible",
        args.lora_strength,
        args.lora_strength,
    )
    model, clip, data = N["FreeFuseLoRALoader"]().load_lora(
        model,
        clip,
        args.violet_lora_name,
        "violetparr",
        args.lora_strength,
        args.lora_strength,
        data,
    )
    progress.report("Krea2 LoRAs loaded")

    data = N["FreeFuseConceptMap"]().create_map(
        "kimpossible",
        KIM_CONCEPT,
        "violetparr",
        VIOLET_CONCEPT,
        "",
        "",
        "",
        "",
        True,
        BACKGROUND_CONCEPT,
        data,
    )[0]
    data = N["FreeFuseTokenPositions"]().compute_positions(clip, PROMPT, data, True, True)[0]
    progress.report("Token positions computed")

    positive = N["CLIPTextEncode"]().encode(clip, PROMPT)[0]
    negative = N["CLIPTextEncode"]().encode(clip, NEGATIVE_PROMPT)[0]
    latent = N["EmptySD3LatentImage"].execute(args.width, args.height, 1)[0]
    progress.report("Conditioning and latent ready")
    return N, model, clip, vae, data, positive, negative, latent


def decode_and_save(N, vae, samples, output_dir: Path, prefix: str) -> Path:
    images = N["VAEDecode"]().decode(vae, samples)[0]
    return save_tensor_images(images, output_dir, prefix)


def run_baseline(N, model, vae, positive, negative, latent, args, progress: Progress) -> Tuple[object, Path | None]:
    progress.report("Baseline sampling start")
    samples = N["KSampler"]().sample(
        model,
        args.seed,
        args.steps,
        args.cfg,
        args.sampler,
        args.scheduler,
        positive,
        negative,
        latent,
        1.0,
    )[0]
    progress.report("Baseline sampling done")
    if args.no_decode:
        return samples, None
    path = decode_and_save(N, vae, samples, args.output_dir, "krea2_baseline")
    progress.report(f"Baseline saved to {path}")
    return samples, path


def run_freefuse(N, model, vae, data, positive, negative, latent, args, progress: Progress) -> Tuple[object, Path | None]:
    phase1_steps = args.phase1_steps or args.steps
    progress.report("FreeFuse Phase 1 start")
    model_after_phase1, masks, _preview = N["FreeFusePhase1Sampler"]().collect_masks(
        model=model,
        conditioning=positive,
        neg_conditioning=negative,
        latent=latent,
        freefuse_data=data,
        seed=args.seed,
        steps=phase1_steps,
        collect_step=args.collect_step,
        cfg=args.cfg,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
        collect_block=args.collect_block,
        collect_block_end=args.collect_block_end,
        temperature=0.0,
        top_k_ratio=0.1,
        disable_lora_phase1=True,
        bg_scale=0.95,
        use_morphological_cleaning=True,
        balance_iterations=args.balance_iterations,
        balance_lr=0.01,
        gravity_weight=0.00004,
        spatial_weight=0.00004,
        momentum=0.2,
        centroid_margin=0.0,
        border_penalty=0.0,
        anisotropy=1.3,
    )
    progress.report("FreeFuse Phase 1 done")
    masked_model = N["FreeFuseMaskApplicator"]().apply_masks(
        model_after_phase1,
        masks,
        data,
        enable_token_masking=True,
        latent=latent,
        enable_attention_bias=(args.bias_blocks != "none"),
        bias_scale=args.bias_scale,
        positive_bias_scale=args.positive_bias_scale,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks=args.bias_blocks,
    )[0]
    progress.report("FreeFuse masks applied")
    samples = N["KSampler"]().sample(
        masked_model,
        args.seed,
        args.steps,
        args.cfg,
        args.sampler,
        args.scheduler,
        positive,
        negative,
        latent,
        1.0,
    )[0]
    progress.report("FreeFuse Phase 2 sampling done")
    if args.no_decode:
        return samples, None
    path = decode_and_save(N, vae, samples, args.output_dir, "krea2_freefuse")
    progress.report(f"FreeFuse saved to {path}")
    return samples, path


def main() -> None:
    args = parse_args()
    if args.width % 16 != 0 or args.height % 16 != 0:
        raise ValueError("Krea2 test dimensions must be multiples of 16")
    args.comfyui_dir = args.comfyui_dir.resolve()
    args.plugin_dir = args.plugin_dir.resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (args.comfyui_dir / args.output_dir).resolve()

    validate_device_model_pair(args)
    ensure_test_layout(args)
    progress = Progress(args.max_rss_gb, args.max_memory_percent)
    nodes = initialize_comfyui(args)
    progress.report("ComfyUI nodes initialized")
    graph = load_base_graph(nodes, args, progress)
    N, model, _clip, vae, data, positive, negative, latent = graph

    paths: Dict[str, Path] = {}
    if args.mode in ("baseline", "both"):
        _samples, path = run_baseline(N, model, vae, positive, negative, latent, args, progress)
        if path is not None:
            paths["baseline"] = path
    if args.mode in ("freefuse", "both"):
        _samples, path = run_freefuse(N, model, vae, data, positive, negative, latent, args, progress)
        if path is not None:
            paths["freefuse"] = path

    contact = make_contact_sheet(paths, args.output_dir)
    if contact is not None:
        progress.report(f"Comparison saved to {contact}")
    print("[krea2-compare] PASS", flush=True)


if __name__ == "__main__":
    main()
