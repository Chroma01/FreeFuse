"""Run MOSS-Image FreeFuse, optionally saving a same-noise baseline."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.pipeline.freefuse_moss_image_pipeline import FreeFuseMossImagePipeline


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Image-SFT")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--subject", action="append", nargs=3, metavar=("NAME", "LORA_PATH", "CONCEPT"),
                        required=True, help="Repeat for each subject. CONCEPT must occur literally in the prompt.")
    parser.add_argument("--background-concept")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-block", type=int, default=20)
    parser.add_argument("--collect-step", type=int, default=4)
    parser.add_argument("--bias-scale", type=float, default=4.0)
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output-dir", default="outputs/moss_image")
    return parser


def main():
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipe = FreeFuseMossImagePipeline.from_pretrained(args.model)
    for name, path, concept in args.subject:
        pipe.load_lora_weights(path, adapter_name=name, weight=args.lora_weight)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    concepts = {name: concept for name, path, concept in args.subject}
    options = dict(height=args.height, width=args.width, num_inference_steps=args.steps,
                   guidance_scale=args.guidance)
    def noise():
        return torch.Generator("cuda").manual_seed(args.seed)
    baseline = None
    if args.compare:
        baseline = pipe(args.prompt, use_freefuse=False, generator=noise(), **options).images[0]
        baseline.save(output / "baseline.png")
    image = pipe(args.prompt, concepts=concepts, generator=noise(),
                 background_concept=args.background_concept, collect_block=args.collect_block,
                 collect_step=args.collect_step, attention_bias_scale=args.bias_scale,
                 debug_save_path=output / "routing", **options).images[0]
    image.save(output / "freefuse.png")
    if baseline is not None:
        compare = Image.new("RGB", (args.width * 2, args.height))
        compare.paste(baseline, (0, 0))
        compare.paste(image, (args.width, 0))
        compare.save(output / "compare.png")
    (output / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
