"""Compare MOSS collection blocks with fixed noise and the same loaded model."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw

from main_freefuse_moss_image import build_parser
from src.pipeline.freefuse_moss_image_pipeline import FreeFuseMossImagePipeline


def main():
    parser = build_parser()
    parser.add_argument("--blocks", default="8,14,20,26,32,38")
    args = parser.parse_args()
    blocks = [int(v) for v in args.blocks.split(",")]
    pipe = FreeFuseMossImagePipeline.from_pretrained(args.model)
    for name, path, concept in args.subject:
        pipe.load_lora_weights(path, adapter_name=name, weight=args.lora_weight)
    pipe.enable_model_cpu_offload() if args.offload else pipe.to("cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    options = dict(height=args.height, width=args.width, num_inference_steps=args.steps,
                   guidance_scale=args.guidance)
    panels = []
    panel_height = 40 + round(args.height * min(512 / args.width, 512 / args.height))
    columns = min(4, 1 + len(blocks))
    rows = []
    for block in [None, *blocks]:
        directory = output / ("baseline" if block is None else f"block_{block}")
        directory.mkdir(exist_ok=True)
        extra = dict(use_freefuse=False) if block is None else dict(
            concepts={name: concept for name, path, concept in args.subject},
            collect_block=block, collect_step=args.collect_step, attention_bias_scale=args.bias_scale,
            background_concept=args.background_concept, debug_save_path=directory / "routing")
        image = pipe(args.prompt, generator=torch.Generator("cuda").manual_seed(args.seed),
                     **options, **extra).images[0]
        image.save(directory / "result.png")
        preview = image.copy()
        preview.thumbnail((512, 512))
        panel = Image.new("RGB", (512, panel_height), "white")
        panel.paste(preview, ((512 - preview.width) // 2, 40))
        ImageDraw.Draw(panel).text((12, 12), "baseline" if block is None else f"collect block {block}", fill="black")
        panels.append(panel)
        if block is not None:
            areas = {name: float(mask.mean()) for name, mask in pipe.last_freefuse_debug["masks"].items()}
            rows.append({"block": block, "subject_mask_fraction": areas})
    grid = Image.new("RGB", (512 * columns, panel_height * ((len(panels) + columns - 1) // columns)), "white")
    for i, panel in enumerate(panels):
        grid.paste(panel, ((i % columns) * 512, (i // columns) * panel_height))
    grid.save(output / "blocks.png")
    (output / "summary.json").write_text(json.dumps({"config": vars(args), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
