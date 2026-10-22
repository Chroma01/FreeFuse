"""Loopback-only MOSS FreeFuse UI; forward its port over SSH."""
import argparse
import uuid
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from src.pipeline.freefuse_moss_image_pipeline import FreeFuseMossImagePipeline, find_moss_concept_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subject", nargs=3, action="append", required=True,
                        metavar=("NAME", "LORA_PATH", "CONCEPT"))
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output-dir", default="outputs/moss_demo")
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()
    pipe = FreeFuseMossImagePipeline.from_pretrained(args.model)
    for name, path, concept in args.subject:
        pipe.load_lora_weights(path, adapter_name=name)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    def generate(prompt, concepts_text, background, width, height, steps, seed, block, collect_step, compare):
        concepts = [line.strip() for line in concepts_text.splitlines() if line.strip()]
        if len(concepts) != len(args.subject):
            raise gr.Error("Enter one concept phrase per loaded subject, in the displayed order.")
        if int(collect_step) >= int(steps):
            raise gr.Error("Collection step must be lower than total steps.")
        if int(width) <= 0 or int(height) <= 0 or int(width) % 16 or int(height) % 16:
            raise gr.Error("Width and height must be positive multiples of 16.")
        try:
            find_moss_concept_positions(pipe.tokenizer, prompt,
                                       {row[0]: concept for row, concept in zip(args.subject, concepts)})
        except ValueError as error:
            raise gr.Error(str(error)) from error
        directory = Path(args.output_dir) / f"run_{uuid.uuid4().hex[:12]}"
        directory.mkdir(parents=True, exist_ok=True)
        options = dict(width=int(width), height=int(height), num_inference_steps=int(steps))
        def noise():
            return torch.Generator("cuda").manual_seed(int(seed))
        baseline = None
        if compare:
            baseline = pipe(prompt, use_freefuse=False, generator=noise(), **options).images[0]
            baseline.save(directory / "baseline.png")
        result = pipe(prompt, concepts={row[0]: concept for row, concept in zip(args.subject, concepts)},
                      background_concept=background.strip() or None, generator=noise(), collect_block=int(block),
                      collect_step=int(collect_step), debug_save_path=directory / "routing", **options).images[0]
        result.save(directory / "freefuse.png")
        masks = [(Image.open(directory / "routing" / f"mask_{i}.png").copy(), name)
                 for i, name in enumerate(pipe.last_freefuse_debug["masks"])]
        return baseline, result, masks
    with gr.Blocks(title="FreeFuse · MOSS Image") as demo:
        gr.Markdown("# FreeFuse · MOSS Image\nCompose multiple subject LoRAs with automatic token routing.")
        gr.Markdown("Loaded subjects: " + ", ".join(row[0] for row in args.subject))
        prompt = gr.Textbox(label="Prompt", lines=4)
        concepts = gr.Textbox(label="Concept phrases (one per subject, in the order above)",
                              value="\n".join(row[2] for row in args.subject), lines=len(args.subject))
        background = gr.Textbox(label="Background phrase (optional)")
        with gr.Row():
            width = gr.Number(label="Width", value=1024, precision=0)
            height = gr.Number(label="Height", value=1024, precision=0)
            steps = gr.Slider(6, 60, value=30, step=1, label="Steps")
            seed = gr.Number(label="Seed", value=42, precision=0)
        with gr.Row():
            block = gr.Slider(0, len(pipe.transformer.layers) - 1, value=20, step=1, label="Collection block")
            collect_step = gr.Slider(0, 10, value=4, step=1, label="Collection step (zero based)")
            compare = gr.Checkbox(label="Generate same-seed baseline", value=True)
        button = gr.Button("Generate", variant="primary")
        with gr.Row():
            baseline = gr.Image(label="Baseline")
            result = gr.Image(label="FreeFuse")
        masks = gr.Gallery(label="Automatic subject masks")
        button.click(generate, [prompt, concepts, background, width, height, steps, seed, block, collect_step, compare],
                     [baseline, result, masks], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
