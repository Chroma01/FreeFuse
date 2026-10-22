#!/usr/bin/env python
"""Attach single-LoRA references and optionally append verified MOSS results."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROWS = [
    ("flux", [("harry", "Harry Potter"), ("daiyu", "Lin Daiyu")]),
    ("z_image", [("jinx", "Jinx"), ("skeleton", "Skeletor")]),
    ("klein9b", [("harry_klein", "Harry Potter"), ("shalnark_klein", "Shalnark")]),
    ("klein4b", [("harry_klein", "Harry Potter"), ("shalnark_klein", "Shalnark")]),
    ("sdxl", [("harry", "Harry Potter"), ("daiyu", "Lin Daiyu")]),
]

def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def append_moss_row(output, row_y, comparison_dir, reference_dir):
    config = json.loads((reference_dir / "config.json").read_text())
    if config["freefuse"] is not False:
        raise ValueError("MOSS references must have FreeFuse disabled")
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, row_y, 2559, row_y + 1151), fill=(18, 22, 28))
    draw.text((256, row_y + 45), "CHARACTER REFERENCES", fill="white", font=font(27), anchor="mm")
    draw.text((256, row_y + 87), "FreeFuse disabled", fill=(185, 194, 207), font=font(24), anchor="mm")
    for i, (name, title) in enumerate([("tinkerbell", "Tinkerbell"), ("albedo", "Albedo")]):
        image = Image.open(reference_dir / (name + ".png")).convert("RGB")
        tile = ImageOps.contain(image, (480, 480), Image.Resampling.LANCZOS)
        tile_y = row_y + 128 + i * 512
        output.paste(tile, ((512-tile.width)//2, tile_y+(480-tile.height)//2))
        draw.text((256, tile_y + 495), title, fill="white", font=font(24), anchor="mm")
    for i, (filename, label, color) in enumerate([
        ("baseline.png", "MOSS IMAGE WITHOUT FREEFUSE", (235, 46, 46)),
        ("freefuse.png", "MOSS IMAGE WITH FREEFUSE", (24, 215, 196)),
    ]):
        x = 512 + i * 1024
        for size in range(84, 23, -2):
            heading_font = font(size)
            if draw.textbbox((0, 0), label, font=heading_font)[2] <= 940:
                break
        draw.text((x+515, row_y+69), label, font=heading_font, fill="black", anchor="mm")
        draw.text((x+512, row_y+66), label, font=heading_font, fill=color, anchor="mm",
                  stroke_width=max(2, size//14), stroke_fill="white")
        image = Image.open(comparison_dir / filename).convert("RGB")
        # Preserve the verified landscape composition; letterbox to the square row.
        panel = ImageOps.contain(image, (1024, 1024), Image.Resampling.LANCZOS)
        output.paste(panel, (x+(1024-panel.width)//2, row_y+128+(1024-panel.height)//2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-comparison", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="flux,z_image,klein9b,klein4b,sdxl", help="Optional subset of rows")
    parser.add_argument("--moss-comparison-dir", type=Path, help="Verified native baseline.png and freefuse.png")
    parser.add_argument("--moss-reference-dir", type=Path, help="Single-LoRA tinkerbell.png, albedo.png and config.json")
    args = parser.parse_args()
    if bool(args.moss_comparison_dir) != bool(args.moss_reference_dir):
        parser.error("Supply both MOSS directories to append its row")
    original = Image.open(args.original_comparison).convert("RGB")
    if original.size != (2048, 5940):
        raise ValueError("Expected the original 2048x5940 five-row comparison")
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    indexed = {(r["model"], r["character"]): r for r in manifest["runs"]}
    selected = args.models.split(",")
    if not selected or len(set(selected)) != len(selected) or set(selected) - {r[0] for r in ROWS}:
        raise ValueError("Unknown or duplicate model names")
    rows = [(i, row) for i, row in enumerate(ROWS) if row[0] in selected]
    row_count = len(rows) + int(args.moss_comparison_dir is not None)
    output = Image.new("RGB", (2560, 60 + 1152 * row_count + 30 * (row_count-1)), "white")
    output.paste(original.crop((0, 0, 2048, 60)), (512, 0))
    draw = ImageDraw.Draw(output)
    draw.text((256, 30), "SINGLE LoRA", fill="black", font=font(30), anchor="mm")
    for row_index, (original_index, (model, characters)) in enumerate(rows):
        row_y = 60 + row_index * 1182
        original_y = 60 + original_index * 1182
        output.paste(original.crop((0, original_y, 2048, original_y + 1152)), (512, row_y))
        draw.rectangle((0, row_y, 511, row_y + 1151), fill=(18, 22, 28))
        draw.text((256, row_y + 45), "CHARACTER REFERENCES", fill="white", font=font(27), anchor="mm")
        draw.text((256, row_y + 87), "FreeFuse disabled", fill=(185, 194, 207), font=font(24), anchor="mm")
        for i, (character, title) in enumerate(characters):
            run = indexed[(model, character)]
            if run["freefuse"] is not False:
                raise ValueError("Reference must have FreeFuse disabled")
            path = args.reference_dir / model / (character + ".png")
            image = Image.open(path).convert("RGB")
            tile_y = row_y + 128 + i * 512
            # Contain rather than crop, so all generated content remains visible.
            tile = ImageOps.contain(image, (480, 480), Image.Resampling.LANCZOS)
            output.paste(tile, ((512-tile.width)//2, tile_y+(480-tile.height)//2))
            draw.text((256, tile_y + 495), title, fill="white", font=font(24), anchor="mm")
    if args.moss_comparison_dir:
        append_moss_row(output, 60+len(rows)*1182, args.moss_comparison_dir, args.moss_reference_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(args.output)

if __name__ == "__main__":
    main()
