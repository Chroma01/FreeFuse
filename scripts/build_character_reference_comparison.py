#!/usr/bin/env python
"""Attach single-LoRA references to the unchanged five-row comparison figure."""
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-comparison", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="flux,z_image,klein9b,klein4b,sdxl", help="Optional subset of rows")
    args = parser.parse_args()
    original = Image.open(args.original_comparison).convert("RGB")
    if original.size != (2048, 5940):
        raise ValueError("Expected the original 2048x5940 five-row comparison")
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    indexed = {(r["model"], r["character"]): r for r in manifest["runs"]}
    selected = args.models.split(",")
    if not selected or len(set(selected)) != len(selected) or set(selected) - {r[0] for r in ROWS}:
        raise ValueError("Unknown or duplicate model names")
    rows = [(i, row) for i, row in enumerate(ROWS) if row[0] in selected]
    output = Image.new("RGB", (2560, 60 + 1152 * len(rows) + 30 * (len(rows)-1)), "white")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(args.output)

if __name__ == "__main__":
    main()
