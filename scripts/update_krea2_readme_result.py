#!/usr/bin/env python3
"""Copy a Krea2 comparison image into assets/ and update README.md."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import struct


START = "<!-- KREA2_RESULTS_START -->"
END = "<!-- KREA2_RESULTS_END -->"


def default_repo_root(script_path: Path) -> Path:
    plugin_root = script_path.parents[1]
    if plugin_root.name == "freefuse_comfyui":
        return script_path.parents[2]
    return plugin_root


def read_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = default_repo_root(script_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--asset-path", type=Path, default=Path("assets/krea2_baseline_vs_freefuse.png"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path} is not a PNG image")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def render_section(asset_path: Path, width: int, height: int) -> str:
    asset_posix = asset_path.as_posix()
    return f"""{START}
### Results on Krea2 Turbo

<p align="center">
  <img src="{asset_posix}" alt="Krea2 baseline vs FreeFuse" width="100%">
</p>

<p align="center">
  <em>Krea2 Turbo comparison using the same seed and character LoRAs. Left: standard multi-LoRA baseline. Right: FreeFuse with adaptive token-level routing. Source image size: {width}x{height}.</em>
</p>
{END}"""


def update_readme(readme_path: Path, section: str) -> str:
    text = readme_path.read_text()
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _old, after = rest.split(END, 1)
        return before.rstrip() + "\n\n" + section + after
    anchor = "## 🎨 Results"
    if anchor not in text:
        raise ValueError(f"Could not find README results anchor: {anchor}")
    head, tail = text.split(anchor, 1)
    return head + anchor + "\n\n" + section + "\n" + tail


def main() -> None:
    args = read_args()
    repo_root = args.repo_root.resolve()
    source = args.contact_sheet.resolve()
    readme_path = (repo_root / args.readme).resolve()
    asset_path = args.asset_path
    target = (repo_root / asset_path).resolve()

    if not source.exists():
        raise FileNotFoundError(source)
    if not readme_path.exists():
        raise FileNotFoundError(readme_path)

    width, height = png_size(source)
    section = render_section(asset_path, width, height)
    new_readme = update_readme(readme_path, section)

    if args.dry_run:
        print(f"[krea2-readme] would copy {source} -> {target}")
        print(f"[krea2-readme] would update {readme_path}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    readme_path.write_text(new_readme)
    print(f"[krea2-readme] copied {source} -> {target}")
    print(f"[krea2-readme] updated {readme_path}")


if __name__ == "__main__":
    main()
