#!/usr/bin/env python3
"""Prepare assets and optionally submit the Krea2 CUDA comparison to HF Jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def default_repo_root(script_path: Path) -> Path:
    plugin_root = script_path.parents[1]
    if plugin_root.name == "freefuse_comfyui":
        return script_path.parents[2]
    return plugin_root


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def read_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = default_repo_root(script_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--hf", default="hf")
    parser.add_argument("--asset-repo", default="lsmpp/freefuse-krea2-assets")
    parser.add_argument("--results-repo", default="lsmpp/freefuse-krea2-results")
    parser.add_argument("--lora-bundle", type=Path, default=Path("/tmp/freefuse-krea2/krea2_character_loras.tar"))
    parser.add_argument("--github-repo", default="https://github.com/yaoliliu/FreeFuse.git")
    parser.add_argument("--git-ref", default="master")
    parser.add_argument("--image", default="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel")
    parser.add_argument("--flavor", default="l4x4")
    parser.add_argument("--timeout", default="8h")
    parser.add_argument("--work-dir", default="/tmp/freefuse-krea2")
    parser.add_argument("--upload-assets", action="store_true")
    parser.add_argument("--submit-job", action="store_true")
    parser.add_argument("--confirm-paid-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_remote_script(args: argparse.Namespace) -> str:
    work_dir = shlex.quote(args.work_dir)
    repo_dir = shlex.quote(f"{args.work_dir}/FreeFuse")
    return f"""
set -euo pipefail
export HF_HOME={work_dir}/hf_home
export HF_HUB_CACHE={work_dir}/hf_hub_cache
export HF_XET_CACHE={work_dir}/hf_xet_cache
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p {work_dir}
apt-get update
apt-get install -y --no-install-recommends git git-lfs ca-certificates
python -m pip install -U pip wheel
python -m pip install -U 'huggingface_hub[hf_xet]' hf_xet
hf auth whoami
hf download {shlex.quote(args.asset_repo)} krea2_character_loras.tar --repo-type dataset --local-dir {work_dir}
git clone --depth 1 --branch {shlex.quote(args.git_ref)} {shlex.quote(args.github_repo)} {repo_dir}
cd {repo_dir}
WORK_DIR={work_dir} FREEFUSE_REPO={repo_dir} bash freefuse_comfyui/scripts/run_krea2_remote_full_test.sh
hf upload {shlex.quote(args.results_repo)} {work_dir}/krea2_cuda_results.tar.gz krea2_cuda_results.tar.gz --repo-type dataset --private
hf upload {shlex.quote(args.results_repo)} {work_dir}/results/krea2_matrix results/krea2_matrix --repo-type dataset --private
""".strip()


def main() -> None:
    args = read_args()
    dry_run = args.dry_run or not args.upload_assets and not args.submit_job

    whoami = capture([args.hf, "auth", "whoami"])
    print(f"[krea2-hf] authenticated: {whoami}", flush=True)
    print(f"[krea2-hf] asset repo: {args.asset_repo}", flush=True)
    print(f"[krea2-hf] results repo: {args.results_repo}", flush=True)
    print(f"[krea2-hf] flavor: {args.flavor}, timeout: {args.timeout}", flush=True)
    if dry_run:
        print("[krea2-hf] dry-run mode; no upload or GPU job will be started", flush=True)
    if args.submit_job and not args.confirm_paid_gpu:
        raise SystemExit(
            "--submit-job starts a paid Hugging Face GPU job. "
            "Re-run with --confirm-paid-gpu after confirming the cost and asset upload policy."
        )

    if args.upload_assets:
        if not args.lora_bundle.exists():
            raise FileNotFoundError(args.lora_bundle)
        run(
            [
                args.hf,
                "upload",
                args.asset_repo,
                str(args.lora_bundle),
                "krea2_character_loras.tar",
                "--repo-type",
                "dataset",
                "--private",
                "--commit-message",
                "Upload Krea2 character LoRA bundle",
            ],
            dry_run=args.dry_run,
        )
    else:
        print("[krea2-hf] pass --upload-assets to upload the LoRA tar first", flush=True)

    remote_script = build_remote_script(args)
    job_cmd = [
        args.hf,
        "jobs",
        "run",
        "--flavor",
        args.flavor,
        "--timeout",
        args.timeout,
        "--secrets",
        "HF_TOKEN",
        "--label",
        "freefuse-krea2",
        "-d",
        args.image,
        "bash",
        "-lc",
        remote_script,
    ]
    if args.submit_job:
        run(job_cmd, dry_run=args.dry_run)
    else:
        print("[krea2-hf] pass --submit-job to launch the GPU job", flush=True)
        print("[krea2-hf] job command:", flush=True)
        print(" ".join(shlex.quote(part) for part in job_cmd), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
