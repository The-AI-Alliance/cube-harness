#!/usr/bin/env python3
"""Upload the WAA Windows VM disk image to HuggingFace.

Prerequisites:
    huggingface-cli login   # one-time auth

Usage:
    python scripts/upload_image.py
    python scripts/upload_image.py --repo-id your-org/waa-windows-image
    python scripts/upload_image.py --image-path /path/to/data.img
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

_DEFAULT_IMAGE = Path.home() / ".cube" / "waa" / "storage" / "data.img"
_DEFAULT_REPO = "The-AI-Alliance/waa-windows-image"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-path", type=str, default=str(_DEFAULT_IMAGE), help="Path to data.img")
    parser.add_argument("--repo-id", type=str, default=_DEFAULT_REPO, help="HuggingFace repo (dataset)")
    parser.add_argument("--filename", type=str, default="data.img", help="Filename in the repo")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: {image_path} not found", file=sys.stderr)
        sys.exit(1)

    size_gb = image_path.stat().st_size / (1024**3)
    print(f"Uploading {image_path} ({size_gb:.1f} GB) to {args.repo_id}/{args.filename}")

    api = HfApi()

    # Create the repo if it doesn't exist
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=True)

    # Upload with progress
    api.upload_file(
        path_or_fileobj=str(image_path),
        path_in_repo=args.filename,
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    print(f"Done! Image uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
