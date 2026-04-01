#!/usr/bin/env python3
"""Download checkpoints used by the Grounded-SAM RGB-D pipeline."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "checkpoints"

MODEL_URLS = {
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "groundingdino_swint_ogc": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "ram_swin_large": "https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth",
}

MODEL_SETS = {
    "default": ["sam_vit_b", "groundingdino_swint_ogc"],
    "sam_h": ["sam_vit_h", "groundingdino_swint_ogc"],
    "ram": ["sam_vit_b", "groundingdino_swint_ogc", "ram_swin_large"],
}


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"skip  {out_path}")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(out_path)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument(
        "--model-set",
        choices=sorted(MODEL_SETS),
        default="default",
        help="named checkpoint bundle",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_URLS),
        help="override model-set and download the explicit model list",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    requested = args.models if args.models else MODEL_SETS[args.model_set]
    for model_name in requested:
        url = MODEL_URLS[model_name]
        filename = url.rstrip("/").split("/")[-1]
        download(url, out_dir / filename)


if __name__ == "__main__":
    main()
