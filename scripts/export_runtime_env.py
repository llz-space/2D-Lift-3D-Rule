#!/usr/bin/env python3
"""Export a human-readable runtime environment summary."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("opencv-python", "cv2"),
    ("Pillow", "PIL"),
    ("segment-anything", "segment_anything"),
    ("groundingdino", "groundingdino"),
    ("transformers", "transformers"),
    ("supervision", "supervision"),
    ("timm", "timm"),
]


def package_version(import_name: str) -> str:
    try:
        module = importlib.import_module(import_name)
    except Exception:
        return "not-installed"
    return getattr(module, "__version__", "unknown")


def run_text(command: list[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip() or completed.stderr.strip() or "ok"
    except Exception as exc:
        return f"unavailable ({exc})"


def build_report() -> str:
    repo_root = ROOT / "third_party" / "Grounded-Segment-Anything"
    lines = [
        "# Runtime Environment",
        "",
        f"project_root: {ROOT}",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"conda_default_env: {os.environ.get('CONDA_DEFAULT_ENV', 'unset')}",
        f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        "",
        "## GPU",
        run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "",
        "## Packages",
    ]

    for display_name, import_name in PACKAGES:
        lines.append(f"{display_name}: {package_version(import_name)}")

    lines.extend(
        [
            "",
            "## Grounded-Segment-Anything",
            f"repo_exists: {repo_root.exists()}",
        ]
    )
    if repo_root.exists():
        lines.append(f"repo_commit: {run_text(['git', 'rev-parse', 'HEAD'], cwd=repo_root)}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/runtime_environment.txt")
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report()
    out_path.write_text(report, encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
