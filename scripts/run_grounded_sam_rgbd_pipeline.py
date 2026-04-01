#!/usr/bin/env python3
"""Run the RGB-D -> Grounded-SAM -> 3D OBB pipeline from a JSON manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.instance_bbox_pipeline import (  # noqa: E402
    CameraIntrinsics,
    FrameInput,
    GroundedSAMSegmentor,
    SceneInstanceProcessor,
    dump_outputs_json,
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_depth(path: Path, depth_scale: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
        return depth.astype(np.float32)
    if suffix == ".npz":
        data = np.load(path)
        if "depth" in data:
            return data["depth"].astype(np.float32)
        first_key = next(iter(data.keys()))
        return data[first_key].astype(np.float32)

    depth = np.asarray(Image.open(path))
    depth = depth.astype(np.float32)
    return depth / float(depth_scale)


def load_optional_map(path_str: Optional[str]) -> Optional[np.ndarray]:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        first_key = next(iter(data.keys()))
        return data[first_key]
    return np.asarray(Image.open(path))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_world_T_camera(raw: Any) -> np.ndarray:
    matrix = np.asarray(raw, dtype=np.float32)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"world_T_camera must be 4x4, got {matrix.shape}")
    return matrix


def build_frames(manifest: Dict[str, Any], depth_scale: float) -> List[FrameInput]:
    frames: List[FrameInput] = []
    for frame in manifest["frames"]:
        image_path = resolve_path(frame["image_path"])
        depth_path = resolve_path(frame["depth_path"])
        frames.append(
            FrameInput(
                rgb=load_rgb(image_path),
                depth=load_depth(depth_path, depth_scale=depth_scale),
                world_T_camera=parse_world_T_camera(frame["world_T_camera"]),
                image_path=str(image_path),
                depth_path=str(depth_path),
                instance_id_map=load_optional_map(frame.get("instance_id_map_path")),
                semantic_class_map=load_optional_map(frame.get("semantic_class_map_path")),
                semantic_artifact_path=frame.get("semantic_artifact_path"),
            )
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="outputs/scene_instances.json")
    parser.add_argument("--semantic-label-map", default="outputs/semantic_label_map.json")
    parser.add_argument("--class-names", nargs="*")
    parser.add_argument("--text-prompt")
    parser.add_argument("--classes-file")
    parser.add_argument("--backend", choices=["auto", "repo", "transformers"], default="auto")
    parser.add_argument("--auto-label-source", choices=["none", "ram"], default="none")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--grounded-checkpoint")
    parser.add_argument("--grounding-config")
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--ram-checkpoint")
    parser.add_argument("--sam-encoder-version", default="vit_b")
    parser.add_argument("--artifact-dir", default="outputs/grounded_sam")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.8)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=8.0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--use-tracking", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics(**manifest["camera_intrinsics"])
    frames = build_frames(manifest, depth_scale=args.depth_scale)

    processor = SceneInstanceProcessor(intrinsics, depth_min=args.depth_min, depth_max=args.depth_max)
    segmentor = None
    if any(frame.instance_id_map is None or frame.semantic_class_map is None for frame in frames):
        segmentor = GroundedSAMSegmentor(
            class_names=args.class_names,
            text_prompt=args.text_prompt,
            classes_file=args.classes_file,
            backend=args.backend,
            auto_label_source=args.auto_label_source,
            sam_checkpoint=args.sam_checkpoint,
            grounded_checkpoint=args.grounded_checkpoint,
            grounding_config=args.grounding_config,
            grounding_model_id=args.grounding_model_id,
            ram_checkpoint=args.ram_checkpoint,
            sam_encoder_version=args.sam_encoder_version,
            artifact_dir=args.artifact_dir,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            nms_threshold=args.nms_threshold,
            device=args.device,
        )

    outputs = processor.process_sequence(
        frames,
        detector_segmentor=segmentor,
        use_tracking=args.use_tracking,
    )

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    dump_outputs_json(outputs, str(output_json))
    print(f"saved {output_json}")

    if segmentor is not None and getattr(segmentor, "semantic_id_to_name", None):
        label_map_path = resolve_path(args.semantic_label_map)
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        label_map = {str(k): v for k, v in segmentor.semantic_id_to_name.items()}
        label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {label_map_path}")
        print(f"detector_backend={segmentor.detector_backend}")


if __name__ == "__main__":
    main()
