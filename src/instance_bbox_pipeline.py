"""Scene-level 3D instance export pipeline.

Supports two operation modes:
1) External instance segmentation provided (preferred): no track maintenance required.
2) RGB-D + camera only: call detector/segmentor provider to generate instance/semantic maps.

Outputs include per-frame metadata (image/depth path, camera params) and per-instance:
- 2D bbox
- 2D semantic artifact path
- 3D OBB in unified world coordinates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

try:
    from .grounded_sam_segmentor import GroundedSAMSegmentor as _GroundedSAMSegmentorImpl
except ImportError:
    try:
        from grounded_sam_segmentor import GroundedSAMSegmentor as _GroundedSAMSegmentorImpl
    except ImportError:
        _GroundedSAMSegmentorImpl = None


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class FrameInput:
    rgb: Optional[np.ndarray]  # (H, W, 3) optional
    depth: np.ndarray  # (H, W), meters
    world_T_camera: np.ndarray  # (4, 4)
    image_path: str
    depth_path: str
    instance_id_map: Optional[np.ndarray] = None  # (H, W), 0 bg
    semantic_class_map: Optional[np.ndarray] = None  # (H, W)
    semantic_artifact_path: Optional[str] = None  # relative path to 2D semantic visualization


@dataclass
class InstanceObservation:
    instance_id: int
    semantic_class_id: int
    mask: np.ndarray  # (H, W) bool


@dataclass
class OBB:
    center_xyz: np.ndarray  # (3,)
    size_xyz: np.ndarray  # (3,) along local axes
    rotation_world_from_obb: np.ndarray  # (3, 3)


@dataclass
class InstanceOutput:
    instance_id: int
    semantic_class_id: int
    bbox2d_xyxy: np.ndarray
    semantic_artifact_path: Optional[str]
    obb3d: OBB


@dataclass
class FrameOutput:
    frame_index: int
    image_path: str
    depth_path: str
    camera_intrinsics: CameraIntrinsics
    world_T_camera: np.ndarray
    instances: List[InstanceOutput] = field(default_factory=list)


class DetectorSegmentor(Protocol):
    """External model interface for RGB-D + camera-only inputs."""

    def infer(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """Return (instance_id_map, semantic_class_map, semantic_artifact_path)."""


class GroundedSAMSegmentor:
    """Adapter around Grounded-Segment-Anything for RGB-only semantic completion."""

    def __init__(self, *args, **kwargs):
        if _GroundedSAMSegmentorImpl is None:
            raise ImportError(
                "GroundedSAMSegmentor implementation is unavailable. "
                "Ensure src/grounded_sam_segmentor.py is importable."
            )
        self._impl = _GroundedSAMSegmentorImpl(*args, **kwargs)

    def infer(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        return self._impl.infer(rgb, depth)

    @property
    def semantic_id_to_name(self) -> Dict[int, str]:
        return getattr(self._impl, "semantic_id_to_name", {})

    @property
    def detector_backend(self) -> Optional[str]:
        return getattr(self._impl, "detector_backend", None)


@dataclass
class TrackState:
    track_id: int
    semantic_class_id: int
    points_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    last_bbox2d: Optional[np.ndarray] = None
    age: int = 0

    def update(self, points_world: np.ndarray, bbox2d: np.ndarray, voxel_size: float = 0.03) -> None:
        merged = np.vstack([self.points_world, points_world]) if self.points_world.size else points_world
        if voxel_size > 0 and merged.shape[0] > 0:
            voxel = np.floor(merged / voxel_size).astype(np.int32)
            _, idx = np.unique(voxel, axis=0, return_index=True)
            merged = merged[np.sort(idx)]
        self.points_world = merged
        self.last_bbox2d = bbox2d
        self.age = 0


class SceneInstanceProcessor:
    """Main entry for scene-level 3D OBB generation."""

    def __init__(self, intrinsics: CameraIntrinsics, depth_min: float = 0.2, depth_max: float = 8.0):
        self.intrinsics = intrinsics
        self.depth_min = depth_min
        self.depth_max = depth_max
        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1

    def process_sequence(
        self,
        frames: List[FrameInput],
        detector_segmentor: Optional[DetectorSegmentor] = None,
        use_tracking: bool = False,
    ) -> List[FrameOutput]:
        outputs: List[FrameOutput] = []

        for idx, frame in enumerate(frames):
            observations = self._resolve_observations(frame, detector_segmentor)
            frame_out = FrameOutput(
                frame_index=idx,
                image_path=frame.image_path,
                depth_path=frame.depth_path,
                camera_intrinsics=self.intrinsics,
                world_T_camera=frame.world_T_camera,
            )

            if use_tracking:
                instance_outputs = self._process_with_tracking(frame, observations)
            else:
                # No trajectory maintenance: trust external instance ids per-frame or scene-level ids directly.
                instance_outputs = self._process_without_tracking(frame, observations)

            frame_out.instances = instance_outputs
            outputs.append(frame_out)

        return outputs

    def _resolve_observations(
        self,
        frame: FrameInput,
        detector_segmentor: Optional[DetectorSegmentor],
    ) -> List[InstanceObservation]:
        if frame.instance_id_map is None or frame.semantic_class_map is None:
            if detector_segmentor is None or frame.rgb is None:
                raise ValueError(
                    "Missing semantic/instance data. Provide instance+semantic maps, or RGB with detector_segmentor."
                )
            instance_map, semantic_map, semantic_vis = detector_segmentor.infer(frame.rgb, frame.depth)
            frame.instance_id_map = instance_map
            frame.semantic_class_map = semantic_map
            if frame.semantic_artifact_path is None:
                frame.semantic_artifact_path = semantic_vis

        assert frame.instance_id_map is not None and frame.semantic_class_map is not None
        instance_ids = np.unique(frame.instance_id_map)
        instance_ids = instance_ids[instance_ids > 0]

        obs: List[InstanceObservation] = []
        for instance_id in instance_ids:
            mask = frame.instance_id_map == instance_id
            if mask.sum() < 20:
                continue
            cls = int(np.bincount(frame.semantic_class_map[mask]).argmax())
            obs.append(InstanceObservation(instance_id=int(instance_id), semantic_class_id=cls, mask=mask))
        return obs

    def _process_without_tracking(self, frame: FrameInput, obs: List[InstanceObservation]) -> List[InstanceOutput]:
        outputs: List[InstanceOutput] = []
        for o in obs:
            points_world = self._masked_points_world(frame, o.mask)
            if points_world.shape[0] < 30:
                continue
            obb = compute_obb(points_world)
            bbox2d = mask_to_xyxy(o.mask)
            outputs.append(
                InstanceOutput(
                    instance_id=o.instance_id,
                    semantic_class_id=o.semantic_class_id,
                    bbox2d_xyxy=bbox2d,
                    semantic_artifact_path=frame.semantic_artifact_path,
                    obb3d=obb,
                )
            )
        return outputs

    def _process_with_tracking(self, frame: FrameInput, obs: List[InstanceObservation]) -> List[InstanceOutput]:
        matches, unmatched_obs = self._associate(frame, obs)
        for t in self._tracks.values():
            t.age += 1

        outputs: List[InstanceOutput] = []
        for obs_idx, track_id in matches.items():
            o = obs[obs_idx]
            points_world = self._masked_points_world(frame, o.mask)
            if points_world.shape[0] < 30:
                continue
            bbox2d = mask_to_xyxy(o.mask)
            track = self._tracks[track_id]
            track.update(points_world, bbox2d)
            outputs.append(
                InstanceOutput(
                    instance_id=track_id,
                    semantic_class_id=o.semantic_class_id,
                    bbox2d_xyxy=bbox2d,
                    semantic_artifact_path=frame.semantic_artifact_path,
                    obb3d=compute_obb(track.points_world),
                )
            )

        for obs_idx in unmatched_obs:
            o = obs[obs_idx]
            points_world = self._masked_points_world(frame, o.mask)
            if points_world.shape[0] < 30:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            bbox2d = mask_to_xyxy(o.mask)
            self._tracks[track_id] = TrackState(track_id=track_id, semantic_class_id=o.semantic_class_id)
            self._tracks[track_id].update(points_world, bbox2d)
            outputs.append(
                InstanceOutput(
                    instance_id=track_id,
                    semantic_class_id=o.semantic_class_id,
                    bbox2d_xyxy=bbox2d,
                    semantic_artifact_path=frame.semantic_artifact_path,
                    obb3d=compute_obb(points_world),
                )
            )

        # remove stale tracks
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.age <= 20}
        return outputs

    def _associate(self, frame: FrameInput, obs: List[InstanceObservation]) -> Tuple[Dict[int, int], List[int]]:
        """Improved association for close objects: combine 3D distance + 3D IoU + 2D IoU."""
        if not self._tracks:
            return {}, list(range(len(obs)))

        cost_items: List[Tuple[float, int, int]] = []  # (cost, obs_idx, track_id)
        for obs_idx, o in enumerate(obs):
            bbox2d = mask_to_xyxy(o.mask)
            points_world = self._masked_points_world(frame, o.mask)
            if points_world.shape[0] < 30:
                continue
            obb = compute_obb(points_world)
            for track_id, track in self._tracks.items():
                if track.semantic_class_id != o.semantic_class_id or track.points_world.shape[0] < 30:
                    continue
                track_obb = compute_obb(track.points_world)
                center_dist = np.linalg.norm(obb.center_xyz - track_obb.center_xyz)
                iou3d = obb_iou_approx(obb, track_obb)
                iou2d = bbox_iou_xyxy(bbox2d, track.last_bbox2d) if track.last_bbox2d is not None else 0.0

                if center_dist > 2.0 and iou3d < 0.01 and iou2d < 0.05:
                    continue

                # lower is better
                cost = center_dist - 0.8 * iou3d - 0.3 * iou2d
                cost_items.append((float(cost), obs_idx, track_id))

        cost_items.sort(key=lambda x: x[0])
        matched_obs = set()
        matched_tracks = set()
        matches: Dict[int, int] = {}

        for _, obs_idx, track_id in cost_items:
            if obs_idx in matched_obs or track_id in matched_tracks:
                continue
            matched_obs.add(obs_idx)
            matched_tracks.add(track_id)
            matches[obs_idx] = track_id

        unmatched_obs = [i for i in range(len(obs)) if i not in matched_obs]
        return matches, unmatched_obs

    def _masked_points_world(self, frame: FrameInput, mask: np.ndarray) -> np.ndarray:
        pts_cam = back_project_depth(frame.depth, mask, self.intrinsics, self.depth_min, self.depth_max)
        return transform_points(frame.world_T_camera, pts_cam)


def back_project_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_min: float = 0.2,
    depth_max: float = 8.0,
) -> np.ndarray:
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (xs - intrinsics.cx) * z / intrinsics.fx
    y = (ys - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1)


def transform_points(world_T_camera: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    if pts_cam.size == 0:
        return pts_cam
    ones = np.ones((pts_cam.shape[0], 1), dtype=pts_cam.dtype)
    pts_h = np.concatenate([pts_cam, ones], axis=1)
    pts_w = (world_T_camera @ pts_h.T).T
    return pts_w[:, :3]


def mask_to_xyxy(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    if b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def compute_obb(points_world: np.ndarray) -> OBB:
    """PCA-based OBB."""
    if points_world.shape[0] < 4:
        center = points_world.mean(axis=0) if points_world.size else np.zeros(3, dtype=np.float32)
        return OBB(center_xyz=center, size_xyz=np.zeros(3, dtype=np.float32), rotation_world_from_obb=np.eye(3))

    center = points_world.mean(axis=0)
    centered = points_world - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]

    proj = centered @ R
    pmin = proj.min(axis=0)
    pmax = proj.max(axis=0)
    size = pmax - pmin
    obb_center_local = 0.5 * (pmax + pmin)
    obb_center_world = center + R @ obb_center_local
    return OBB(center_xyz=obb_center_world, size_xyz=size, rotation_world_from_obb=R)


def obb_iou_approx(a: OBB, b: OBB) -> float:
    """Approximate IoU via AABB envelopes of OBBs (fast gating signal)."""
    amin = a.center_xyz - a.size_xyz / 2.0
    amax = a.center_xyz + a.size_xyz / 2.0
    bmin = b.center_xyz - b.size_xyz / 2.0
    bmax = b.center_xyz + b.size_xyz / 2.0

    inter_min = np.maximum(amin, bmin)
    inter_max = np.minimum(amax, bmax)
    inter = np.maximum(0.0, inter_max - inter_min)
    inter_v = float(inter[0] * inter[1] * inter[2])
    va = float(np.prod(np.maximum(0.0, amax - amin)))
    vb = float(np.prod(np.maximum(0.0, bmax - bmin)))
    union = va + vb - inter_v + 1e-6
    return inter_v / union


def dump_outputs_json(outputs: List[FrameOutput], out_path: str) -> None:
    """Export frame-level outputs to a portable JSON file."""
    import json

    def _arr(x: np.ndarray) -> List[float]:
        return [float(v) for v in x.reshape(-1)]

    serialized = []
    for f in outputs:
        serialized.append(
            {
                "frame_index": f.frame_index,
                "image_path": f.image_path,
                "depth_path": f.depth_path,
                "camera_intrinsics": {
                    "fx": f.camera_intrinsics.fx,
                    "fy": f.camera_intrinsics.fy,
                    "cx": f.camera_intrinsics.cx,
                    "cy": f.camera_intrinsics.cy,
                },
                "world_T_camera": _arr(f.world_T_camera),
                "instances": [
                    {
                        "instance_id": inst.instance_id,
                        "semantic_class_id": inst.semantic_class_id,
                        "bbox2d_xyxy": _arr(inst.bbox2d_xyxy),
                        "semantic_artifact_path": inst.semantic_artifact_path,
                        "obb3d": {
                            "center_xyz": _arr(inst.obb3d.center_xyz),
                            "size_xyz": _arr(inst.obb3d.size_xyz),
                            "rotation_world_from_obb": _arr(inst.obb3d.rotation_world_from_obb),
                        },
                    }
                    for inst in f.instances
                ],
            }
        )

    Path(out_path).write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
