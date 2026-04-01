"""Grounded-SAM adapter for RGB-only instance+semantic generation.

The adapter prefers the official Grounded-Segment-Anything repository when the
GroundingDINO package is installed successfully. On Windows or lighter setups,
it can fall back to Hugging Face Transformers for GroundingDINO detection while
still reusing the official SAM package from the cloned repository.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.strip().lower().strip(". ,;:"))
    return normalized


def _split_prompt(prompt: str) -> List[str]:
    parts = re.split(r"[,\n]+|\s*\.\s*", prompt)
    return [part.strip() for part in parts if part and part.strip()]


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        order = order[1:][iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


class GroundedSAMSegmentor:
    """Grounded-SAM based implementation of the DetectorSegmentor protocol."""

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        text_prompt: Optional[str] = None,
        classes_file: Optional[str] = None,
        backend: str = "auto",
        auto_label_source: str = "none",
        model_root: Optional[str] = None,
        artifact_dir: str = "outputs/grounded_sam",
        sam_encoder_version: str = "vit_b",
        sam_checkpoint: Optional[str] = None,
        grounded_checkpoint: Optional[str] = None,
        grounding_config: Optional[str] = None,
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        ram_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.8,
        min_mask_area: int = 20,
    ):
        self.project_root = _project_root()
        self.model_root = Path(model_root) if model_root else self.project_root / "third_party" / "Grounded-Segment-Anything"
        self.artifact_dir = self.project_root / artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.backend = backend.lower()
        self.auto_label_source = auto_label_source.lower()
        self.sam_encoder_version = sam_encoder_version
        self.sam_checkpoint = self._resolve_optional_path(
            sam_checkpoint or os.environ.get("GROUNDING_SAM_SAM_CHECKPOINT")
        )
        self.grounded_checkpoint = self._resolve_optional_path(
            grounded_checkpoint or os.environ.get("GROUNDING_SAM_GROUNDING_CHECKPOINT")
        )
        self.grounding_config = self._resolve_optional_path(
            grounding_config
            or os.environ.get("GROUNDING_SAM_GROUNDING_CONFIG")
            or str(self.model_root / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
        )
        self.grounding_model_id = grounding_model_id
        self.ram_checkpoint = self._resolve_optional_path(ram_checkpoint or os.environ.get("GROUNDING_SAM_RAM_CHECKPOINT"))

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.min_mask_area = int(min_mask_area)
        self.device = self._select_device(device)

        self._configured_classes = self._load_configured_classes(
            class_names=class_names, text_prompt=text_prompt, classes_file=classes_file
        )
        self._sam_predictor = None
        self._detector = None
        self._detector_backend: Optional[str] = None
        self._ram_model = None
        self._semantic_id_to_name: Dict[int, str] = {}
        self._semantic_name_to_id: Dict[str, int] = {}

    @property
    def detector_backend(self) -> Optional[str]:
        return self._detector_backend

    @property
    def semantic_id_to_name(self) -> Dict[int, str]:
        return dict(self._semantic_id_to_name)

    def infer(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        del depth  # This adapter uses RGB for detection/segmentation and leaves depth to the 3D stage.

        rgb_uint8 = self._ensure_uint8_rgb(rgb)
        image_classes = self._resolve_classes_for_image(rgb_uint8)
        detections = self._detect(rgb_uint8, image_classes)
        if detections["boxes"].size == 0:
            empty_instance = np.zeros(rgb_uint8.shape[:2], dtype=np.int32)
            empty_semantic = np.zeros(rgb_uint8.shape[:2], dtype=np.int32)
            artifact_rel = self._save_visualization(rgb_uint8, empty_instance, {})
            return empty_instance, empty_semantic, artifact_rel

        masks = self._segment(rgb_uint8, detections["boxes"])
        instance_map, semantic_map, accepted = self._build_maps(
            image_shape=rgb_uint8.shape[:2],
            masks=masks,
            scores=detections["scores"],
            labels=detections["labels"],
        )
        artifact_rel = self._save_visualization(rgb_uint8, instance_map, accepted)
        return instance_map, semantic_map, artifact_rel

    def _resolve_optional_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _select_device(self, requested: Optional[str]) -> str:
        if requested:
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_configured_classes(
        self,
        class_names: Optional[Sequence[str]],
        text_prompt: Optional[str],
        classes_file: Optional[str],
    ) -> List[str]:
        configured: List[str] = []
        if class_names:
            configured.extend(str(name).strip() for name in class_names if str(name).strip())
        if text_prompt:
            configured.extend(_split_prompt(text_prompt))
        if classes_file:
            classes_path = Path(classes_file)
            if not classes_path.is_absolute():
                classes_path = self.project_root / classes_path
            text = classes_path.read_text(encoding="utf-8")
            configured.extend(line.strip() for line in text.splitlines() if line.strip())

        deduped: List[str] = []
        seen = set()
        for name in configured:
            normalized = _normalize_label(name)
            if normalized and normalized not in seen:
                deduped.append(name.strip())
                seen.add(normalized)
        return deduped

    def _resolve_classes_for_image(self, rgb: np.ndarray) -> List[str]:
        if self._configured_classes:
            return list(self._configured_classes)
        if self.auto_label_source == "ram":
            return self._predict_classes_with_ram(rgb)
        raise ValueError(
            "GroundedSAMSegmentor requires class names/text_prompt/classes_file, "
            "or auto_label_source='ram' with a valid RAM checkpoint."
        )

    def _predict_classes_with_ram(self, rgb: np.ndarray) -> List[str]:
        self._ensure_ram_model()
        from torchvision import transforms as TS
        from ram import inference_ram

        image = Image.fromarray(rgb)
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
        tensor = transform(image).unsqueeze(0).to(self.device)
        tags = inference_ram(tensor, self._ram_model)[0]
        classes = [tag.strip() for tag in tags.split(" | ") if tag.strip()]
        if not classes:
            raise RuntimeError("RAM did not produce any usable tags for Grounded-SAM detection.")
        return classes

    def _ensure_ram_model(self) -> None:
        if self._ram_model is not None:
            return
        if self.ram_checkpoint is None:
            raise ValueError("auto_label_source='ram' requires ram_checkpoint or GROUNDING_SAM_RAM_CHECKPOINT.")

        try:
            from ram.models import ram
        except Exception as exc:
            raise RuntimeError(
                "RAM auto labeling dependencies are missing. Install the recognize-anything 'ram' package first."
            ) from exc

        ram_model = ram(pretrained=str(self.ram_checkpoint), image_size=384, vit="swin_l")
        ram_model.eval()
        self._ram_model = ram_model.to(self.device)

    def _detect(self, rgb: np.ndarray, image_classes: Sequence[str]) -> Dict[str, np.ndarray | List[str]]:
        self._ensure_detector()
        if self._detector_backend == "repo":
            return self._detect_with_repo(rgb, image_classes)
        if self._detector_backend == "transformers":
            return self._detect_with_transformers(rgb, image_classes)
        raise RuntimeError("GroundingDINO detector backend was not initialized.")

    def _ensure_detector(self) -> None:
        if self._detector is not None:
            return

        errors: List[str] = []
        if self.backend in {"auto", "repo"}:
            try:
                self._detector = self._build_repo_detector()
                self._detector_backend = "repo"
                return
            except Exception as exc:
                errors.append(f"repo backend unavailable: {exc}")
                if self.backend == "repo":
                    raise

        if self.backend in {"auto", "transformers"}:
            try:
                self._detector = self._build_transformers_detector()
                self._detector_backend = "transformers"
                return
            except Exception as exc:
                errors.append(f"transformers backend unavailable: {exc}")

        raise RuntimeError("Unable to initialize a GroundingDINO backend. " + " | ".join(errors))

    def _ensure_sam_predictor(self) -> None:
        if self._sam_predictor is not None:
            return
        if self.sam_checkpoint is None:
            raise ValueError("SAM checkpoint is required. Set sam_checkpoint or GROUNDING_SAM_SAM_CHECKPOINT.")

        self._ensure_repo_paths()
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise RuntimeError(
                "segment_anything could not be imported. Run the environment setup script first."
            ) from exc

        sam_model = sam_model_registry[self.sam_encoder_version](checkpoint=str(self.sam_checkpoint))
        sam_model = sam_model.to(device=self.device)
        self._sam_predictor = SamPredictor(sam_model)

    def _ensure_repo_paths(self) -> None:
        candidates = [self.model_root, self.model_root / "GroundingDINO"]
        for candidate in candidates:
            candidate_str = str(candidate.resolve())
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    def _build_repo_detector(self):
        if self.grounded_checkpoint is None:
            raise ValueError(
                "repo backend requires grounded_checkpoint or GROUNDING_SAM_GROUNDING_CHECKPOINT."
            )
        if self.grounding_config is None or not self.grounding_config.exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {self.grounding_config}")

        self._ensure_repo_paths()
        from groundingdino.util.inference import Model

        return Model(
            model_config_path=str(self.grounding_config),
            model_checkpoint_path=str(self.grounded_checkpoint),
            device=self.device,
        )

    def _build_transformers_detector(self):
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise RuntimeError("transformers is not installed.") from exc

        processor = AutoProcessor.from_pretrained(self.grounding_model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model_id).to(self.device)
        model.eval()
        return {"processor": processor, "model": model}

    def _detect_with_repo(self, rgb: np.ndarray, image_classes: Sequence[str]) -> Dict[str, np.ndarray | List[str]]:
        image_bgr = rgb[:, :, ::-1]
        detections = self._detector.predict_with_classes(
            image=image_bgr,
            classes=list(image_classes),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        boxes = np.asarray(detections.xyxy, dtype=np.float32)
        scores = np.asarray(detections.confidence, dtype=np.float32)
        class_ids = np.asarray(detections.class_id, dtype=np.int64)
        if boxes.size == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": [],
            }

        keep = _nms_xyxy(boxes, scores, self.nms_threshold)
        labels = [str(image_classes[int(class_ids[idx])]) for idx in keep]
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels}

    def _detect_with_transformers(
        self,
        rgb: np.ndarray,
        image_classes: Sequence[str],
    ) -> Dict[str, np.ndarray | List[str]]:
        import torch

        processor = self._detector["processor"]
        model = self._detector["model"]
        image = Image.fromarray(rgb)
        prompt = ". ".join(image_classes)
        if prompt and not prompt.endswith("."):
            prompt = prompt + "."

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(rgb.shape[0], rgb.shape[1])],
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        raw_labels = results["labels"]
        labels = [self._match_label_to_vocab(str(label), image_classes) for label in raw_labels]
        if boxes.size == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": [],
            }

        keep = _nms_xyxy(boxes, scores, self.nms_threshold)
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": [labels[idx] for idx in keep]}

    def _match_label_to_vocab(self, detected_label: str, image_classes: Sequence[str]) -> str:
        normalized = _normalize_label(detected_label)
        if not normalized:
            return str(image_classes[0])

        for candidate in image_classes:
            candidate_norm = _normalize_label(str(candidate))
            if normalized == candidate_norm:
                return str(candidate)

        first_token = normalized.split()[0]
        for candidate in image_classes:
            candidate_norm = _normalize_label(str(candidate))
            if first_token in candidate_norm or candidate_norm in normalized:
                return str(candidate)

        return detected_label.strip()

    def _segment(self, rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        self._ensure_sam_predictor()
        if boxes.size == 0:
            return np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=bool)

        predictor = self._sam_predictor
        predictor.set_image(rgb)
        masks: List[np.ndarray] = []
        for box in boxes:
            predicted_masks, predicted_scores, _ = predictor.predict(box=box, multimask_output=True)
            best_idx = int(np.argmax(predicted_scores))
            masks.append(predicted_masks[best_idx].astype(bool))
        return np.asarray(masks, dtype=bool)

    def _register_semantic_label(self, label: str) -> int:
        normalized = _normalize_label(label)
        if normalized in self._semantic_name_to_id:
            return self._semantic_name_to_id[normalized]
        semantic_id = len(self._semantic_name_to_id) + 1
        self._semantic_name_to_id[normalized] = semantic_id
        self._semantic_id_to_name[semantic_id] = label.strip()
        return semantic_id

    def _build_maps(
        self,
        image_shape: Tuple[int, int],
        masks: np.ndarray,
        scores: np.ndarray,
        labels: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, object]]]:
        height, width = image_shape
        instance_map = np.zeros((height, width), dtype=np.int32)
        semantic_map = np.zeros((height, width), dtype=np.int32)
        score_map = np.full((height, width), -1.0, dtype=np.float32)
        accepted: Dict[int, Dict[str, object]] = {}

        if masks.size == 0:
            return instance_map, semantic_map, accepted

        sort_order = np.argsort(scores)[::-1]
        next_instance_id = 1
        for idx in sort_order:
            mask = masks[idx].astype(bool)
            if int(mask.sum()) < self.min_mask_area:
                continue

            label = str(labels[idx]).strip()
            semantic_id = self._register_semantic_label(label)
            winner = mask & (scores[idx] > score_map)
            if int(winner.sum()) < self.min_mask_area:
                continue

            instance_id = next_instance_id
            next_instance_id += 1
            instance_map[winner] = instance_id
            semantic_map[winner] = semantic_id
            score_map[winner] = float(scores[idx])
            accepted[instance_id] = {
                "label": label,
                "semantic_id": semantic_id,
                "score": float(scores[idx]),
            }

        return instance_map, semantic_map, accepted

    def _save_visualization(
        self,
        rgb: np.ndarray,
        instance_map: np.ndarray,
        accepted: Dict[int, Dict[str, object]],
    ) -> str:
        overlay = rgb.astype(np.float32).copy()
        unique_ids = [int(v) for v in np.unique(instance_map) if v > 0]
        for instance_id in unique_ids:
            mask = instance_map == instance_id
            color = self._instance_color(instance_id)
            overlay[mask] = overlay[mask] * 0.45 + color * 0.55

        image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for instance_id in unique_ids:
            mask = instance_map == instance_id
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            color = tuple(int(v) for v in self._instance_color(instance_id))
            meta = accepted.get(instance_id, {})
            label = str(meta.get("label", f"instance_{instance_id}"))
            score = meta.get("score")
            text = f"{instance_id}:{label}"
            if score is not None:
                text = f"{text} {float(score):.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color)

        digest = hashlib.sha1(rgb.tobytes()).hexdigest()[:12]
        out_name = f"grounded_sam_{digest}.png"
        out_path = self.artifact_dir / out_name
        image.save(out_path)
        return str(out_path.relative_to(self.project_root))

    def _instance_color(self, instance_id: int) -> np.ndarray:
        seed = np.array(
            [
                (instance_id * 37) % 255,
                (instance_id * 97) % 255,
                (instance_id * 57) % 255,
            ],
            dtype=np.float32,
        )
        return np.maximum(seed, np.array([40.0, 40.0, 40.0], dtype=np.float32))

    def _ensure_uint8_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
        if rgb.dtype == np.uint8:
            return rgb
        array = rgb.astype(np.float32)
        if array.max() <= 1.0:
            array = array * 255.0
        return np.clip(array, 0.0, 255.0).astype(np.uint8)
