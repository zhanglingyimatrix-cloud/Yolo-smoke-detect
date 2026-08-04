import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "smoking.yaml"

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "configs" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "configs" / "matplotlib"))
os.environ.setdefault("HF_HOME", str(ROOT / "configs" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "configs" / "huggingface" / "hub"))
os.environ.setdefault("PYTHONUTF8", "1")

from ultralytics import YOLO


@dataclass
class PersonEvidence:
    box: tuple[int, int, int, int]
    mouth: tuple[int, int] | None
    face_box: tuple[int, int, int, int] | None
    face_center: tuple[int, int] | None
    hand_to_face_hit: bool
    fingertip_to_mouth_hit: bool
    hand_hit: bool
    cigarette_hit: bool
    smoke_hit: bool
    hands: list[dict]
    nearest_hand: str | None
    nearest_fingertip_hand: str | None
    hand_face_distance_px: float | None
    hand_face_distance_ratio: float | None
    hand_face_threshold: float
    hand_face_norm: float | None
    fingertip_mouth_distance_px: float | None
    fingertip_mouth_distance_ratio: float | None
    fingertip_mouth_threshold: float
    fingertip_mouth_norm: float | None
    detail_trigger: bool
    detail_crop_box: tuple[int, int, int, int] | None
    detail_crop_path: str | None
    detail_model_status: str
    detail_model_hit: bool
    detail_model_detections: list[dict]
    depth_status: str
    depth_consistent: bool | None
    depth_delta: float | None
    depth_delta_ratio: float | None
    depth_threshold: float | None
    depth_mouth: float | None
    depth_hand: float | None
    depth_debug_path: str | None
    score: float
    label: str


SMOKE_CLASS_NAMES = {"smoke", "cigarette_smoke", "tobacco_smoke"}
CIGARETTE_CLASS_NAMES = {"cigarette", "cigarette_tip", "cigarette_butt", "smoking", "tobacco"}


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def detail_detection_kind(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if normalized in SMOKE_CLASS_NAMES or "smoke" in normalized:
        return "smoke"
    if normalized in CIGARETTE_CLASS_NAMES or "cigarette" in normalized:
        return "cigarette"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect suspected smoking from pose and optional cigarette model.")
    parser.add_argument("--source", type=Path, default=None, help="Input mp4 path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pose-model", type=Path, default=None)
    parser.add_argument("--cigarette-model", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "opened": cap.isOpened(),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0,
    }
    cap.release()
    return info


def point_visible(kpts: np.ndarray, confs: np.ndarray, idx: int, min_conf: float) -> bool:
    return idx < len(kpts) and idx < len(confs) and confs[idx] >= min_conf


def mouth_point(kpts: np.ndarray, confs: np.ndarray, min_conf: float) -> tuple[float, float] | None:
    if not point_visible(kpts, confs, 0, min_conf):
        return None

    nose = kpts[0].astype(float)
    shoulder_points = []
    for idx in (5, 6):
        if point_visible(kpts, confs, idx, min_conf):
            shoulder_points.append(kpts[idx].astype(float))

    if shoulder_points:
        shoulder_center = np.mean(np.stack(shoulder_points), axis=0)
        return tuple(nose + 0.18 * (shoulder_center - nose))

    return tuple(nose)


def normalizer(box: np.ndarray, kpts: np.ndarray, confs: np.ndarray, min_conf: float) -> float:
    x1, y1, x2, y2 = box.astype(float)
    box_norm = max(x2 - x1, y2 - y1, 1.0)

    if point_visible(kpts, confs, 5, min_conf) and point_visible(kpts, confs, 6, min_conf):
        shoulder_width = float(np.linalg.norm(kpts[5] - kpts[6]))
        return max(shoulder_width, box_norm * 0.22, 1.0)

    return max(box_norm * 0.30, 1.0)


def face_box_from_keypoints(
    box: np.ndarray,
    kpts: np.ndarray,
    confs: np.ndarray,
    min_conf: float,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    face_points = []
    for idx in (0, 1, 2, 3, 4):
        if point_visible(kpts, confs, idx, min_conf):
            face_points.append(kpts[idx].astype(float))

    if not face_points:
        return None

    points = np.stack(face_points)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    person_x1, person_y1, person_x2, person_y2 = box.astype(float)
    person_w = max(person_x2 - person_x1, 1.0)
    person_h = max(person_y2 - person_y1, 1.0)

    face_w = max(x2 - x1, person_w * 0.12)
    face_h = max(y2 - y1, person_h * 0.12)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    scaled_w = face_w * scale
    scaled_h = face_h * scale
    return (
        max(0, int(cx - scaled_w / 2)),
        max(0, int(cy - scaled_h / 2)),
        min(frame_width - 1, int(cx + scaled_w / 2)),
        min(frame_height - 1, int(cy + scaled_h / 2)),
    )


def box_center(box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)


def box_norm(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return max(x2 - x1, y2 - y1, 1.0)


def hand_to_face_evidence(
    kpts: np.ndarray,
    confs: np.ndarray,
    face_box: tuple[int, int, int, int],
    body_norm: float,
    min_conf: float,
    ratio_threshold: float,
) -> dict:
    candidates = []
    for idx, side in ((9, "left"), (10, "right")):
        if point_visible(kpts, confs, idx, min_conf):
            candidates.append(
                {
                    "side": side,
                    "point": kpts[idx].astype(float),
                    "conf": float(confs[idx]),
                }
            )

    if not candidates:
        return {
            "hit": False,
            "nearest_hand": None,
            "distance_px": None,
            "distance_ratio": None,
            "norm": None,
            "hands": [],
        }

    center = box_center(face_box)
    face_norm = max(box_norm(face_box), body_norm * 0.18, 1.0)
    hands = []
    for candidate in candidates:
        point = candidate["point"]
        distance_px = float(np.linalg.norm(point - center))
        distance_ratio = distance_px / face_norm
        inside_face_focus = point_in_box((float(point[0]), float(point[1])), face_box)
        hands.append(
            {
                "side": candidate["side"],
                "point": (int(point[0]), int(point[1])),
                "conf": candidate["conf"],
                "distance_px": distance_px,
                "distance_ratio": distance_ratio,
                "inside_face_box": inside_face_focus,
                "hit": inside_face_focus or distance_ratio <= ratio_threshold,
            }
        )

    nearest = min(hands, key=lambda hand: hand["distance_ratio"])
    return {
        "hit": any(hand["hit"] for hand in hands),
        "nearest_hand": nearest["side"],
        "distance_px": nearest["distance_px"],
        "distance_ratio": nearest["distance_ratio"],
        "norm": face_norm,
        "hands": hands,
    }


def estimate_fingertip(
    kpts: np.ndarray,
    confs: np.ndarray,
    side: str,
    min_conf: float,
    extension_ratio: float,
) -> dict | None:
    indexes = {
        "left": {"elbow": 7, "wrist": 9},
        "right": {"elbow": 8, "wrist": 10},
    }[side]
    wrist_idx = indexes["wrist"]
    elbow_idx = indexes["elbow"]

    if not point_visible(kpts, confs, wrist_idx, min_conf):
        return None

    wrist = kpts[wrist_idx].astype(float)
    fingertip = wrist.copy()
    method = "wrist"

    if point_visible(kpts, confs, elbow_idx, min_conf):
        elbow = kpts[elbow_idx].astype(float)
        forearm = wrist - elbow
        if float(np.linalg.norm(forearm)) > 1.0:
            fingertip = wrist + extension_ratio * forearm
            method = "forearm_extension"

    return {
        "side": side,
        "wrist": (int(wrist[0]), int(wrist[1])),
        "fingertip": (int(fingertip[0]), int(fingertip[1])),
        "conf": float(confs[wrist_idx]),
        "method": method,
    }


def fingertip_to_mouth_evidence(
    kpts: np.ndarray,
    confs: np.ndarray,
    mouth: tuple[float, float],
    face_box: tuple[int, int, int, int] | None,
    body_norm: float,
    min_conf: float,
    ratio_threshold: float,
    extension_ratio: float,
) -> dict:
    candidates = []
    mouth_np = np.array(mouth, dtype=float)
    mouth_point_int = (int(mouth[0]), int(mouth[1]))

    norm = max(box_norm(face_box) if face_box is not None else 0.0, body_norm * 0.18, 1.0)
    for side in ("left", "right"):
        candidate = estimate_fingertip(kpts, confs, side, min_conf, extension_ratio)
        if candidate is None:
            continue
        fingertip_np = np.array(candidate["fingertip"], dtype=float)
        distance_px = float(np.linalg.norm(fingertip_np - mouth_np))
        distance_ratio = distance_px / norm
        mouth_zone = expand_box(
            (mouth_point_int[0] - int(norm * 0.28), mouth_point_int[1] - int(norm * 0.28),
             mouth_point_int[0] + int(norm * 0.28), mouth_point_int[1] + int(norm * 0.28)),
            1.0,
            100000,
            100000,
        )
        inside_mouth_zone = point_in_box(candidate["fingertip"], mouth_zone)
        candidate.update(
            {
                "distance_px": distance_px,
                "distance_ratio": distance_ratio,
                "inside_mouth_zone": inside_mouth_zone,
                "hit": inside_mouth_zone or distance_ratio <= ratio_threshold,
            }
        )
        candidates.append(candidate)

    if not candidates:
        return {
            "hit": False,
            "nearest_hand": None,
            "distance_px": None,
            "distance_ratio": None,
            "norm": norm,
            "fingertips": [],
        }

    nearest = min(candidates, key=lambda item: item["distance_ratio"])
    return {
        "hit": any(item["hit"] for item in candidates),
        "nearest_hand": nearest["side"],
        "distance_px": nearest["distance_px"],
        "distance_ratio": nearest["distance_ratio"],
        "norm": norm,
        "fingertips": candidates,
    }


def wrist_hits(
    kpts: np.ndarray,
    confs: np.ndarray,
    mouth: tuple[float, float],
    norm: float,
    min_conf: float,
    ratio_threshold: float,
) -> list[tuple[int, int]]:
    hits = []
    mouth_np = np.array(mouth, dtype=float)
    for idx in (9, 10):
        if not point_visible(kpts, confs, idx, min_conf):
            continue
        wrist = kpts[idx].astype(float)
        if float(np.linalg.norm(wrist - mouth_np)) / norm <= ratio_threshold:
            hits.append((int(wrist[0]), int(wrist[1])))
    return hits


def expand_box(box: tuple[int, int, int, int], scale: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return (
        max(0, int(cx - w / 2)),
        max(0, int(cy - h / 2)),
        min(width - 1, int(cx + w / 2)),
        min(height - 1, int(cy + h / 2)),
    )


def point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(0, min(width - 1, int(x2))),
        max(0, min(height - 1, int(y2))),
    )


def union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    valid = [box for box in boxes if box is not None and box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def focus_roi_box(
    person: PersonEvidence,
    frame_width: int,
    frame_height: int,
    crop_scale: float,
) -> tuple[int, int, int, int] | None:
    boxes = []
    if person.face_box is not None:
        boxes.append(person.face_box)
    if person.mouth is not None:
        x, y = person.mouth
        margin = max(24, int((box_norm(person.face_box) if person.face_box else box_norm(person.box)) * 0.18))
        boxes.append((x - margin, y - margin, x + margin, y + margin))

    for hand in person.hands:
        for key in ("point", "fingertip"):
            point = hand.get(key)
            if point is None:
                continue
            x, y = point
            margin = max(32, int((box_norm(person.face_box) if person.face_box else box_norm(person.box)) * 0.16))
            boxes.append((x - margin, y - margin, x + margin, y + margin))

    roi = union_boxes(boxes)
    if roi is None:
        roi = person.box

    return expand_box(clip_box(roi, frame_width, frame_height), crop_scale, frame_width, frame_height)


def face_detail_roi_box(
    person: PersonEvidence,
    frame_width: int,
    frame_height: int,
    crop_scale: float,
) -> tuple[int, int, int, int] | None:
    if person.face_box is not None:
        return expand_box(clip_box(person.face_box, frame_width, frame_height), crop_scale, frame_width, frame_height)

    if person.mouth is not None:
        norm = max(box_norm(person.box) * 0.18, 1.0)
        margin_x = max(28, int(norm * 0.75))
        margin_y = max(28, int(norm * 0.65))
        x, y = person.mouth
        mouth_box = (x - margin_x, y - margin_y, x + margin_x, y + margin_y)
        return expand_box(clip_box(mouth_box, frame_width, frame_height), crop_scale, frame_width, frame_height)

    return None


def detail_model_roi_box(
    person: PersonEvidence,
    frame_width: int,
    frame_height: int,
    cfg: dict,
) -> tuple[int, int, int, int] | None:
    detail_cfg = cfg.get("detail", {})
    roi_mode = str(detail_cfg.get("roi_mode", "focus")).strip().lower()
    face_crop_scale = float(detail_cfg.get("face_crop_scale", detail_cfg.get("crop_scale", 1.0)))
    if roi_mode in {"face", "face_box", "mouth_face"}:
        return face_detail_roi_box(person, frame_width, frame_height, face_crop_scale)

    crop_scale = float(detail_cfg.get("crop_scale", 1.35))
    return focus_roi_box(person, frame_width, frame_height, crop_scale)


def write_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return path.exists() and path.stat().st_size > 0


class DepthEstimator:
    def __init__(self, cfg: dict, device: str) -> None:
        self.cfg = cfg.get("depth", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.device_arg = device
        self.processor = None
        self.model = None
        self.torch = None
        self.load_error = None
        self.loaded = False

    def load(self) -> bool:
        if not self.enabled:
            self.load_error = "disabled"
            return False
        if self.loaded:
            return True
        if self.load_error:
            return False

        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            self.load_error = f"missing_dependency: {exc}"
            return False

        try:
            model_name = self.cfg.get("model", "depth-anything/Depth-Anything-V2-Small-hf")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            if self.device_arg.startswith("cuda") and torch.cuda.is_available():
                cuda_index = self.device_arg.split(":", 1)[1] if ":" in self.device_arg else "0"
                self.model = self.model.to(f"cuda:{cuda_index}")
            else:
                self.model = self.model.to("cpu")
            self.model.eval()
            self.torch = torch
            self.image_cls = Image
            self.loaded = True
            return True
        except Exception as exc:
            self.load_error = f"model_unavailable: {exc}"
            return False

    def estimate(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
        if not self.enabled:
            return None, "disabled"
        if not self.load():
            return None, self.load_error or "unavailable"

        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image = self.image_cls.fromarray(image_rgb)
            inputs = self.processor(images=image, return_tensors="pt")
            model_device = next(self.model.parameters()).device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
                prediction = self.torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image_bgr.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.detach().cpu().numpy().astype(np.float32)
            finite = np.isfinite(depth)
            if not finite.any():
                return None, "invalid_depth"
            depth = np.where(finite, depth, np.nanmedian(depth[finite]))
            return depth, "ok"
        except Exception as exc:
            return None, f"inference_failed: {exc}"


def sample_depth(depth: np.ndarray, point: tuple[int, int], radius: int) -> float | None:
    x, y = point
    h, w = depth.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)
    patch = depth[y1:y2, x1:x2]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def depth_to_debug_image(depth: np.ndarray) -> np.ndarray:
    depth_min = float(np.nanpercentile(depth, 2))
    depth_max = float(np.nanpercentile(depth, 98))
    if depth_max <= depth_min:
        scaled = np.zeros(depth.shape, dtype=np.uint8)
    else:
        scaled = np.clip((depth - depth_min) / (depth_max - depth_min) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def selected_detail_hand_point(person: PersonEvidence) -> tuple[int, int] | None:
    preferred_side = person.nearest_fingertip_hand or person.nearest_hand
    if preferred_side is None:
        return None
    hand = next((item for item in person.hands if item["side"] == preferred_side), None)
    if hand is None:
        return None
    return hand.get("fingertip") or hand.get("point")


def run_depth_on_focus_crop(
    person: PersonEvidence,
    crop: np.ndarray,
    crop_box: tuple[int, int, int, int],
    depth_estimator: DepthEstimator | None,
    cfg: dict,
    output_dir: Path,
    frame_idx: int,
    person_idx: int,
) -> None:
    depth_cfg = cfg.get("depth", {})
    if depth_estimator is None or not depth_cfg.get("enabled", False):
        person.depth_status = "disabled"
        return

    if not depth_cfg.get("run_on_detail_trigger", True):
        person.depth_status = "disabled_for_detail"
        return

    depth_map, status = depth_estimator.estimate(crop)
    if status != "ok" or depth_map is None:
        person.depth_status = status
        return

    if person.mouth is None:
        person.depth_status = "missing_mouth"
        return

    hand_point = selected_detail_hand_point(person)
    if hand_point is None:
        person.depth_status = "missing_hand_point"
        return

    x1, y1, _, _ = crop_box
    mouth_local = (int(person.mouth[0] - x1), int(person.mouth[1] - y1))
    hand_local = (int(hand_point[0] - x1), int(hand_point[1] - y1))
    radius = max(2, int(min(crop.shape[:2]) * float(depth_cfg.get("sample_radius_ratio", 0.045))))
    mouth_depth = sample_depth(depth_map, mouth_local, radius)
    hand_depth = sample_depth(depth_map, hand_local, radius)
    if mouth_depth is None or hand_depth is None:
        person.depth_status = "sample_failed"
        return

    d_low = float(np.nanpercentile(depth_map, 5))
    d_high = float(np.nanpercentile(depth_map, 95))
    contrast = max(d_high - d_low, 1e-6)
    threshold = float(depth_cfg.get("hand_mouth_depth_delta_threshold", 0.18))
    min_contrast = float(depth_cfg.get("min_depth_contrast", 0.03))
    delta = abs(hand_depth - mouth_depth)
    delta_ratio = delta / contrast

    person.depth_status = "ok_low_contrast" if contrast < min_contrast else "ok"
    person.depth_mouth = mouth_depth
    person.depth_hand = hand_depth
    person.depth_delta = delta
    person.depth_delta_ratio = delta_ratio
    person.depth_threshold = threshold
    person.depth_consistent = delta_ratio <= threshold or contrast < min_contrast

    if depth_cfg.get("save_depth_debug", False):
        debug = depth_to_debug_image(depth_map)
        cv2.circle(debug, mouth_local, radius, (255, 255, 255), 2)
        cv2.circle(debug, hand_local, radius, (0, 255, 255), 2)
        cv2.line(debug, mouth_local, hand_local, (0, 255, 255), 1, cv2.LINE_AA)
        debug_path = output_dir / "depth_debug" / f"frame_{frame_idx:06d}_person_{person_idx:02d}.jpg"
        if write_image(debug_path, debug):
            person.depth_debug_path = str(debug_path)


def run_detail_model_on_focus(
    frame: np.ndarray,
    people: list[PersonEvidence],
    cigarette_model: YOLO | None,
    depth_estimator: DepthEstimator | None,
    cfg: dict,
    output_dir: Path,
    frame_idx: int,
    saved_crops: int,
    device: str,
    fallback_imgsz: int,
) -> int:
    detail_cfg = cfg.get("detail", {})
    if not detail_cfg.get("enabled", True):
        return saved_crops

    height, width = frame.shape[:2]
    depth_crop_scale = float(detail_cfg.get("depth_crop_scale", detail_cfg.get("crop_scale", 1.35)))
    save_crops = bool(detail_cfg.get("save_focus_crops", True))
    max_saved_crops = int(detail_cfg.get("max_saved_crops", 300))
    model_imgsz = int(detail_cfg.get("model_img_size", fallback_imgsz))
    conf = float(cfg["detection"]["cigarette_conf"])
    depth_cfg = cfg.get("depth", {})
    require_depth_match = bool(depth_cfg.get("require_depth_consistency_for_detail", True))

    for person_idx, person in enumerate(people):
        distance_candidate = person.fingertip_to_mouth_hit
        if not distance_candidate:
            person.detail_trigger = False
            person.detail_model_status = "distance_clear"
            person.label = "person"
            if not person.cigarette_hit:
                person.score = 0.0
            continue

        depth_crop_box = focus_roi_box(person, width, height, depth_crop_scale)
        if depth_crop_box is None:
            person.detail_trigger = False
            person.detail_model_status = "no_depth_crop"
            continue

        dx1, dy1, dx2, dy2 = depth_crop_box
        depth_crop = frame[dy1:dy2, dx1:dx2]
        if depth_crop.size == 0:
            person.detail_trigger = False
            person.detail_model_status = "empty_depth_crop"
            continue

        run_depth_on_focus_crop(person, depth_crop, depth_crop_box, depth_estimator, cfg, output_dir, frame_idx, person_idx)
        depth_checked = person.depth_status in ("ok", "ok_low_contrast")
        if require_depth_match:
            if depth_checked and person.depth_consistent is False:
                person.detail_trigger = False
                person.detail_model_status = "depth_conflict_clear"
                person.label = "clear_depth_conflict"
                if not person.cigarette_hit:
                    person.score = 0.05
                continue
            if not (depth_checked and person.depth_consistent is True):
                person.detail_trigger = False
                person.detail_model_status = "depth_unconfirmed_clear"
                person.label = "clear_depth_unconfirmed"
                if not person.cigarette_hit:
                    person.score = 0.10
                continue

        person.detail_trigger = True
        person.label = "warning_depth_match"
        person.score = max(person.score, 0.72)

        crop_box = detail_model_roi_box(person, width, height, cfg)
        person.detail_crop_box = crop_box
        if crop_box is None:
            person.detail_model_status = "no_face_crop"
            continue

        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            person.detail_model_status = "empty_face_crop"
            continue

        if save_crops and saved_crops < max_saved_crops:
            crop_path = output_dir / "focus_crops" / f"frame_{frame_idx:06d}_person_{person_idx:02d}.jpg"
            if write_image(crop_path, crop):
                person.detail_crop_path = str(crop_path)
                saved_crops += 1

        if cigarette_model is None:
            person.detail_model_status = "pending_model"
            continue

        results = cigarette_model.predict(crop, conf=conf, imgsz=model_imgsz, device=device, verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            person.detail_model_status = "model_ran_no_detection"
            continue

        detections = []
        names = getattr(cigarette_model, "names", {})
        for box in boxes:
            cls_id = int(box.cls.item())
            local_xyxy = [float(value) for value in box.xyxy[0].tolist()]
            global_xyxy = [
                round(local_xyxy[0] + x1, 3),
                round(local_xyxy[1] + y1, 3),
                round(local_xyxy[2] + x1, 3),
                round(local_xyxy[3] + y1, 3),
            ]
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
                    "conf": round(float(box.conf.item()), 3),
                    "local_xyxy": [round(value, 3) for value in local_xyxy],
                    "global_xyxy": global_xyxy,
                }
            )

        for detection in detections:
            detection["kind"] = detail_detection_kind(detection["class_name"])

        person.detail_model_detections = detections
        person.detail_model_hit = bool(detections)
        person.cigarette_hit = person.cigarette_hit or any(
            detection["kind"] == "cigarette" for detection in detections
        )
        person.smoke_hit = person.smoke_hit or any(detection["kind"] == "smoke" for detection in detections)
        person.detail_model_status = "model_ran_detection"
        person.score = min(1.0, person.score + (0.35 if person.cigarette_hit else 0.0) + (0.25 if person.smoke_hit else 0.0))
        if person.fingertip_to_mouth_hit or person.hand_hit:
            person.label = "smoking_evidence"
        else:
            person.label = "suspected_smoking_object"

    return saved_crops


def detect_cigarette_hits(
    cigarette_model: YOLO | None,
    frame: np.ndarray,
    people: list[PersonEvidence],
    conf: float,
    imgsz: int,
    device: str,
) -> list[bool]:
    if cigarette_model is None or not people:
        return [False for _ in people]

    results = cigarette_model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        return [False for _ in people]

    height, width = frame.shape[:2]
    centers = []
    for xyxy in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = xyxy
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    hits = []
    for person in people:
        if person.mouth is None:
            hits.append(False)
            continue
        near_person = expand_box(person.box, 1.08, width, height)
        mouth_zone = expand_box(
            (person.mouth[0] - 40, person.mouth[1] - 40, person.mouth[0] + 40, person.mouth[1] + 40),
            1.0,
            width,
            height,
        )
        hits.append(any(point_in_box(center, near_person) and point_in_box(center, mouth_zone) for center in centers))
    return hits


def analyze_frame(
    pose_model: YOLO,
    cigarette_model: YOLO | None,
    frame: np.ndarray,
    cfg: dict,
    device: str,
    imgsz: int,
    pose_conf: float,
) -> list[PersonEvidence]:
    pose_cfg = cfg["pose"]
    det_cfg = cfg["detection"]
    keypoint_conf = pose_cfg["keypoint_conf"]
    focus_keypoint_conf = pose_cfg.get("focus_keypoint_conf", keypoint_conf)
    frame_height, frame_width = frame.shape[:2]
    results = pose_model.predict(frame, conf=pose_conf, imgsz=imgsz, device=device, verbose=False)
    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or keypoints is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    kxy = keypoints.xy.cpu().numpy()
    kconf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.ones(kxy.shape[:2], dtype=float)

    people = []
    for i, box in enumerate(xyxy):
        if classes[i] != int(det_cfg["classes"]["person"]):
            continue

        mouth = mouth_point(kxy[i], kconf[i], keypoint_conf)
        if mouth is None:
            continue

        norm = normalizer(box, kxy[i], kconf[i], keypoint_conf)
        face_box = face_box_from_keypoints(
            box,
            kxy[i],
            kconf[i],
            focus_keypoint_conf,
            pose_cfg.get("face_box_scale", 2.2),
            frame_width,
            frame_height,
        )
        hand_to_face_hit = False
        hand_face_ratio = None
        hand_face_px = None
        hand_face_norm = None
        nearest_hand = None
        hands = []
        hand_face_threshold = pose_cfg.get("hand_to_face_distance_ratio", 0.55)
        fingertip_to_mouth_hit = False
        fingertip_mouth_ratio = None
        fingertip_mouth_px = None
        fingertip_mouth_norm = None
        nearest_fingertip_hand = None
        fingertips = []
        fingertip_mouth_threshold = pose_cfg.get("fingertip_to_mouth_distance_ratio", 0.55)
        fingertip_extension_ratio = pose_cfg.get("fingertip_extension_ratio", 0.38)
        if face_box is not None:
            hand_face = hand_to_face_evidence(
                kxy[i],
                kconf[i],
                face_box,
                norm,
                focus_keypoint_conf,
                hand_face_threshold,
            )
            hand_to_face_hit = hand_face["hit"]
            hand_face_ratio = hand_face["distance_ratio"]
            hand_face_px = hand_face["distance_px"]
            hand_face_norm = hand_face["norm"]
            nearest_hand = hand_face["nearest_hand"]
            hands = hand_face["hands"]
            fingertip_mouth = fingertip_to_mouth_evidence(
                kxy[i],
                kconf[i],
                mouth,
                face_box,
                norm,
                focus_keypoint_conf,
                fingertip_mouth_threshold,
                fingertip_extension_ratio,
            )
            fingertip_to_mouth_hit = fingertip_mouth["hit"]
            fingertip_mouth_ratio = fingertip_mouth["distance_ratio"]
            fingertip_mouth_px = fingertip_mouth["distance_px"]
            fingertip_mouth_norm = fingertip_mouth["norm"]
            nearest_fingertip_hand = fingertip_mouth["nearest_hand"]
            fingertips = fingertip_mouth["fingertips"]
            for hand in hands:
                matching = next((item for item in fingertips if item["side"] == hand["side"]), None)
                if matching is not None:
                    hand["wrist"] = matching["wrist"]
                    hand["fingertip"] = matching["fingertip"]
                    hand["fingertip_conf"] = matching["conf"]
                    hand["fingertip_method"] = matching["method"]
                    hand["fingertip_mouth_distance_px"] = matching["distance_px"]
                    hand["fingertip_mouth_distance_ratio"] = matching["distance_ratio"]
                    hand["fingertip_inside_mouth_zone"] = matching["inside_mouth_zone"]
                    hand["fingertip_mouth_hit"] = matching["hit"]
        wrist_hit_points = wrist_hits(
            kxy[i],
            kconf[i],
            mouth,
            norm,
            keypoint_conf,
            pose_cfg["hand_to_mouth_distance_ratio"],
        )
        x1, y1, x2, y2 = box.astype(int).tolist()
        people.append(
            PersonEvidence(
                box=(x1, y1, x2, y2),
                mouth=(int(mouth[0]), int(mouth[1])),
                face_box=face_box,
                face_center=tuple(box_center(face_box).astype(int).tolist()) if face_box is not None else None,
                hand_to_face_hit=hand_to_face_hit,
                fingertip_to_mouth_hit=fingertip_to_mouth_hit,
                hand_hit=bool(wrist_hit_points),
                cigarette_hit=False,
                smoke_hit=False,
                hands=hands,
                nearest_hand=nearest_hand,
                nearest_fingertip_hand=nearest_fingertip_hand,
                hand_face_distance_px=hand_face_px,
                hand_face_distance_ratio=hand_face_ratio,
                hand_face_threshold=hand_face_threshold,
                hand_face_norm=hand_face_norm,
                fingertip_mouth_distance_px=fingertip_mouth_px,
                fingertip_mouth_distance_ratio=fingertip_mouth_ratio,
                fingertip_mouth_threshold=fingertip_mouth_threshold,
                fingertip_mouth_norm=fingertip_mouth_norm,
                detail_trigger=False,
                detail_crop_box=None,
                detail_crop_path=None,
                detail_model_status="not_triggered",
                detail_model_hit=False,
                detail_model_detections=[],
                depth_status="not_triggered",
                depth_consistent=None,
                depth_delta=None,
                depth_delta_ratio=None,
                depth_threshold=None,
                depth_mouth=None,
                depth_hand=None,
                depth_debug_path=None,
                score=0.0,
                label="person",
            )
        )

    cigarette_hits = detect_cigarette_hits(
        cigarette_model,
        frame,
        people,
        det_cfg["cigarette_conf"],
        imgsz,
        device,
    )

    for person, cigarette_hit in zip(people, cigarette_hits):
        person.cigarette_hit = cigarette_hit
        person.score = (
            (0.65 if person.hand_hit else 0.0)
            + (0.50 if person.fingertip_to_mouth_hit and not person.hand_hit else 0.0)
            + (0.35 if person.cigarette_hit else 0.0)
            + (
                0.30
                if person.hand_to_face_hit and not person.hand_hit and not person.fingertip_to_mouth_hit
                else 0.0
            )
        )
        if person.cigarette_hit and person.hand_hit:
            person.label = "smoking_evidence"
        elif person.hand_hit:
            person.label = "suspected_smoking_pose"
        elif person.fingertip_to_mouth_hit:
            person.label = "focus_fingertip_to_mouth"
        elif person.cigarette_hit:
            person.label = "suspected_smoking_object"
        elif person.hand_to_face_hit:
            person.label = "focus_hand_to_face"

    return people


def draw_overlay(
    frame: np.ndarray,
    people: list[PersonEvidence],
    temporal_score: float,
    alarm: bool,
    frame_idx: int,
) -> np.ndarray:
    annotated = frame.copy()
    color_alarm = (0, 0, 255)
    color_suspect = (0, 165, 255)
    color_person = (60, 180, 75)
    color_smoke = (255, 180, 60)

    for person in people:
        color = color_person
        display_label = "CLEAR"
        if person.label in ("clear_depth_conflict", "clear_depth_unconfirmed"):
            display_label = "CLEAR_Z"
        if person.label == "warning_depth_match":
            display_label = "WARNING"
            color = color_suspect
        if person.label.startswith("suspected"):
            display_label = "REVIEW"
            color = color_suspect
        if person.label == "smoking_evidence":
            display_label = "SMOKING_MODEL_HIT"
            color = color_alarm

        x1, y1, x2, y2 = person.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if person.mouth is not None:
            cv2.circle(annotated, person.mouth, 5, (255, 0, 255), -1)
        if person.face_box is not None:
            fx1, fy1, fx2, fy2 = person.face_box
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 255), 1)
        if person.face_center is not None:
            cv2.circle(annotated, person.face_center, 4, (255, 0, 255), -1)
        nearest_point = None
        nearest_fingertip = None
        for hand in person.hands:
            hx, hy = hand["point"]
            hand_color = (0, 255, 255) if hand["hit"] else (180, 180, 180)
            radius = 6 if hand["side"] == person.nearest_hand else 4
            cv2.circle(annotated, (hx, hy), radius, hand_color, -1)
            cv2.putText(
                annotated,
                hand["side"][0].upper(),
                (hx + 6, hy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                hand_color,
                1,
                cv2.LINE_AA,
            )
            if hand["side"] == person.nearest_hand:
                nearest_point = (hx, hy)
            fingertip = hand.get("fingertip")
            if fingertip is not None:
                tip_color = (255, 255, 0) if hand.get("fingertip_mouth_hit") else (200, 200, 120)
                tx, ty = fingertip
                cv2.circle(annotated, (tx, ty), 5, tip_color, -1)
                cv2.putText(
                    annotated,
                    f"{hand['side'][0].upper()}tip",
                    (tx + 6, ty + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    tip_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(annotated, (hx, hy), (tx, ty), tip_color, 1, cv2.LINE_AA)
                if hand["side"] == person.nearest_fingertip_hand:
                    nearest_fingertip = (tx, ty)
        if person.face_center is not None and nearest_point is not None:
            cv2.line(annotated, person.face_center, nearest_point, (0, 255, 255), 1, cv2.LINE_AA)
        if person.mouth is not None and nearest_fingertip is not None:
            cv2.line(annotated, person.mouth, nearest_fingertip, (255, 255, 0), 2, cv2.LINE_AA)
        ratio_text = ""
        if person.fingertip_mouth_distance_ratio is not None:
            hit_text = "NEAR" if person.fingertip_to_mouth_hit else "FAR"
            ratio_text = (
                f" tip:{person.nearest_fingertip_hand[0].upper()} "
                f"{person.fingertip_mouth_distance_ratio:.2f}<={person.fingertip_mouth_threshold:.2f} {hit_text}"
            )
        elif person.hand_face_distance_ratio is not None:
            hit_text = "HIT" if person.hand_to_face_hit else "NO"
            ratio_text = (
                f" face:{person.nearest_hand[0].upper()} "
                f"{person.hand_face_distance_ratio:.2f}<={person.hand_face_threshold:.2f} {hit_text}"
            )
        label_y = y1 - 8 if y1 > 92 else max(92, min(annotated.shape[0] - 48, y2 - 42))
        cv2.putText(
            annotated,
            f"{display_label} {person.score:.2f}{ratio_text}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        if person.fingertip_mouth_distance_px is not None and person.fingertip_mouth_norm is not None:
            cv2.putText(
                annotated,
                f"tip_px={person.fingertip_mouth_distance_px:.1f} norm={person.fingertip_mouth_norm:.1f}",
                (x1, min(annotated.shape[0] - 8, label_y + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        if person.hand_face_distance_px is not None and person.hand_face_norm is not None:
            cv2.putText(
                annotated,
                f"face_px={person.hand_face_distance_px:.1f} norm={person.hand_face_norm:.1f}",
                (x1, min(annotated.shape[0] - 8, label_y + 42)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        if person.depth_status not in ("not_triggered", "disabled"):
            if person.depth_delta_ratio is not None and person.depth_threshold is not None:
                depth_hit = "Z_MATCH" if person.depth_consistent else "Z_CONFLICT"
                depth_text = f"z_tip_mouth={person.depth_delta_ratio:.2f}<={person.depth_threshold:.2f} {depth_hit}"
            else:
                depth_text = f"depth={person.depth_status[:34]}"
            depth_color = color_suspect if person.depth_consistent else (0, 0, 255)
            if person.depth_consistent is None:
                depth_color = (180, 180, 180)
            cv2.putText(
                annotated,
                depth_text,
                (x1, min(annotated.shape[0] - 8, label_y + 62)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                depth_color,
                1,
                cv2.LINE_AA,
            )

        for detection in person.detail_model_detections:
            dx1, dy1, dx2, dy2 = [int(round(value)) for value in detection["global_xyxy"]]
            dx1 = max(0, min(annotated.shape[1] - 1, dx1))
            dx2 = max(0, min(annotated.shape[1] - 1, dx2))
            dy1 = max(0, min(annotated.shape[0] - 1, dy1))
            dy2 = max(0, min(annotated.shape[0] - 1, dy2))
            if dx2 <= dx1 or dy2 <= dy1:
                continue
            detection_color = color_smoke if detection.get("kind") == "smoke" else color_alarm
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), detection_color, 2)
            cv2.putText(
                annotated,
                f"{detection['class_name']} {detection['conf']:.2f}",
                (dx1, max(18, dy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                detection_color,
                2,
                cv2.LINE_AA,
            )

    status = "ALARM" if alarm else "monitoring"
    status_color = color_alarm if alarm else (255, 255, 255)
    cv2.rectangle(annotated, (8, 8), (430, 76), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"{status} smoking_score={temporal_score:.2f}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"frame={frame_idx}",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return annotated


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    source = args.source or resolve_path(cfg["video"]["source"])
    if source is None:
        raise SystemExit("Missing --source or video.source in config.")
    source = source.resolve()

    output_dir = args.output_dir or resolve_path(cfg["project"]["output_dir"]) / "smoking"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or cfg["device"]["type"]
    imgsz = args.imgsz or int(cfg["video"]["img_size"])
    pose_conf = args.conf or float(cfg["detection"]["person_conf"])
    pose_model_path = resolve_path(args.pose_model) or resolve_path(cfg["models"]["pose_model"])
    cigarette_model_path = resolve_path(args.cigarette_model) or resolve_path(cfg["models"]["cigarette_detector"])

    if pose_model_path is None or not pose_model_path.exists():
        raise SystemExit(f"Pose model not found: {pose_model_path}")

    cigarette_model = None
    cigarette_model_available = bool(cigarette_model_path and cigarette_model_path.exists())
    if cigarette_model_available:
        cigarette_model = YOLO(str(cigarette_model_path))

    info = video_info(source)
    if not info["opened"]:
        raise SystemExit(f"Cannot open video: {source}")

    fps = info["fps"] if info["fps"] > 0 else 25.0
    sample_fps = max(1, int(cfg["video"]["sample_fps"]))
    stride = max(1, round(fps / sample_fps))
    window_size = max(1, int(float(cfg["temporal"]["window_seconds"]) * sample_fps))

    out_video = output_dir / f"{source.stem}_smoking.mp4"
    out_json = output_dir / f"{source.stem}_smoking.json"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info["width"], info["height"]),
    )

    cap = cv2.VideoCapture(str(source))
    pose_model = YOLO(str(pose_model_path))
    depth_estimator = DepthEstimator(cfg, device)
    events = []
    frame_details = []
    window = deque(maxlen=window_size)
    frame_idx = 0
    analyzed_frames = 0
    alarm_frames = 0
    person_frames = 0
    focus_hand_to_face_frames = 0
    fingertip_to_mouth_frames = 0
    hand_hit_frames = 0
    cigarette_hit_frames = 0
    smoke_hit_frames = 0
    detail_trigger_frames = 0
    detail_crops_saved = 0
    detail_model_hit_frames = 0
    depth_checked_frames = 0
    depth_consistent_frames = 0
    depth_conflict_frames = 0
    depth_pending_frames = 0
    last_people: list[PersonEvidence] = []
    last_temporal_score = 0.0
    last_alarm = False
    started = perf_counter()

    threshold = float(cfg["alarm"].get("pose_only_score_threshold", cfg["alarm"]["smoking_score_threshold"]))
    if cigarette_model_available:
        threshold = float(cfg["alarm"].get("detail_model_score_threshold", cfg["alarm"]["smoking_score_threshold"]))

    min_hand_hits = int(cfg["temporal"]["min_hand_to_mouth_hits"])
    min_cigarette_hits = int(cfg["temporal"]["min_cigarette_hits"])
    min_detail_trigger_hits = int(cfg["temporal"].get("min_detail_trigger_hits", 1))
    min_object_detail_ratio = float(cfg["temporal"].get("min_object_detail_ratio", 0.0))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            last_people = analyze_frame(pose_model, None, frame, cfg, device, imgsz, pose_conf)
            detail_crops_saved = run_detail_model_on_focus(
                frame,
                last_people,
                cigarette_model,
                depth_estimator,
                cfg,
                output_dir,
                frame_idx,
                detail_crops_saved,
                device,
                imgsz,
            )
            focus_hand_to_face = any(person.hand_to_face_hit for person in last_people)
            fingertip_to_mouth = any(person.fingertip_to_mouth_hit for person in last_people)
            hand_hit = any(person.hand_hit for person in last_people)
            cigarette_hit = any(person.cigarette_hit for person in last_people)
            smoke_hit = any(person.smoke_hit for person in last_people)
            detail_trigger = any(person.detail_trigger for person in last_people)
            detail_model_hit = any(person.detail_model_hit for person in last_people)
            depth_checked = any(person.depth_status in ("ok", "ok_low_contrast") for person in last_people)
            depth_consistent = any(person.depth_consistent is True for person in last_people)
            depth_conflict = any(person.depth_consistent is False for person in last_people)
            depth_pending = any(
                person.detail_trigger and person.depth_status not in ("ok", "ok_low_contrast", "disabled")
                for person in last_people
            )
            current_score = max((person.score for person in last_people), default=0.0)
            if last_people:
                person_frames += 1
            if focus_hand_to_face:
                focus_hand_to_face_frames += 1
            if fingertip_to_mouth:
                fingertip_to_mouth_frames += 1
            if hand_hit:
                hand_hit_frames += 1
            if cigarette_hit:
                cigarette_hit_frames += 1
            if smoke_hit:
                smoke_hit_frames += 1
            if detail_trigger:
                detail_trigger_frames += 1
            if detail_model_hit:
                detail_model_hit_frames += 1
            if depth_checked:
                depth_checked_frames += 1
            if depth_consistent:
                depth_consistent_frames += 1
            if depth_conflict:
                depth_conflict_frames += 1
            if depth_pending:
                depth_pending_frames += 1
            frame_record = {
                "frame": frame_idx,
                    "time": round(frame_idx / fps, 3),
                    "focus_hand_to_face": focus_hand_to_face,
                    "fingertip_to_mouth": fingertip_to_mouth,
                    "hand_hit": hand_hit,
                "cigarette_hit": cigarette_hit,
                "smoke_hit": smoke_hit,
                "detail_trigger": detail_trigger,
                "detail_model_hit": detail_model_hit,
                "depth_checked": depth_checked,
                "depth_consistent": depth_consistent,
                "depth_conflict": depth_conflict,
                "depth_pending": depth_pending,
                "score": current_score,
                "people": [
                    {
                        "box": person.box,
                        "mouth": person.mouth,
                        "face_box": person.face_box,
                        "face_center": person.face_center,
                        "hand_to_face_hit": person.hand_to_face_hit,
                        "fingertip_to_mouth_hit": person.fingertip_to_mouth_hit,
                        "hand_hit": person.hand_hit,
                        "cigarette_hit": person.cigarette_hit,
                        "smoke_hit": person.smoke_hit,
                        "nearest_hand": person.nearest_hand,
                        "nearest_fingertip_hand": person.nearest_fingertip_hand,
                        "hand_face_distance_px": (
                            round(person.hand_face_distance_px, 3)
                            if person.hand_face_distance_px is not None
                            else None
                        ),
                        "hand_face_norm": (
                            round(person.hand_face_norm, 3) if person.hand_face_norm is not None else None
                        ),
                        "hand_face_distance_ratio": (
                            round(person.hand_face_distance_ratio, 3)
                            if person.hand_face_distance_ratio is not None
                            else None
                        ),
                        "hand_face_threshold": round(person.hand_face_threshold, 3),
                        "hand_face_calculation": (
                            f"{person.hand_face_distance_px:.3f} / {person.hand_face_norm:.3f} = "
                            f"{person.hand_face_distance_ratio:.3f}"
                            if (
                                person.hand_face_distance_px is not None
                                and person.hand_face_norm is not None
                                and person.hand_face_distance_ratio is not None
                            )
                            else None
                        ),
                        "fingertip_mouth_distance_px": (
                            round(person.fingertip_mouth_distance_px, 3)
                            if person.fingertip_mouth_distance_px is not None
                            else None
                        ),
                        "fingertip_mouth_norm": (
                            round(person.fingertip_mouth_norm, 3)
                            if person.fingertip_mouth_norm is not None
                            else None
                        ),
                        "fingertip_mouth_distance_ratio": (
                            round(person.fingertip_mouth_distance_ratio, 3)
                            if person.fingertip_mouth_distance_ratio is not None
                            else None
                        ),
                        "fingertip_mouth_threshold": round(person.fingertip_mouth_threshold, 3),
                        "fingertip_mouth_calculation": (
                            f"{person.fingertip_mouth_distance_px:.3f} / {person.fingertip_mouth_norm:.3f} = "
                            f"{person.fingertip_mouth_distance_ratio:.3f}"
                            if (
                                person.fingertip_mouth_distance_px is not None
                                and person.fingertip_mouth_norm is not None
                                and person.fingertip_mouth_distance_ratio is not None
                            )
                            else None
                        ),
                        "detail_trigger": person.detail_trigger,
                        "detail_crop_box": person.detail_crop_box,
                        "detail_crop_path": person.detail_crop_path,
                        "detail_model_status": person.detail_model_status,
                        "detail_model_hit": person.detail_model_hit,
                        "detail_model_detections": person.detail_model_detections,
                        "depth_status": person.depth_status,
                        "depth_consistent": person.depth_consistent,
                        "depth_mouth": round(person.depth_mouth, 6) if person.depth_mouth is not None else None,
                        "depth_hand": round(person.depth_hand, 6) if person.depth_hand is not None else None,
                        "depth_delta": round(person.depth_delta, 6) if person.depth_delta is not None else None,
                        "depth_delta_ratio": (
                            round(person.depth_delta_ratio, 6) if person.depth_delta_ratio is not None else None
                        ),
                        "depth_threshold": person.depth_threshold,
                        "depth_debug_path": person.depth_debug_path,
                        "hands": [
                            {
                                "side": hand["side"],
                                "point": hand["point"],
                                "conf": round(hand["conf"], 3),
                                "distance_px": round(hand["distance_px"], 3),
                                "distance_ratio": round(hand["distance_ratio"], 3),
                                "inside_face_box": hand["inside_face_box"],
                                "hit": hand["hit"],
                                "wrist": hand.get("wrist"),
                                "fingertip": hand.get("fingertip"),
                                "fingertip_method": hand.get("fingertip_method"),
                                "fingertip_mouth_distance_px": (
                                    round(hand["fingertip_mouth_distance_px"], 3)
                                    if "fingertip_mouth_distance_px" in hand
                                    else None
                                ),
                                "fingertip_mouth_distance_ratio": (
                                    round(hand["fingertip_mouth_distance_ratio"], 3)
                                    if "fingertip_mouth_distance_ratio" in hand
                                    else None
                                ),
                                "fingertip_inside_mouth_zone": hand.get("fingertip_inside_mouth_zone"),
                                "fingertip_mouth_hit": hand.get("fingertip_mouth_hit"),
                            }
                            for hand in person.hands
                        ],
                        "score": round(person.score, 3),
                        "label": person.label,
                    }
                    for person in last_people
                ],
            }
            window.append(frame_record)
            frame_details.append(frame_record)

            hand_hits = sum(1 for item in window if item["hand_hit"])
            focus_hits = sum(1 for item in window if item["focus_hand_to_face"])
            fingertip_hits = sum(1 for item in window if item["fingertip_to_mouth"])
            cigarette_hits = sum(1 for item in window if item["cigarette_hit"])
            smoke_hits = sum(1 for item in window if item["smoke_hit"])
            detail_hits = sum(1 for item in window if item["detail_trigger"])
            depth_consistent_hits = sum(1 for item in window if item["depth_consistent"])
            hand_ratio = hand_hits / len(window)
            focus_ratio = focus_hits / len(window)
            fingertip_ratio = fingertip_hits / len(window)
            cigarette_ratio = cigarette_hits / len(window)
            smoke_ratio = smoke_hits / len(window)
            if cigarette_model_available:
                warning_ratio = detail_hits / len(window)
                depth_ratio = depth_consistent_hits / len(window)
                object_hits = cigarette_hits + smoke_hits
                object_detail_ratio = object_hits / max(detail_hits, 1)
                visual_ratio = max(cigarette_ratio, smoke_ratio)
                last_temporal_score = 0.20 * warning_ratio + 0.20 * depth_ratio + 0.60 * visual_ratio
                last_alarm = (
                    detail_hits >= min_detail_trigger_hits
                    and object_hits >= min_cigarette_hits
                    and object_detail_ratio >= min_object_detail_ratio
                    and last_temporal_score >= threshold
                )
            else:
                object_hits = cigarette_hits
                object_detail_ratio = 0.0
                last_temporal_score = (
                    0.42 * hand_ratio + 0.30 * fingertip_ratio + 0.12 * focus_ratio + 0.16 * cigarette_ratio
                )
                last_alarm = hand_hits >= min_hand_hits and last_temporal_score >= threshold

            if last_alarm:
                alarm_frames += 1
                events.append(
                    {
                        "frame": frame_idx,
                        "time": round(frame_idx / fps, 3),
                        "score": round(last_temporal_score, 3),
                        "focus_hand_to_face_hits_in_window": focus_hits,
                        "fingertip_to_mouth_hits_in_window": fingertip_hits,
                        "hand_hits_in_window": hand_hits,
                        "cigarette_hits_in_window": cigarette_hits,
                        "smoke_hits_in_window": smoke_hits,
                        "object_hits_in_window": object_hits,
                        "object_detail_ratio_in_window": round(object_detail_ratio, 3),
                        "detail_trigger_hits_in_window": detail_hits,
                        "depth_consistent_hits_in_window": depth_consistent_hits,
                        "score_threshold": threshold,
                        "mode": "pose_object" if cigarette_model_available else "pose_only",
                    }
                )
            analyzed_frames += 1

        annotated = draw_overlay(frame, last_people, last_temporal_score, last_alarm, frame_idx)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = perf_counter() - started

    summary = {
        "source": str(source),
        "output_video": str(out_video),
        "output_json": str(out_json),
        "device": device,
        "pose_model": str(pose_model_path),
        "cigarette_model": str(cigarette_model_path) if cigarette_model_available else None,
        "mode": "pose_object" if cigarette_model_available else "pose_only",
        "frames": frame_idx,
        "analyzed_frames": analyzed_frames,
        "person_frames": person_frames,
        "focus_hand_to_face_frames": focus_hand_to_face_frames,
        "fingertip_to_mouth_frames": fingertip_to_mouth_frames,
        "hand_hit_frames": hand_hit_frames,
        "cigarette_hit_frames": cigarette_hit_frames,
        "smoke_hit_frames": smoke_hit_frames,
        "detail_trigger_frames": detail_trigger_frames,
        "detail_crops_saved": detail_crops_saved,
        "detail_model_hit_frames": detail_model_hit_frames,
        "depth_enabled": bool(cfg.get("depth", {}).get("enabled", False)),
        "depth_model": cfg.get("depth", {}).get("model"),
        "depth_checked_frames": depth_checked_frames,
        "depth_consistent_frames": depth_consistent_frames,
        "depth_conflict_frames": depth_conflict_frames,
        "depth_pending_frames": depth_pending_frames,
        "alarm_frames": alarm_frames,
        "events": events,
        "frame_details": frame_details,
        "elapsed_seconds": round(elapsed, 3),
        "avg_input_fps": round(frame_idx / elapsed, 3) if elapsed > 0 else 0,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    console_summary = {key: value for key, value in summary.items() if key != "frame_details"}
    console_summary["frame_details_count"] = len(frame_details)
    print(json.dumps(console_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
