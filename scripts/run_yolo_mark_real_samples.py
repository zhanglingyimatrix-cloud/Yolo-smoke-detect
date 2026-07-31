import json
import os
import argparse
from pathlib import Path
from time import perf_counter

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "videos" / "real_samples" / "intel"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "outputs" / "yolo" / "real_samples" / "intel"

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "configs" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "configs" / "matplotlib"))
os.environ.setdefault("PYTHONUTF8", "1")

from ultralytics import YOLO


TASKS = [
    {
        "name": "detect",
        "model": ROOT / "yolo11n.pt",
        "conf": 0.35,
        "imgsz": 416,
    },
    {
        "name": "pose",
        "model": ROOT / "yolo11n-pose.pt",
        "conf": 0.25,
        "imgsz": 416,
    },
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detect and pose marking for mp4 samples.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=os.environ.get("YOLO_DEVICE", "cpu"))
    return parser.parse_args()


def run_task(model: YOLO, task: dict, video: Path, output_root: Path, device: str) -> dict:
    task_name = task["name"]
    out_dir = output_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{video.stem}_{task_name}.mp4"
    info = video_info(video)
    if not info["opened"]:
        return {
            "task": task_name,
            "input": str(video),
            "output": str(out_path),
            "ok": False,
            "error": "input video cannot be opened",
            "input_info": info,
        }

    existing = video_info(out_path) if out_path.exists() else {"opened": False, "frames": 0}
    if existing["opened"] and existing["frames"] > 0:
        return {
            "task": task_name,
            "input": str(video),
            "output": str(out_path),
            "ok": True,
            "skipped": True,
            "input_info": info,
            "output_info": existing,
            "frames_written": existing["frames"],
            "detection_frames": None,
            "total_instances": None,
            "elapsed_seconds": 0,
            "avg_fps": 0,
        }

    fps = info["fps"] if info["fps"] > 0 else 25.0
    size = (info["width"], info["height"])
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )

    started = perf_counter()
    frame_count = 0
    detection_frames = 0
    total_instances = 0

    results = model.predict(
        source=str(video),
        imgsz=task["imgsz"],
        conf=task["conf"],
        device=device,
        stream=True,
        verbose=False,
    )

    for result in results:
        plotted = result.plot()
        if plotted.shape[1] != size[0] or plotted.shape[0] != size[1]:
            plotted = cv2.resize(plotted, size)

        boxes = result.boxes
        instances = len(boxes) if boxes is not None else 0
        if instances:
            detection_frames += 1
            total_instances += instances

        writer.write(plotted)
        frame_count += 1

        if frame_count % 250 == 0:
            print(f"{task_name}: {video.name}: {frame_count}/{info['frames']} frames", flush=True)

    writer.release()
    elapsed = perf_counter() - started
    out_info = video_info(out_path)

    return {
        "task": task_name,
        "input": str(video),
        "output": str(out_path),
        "ok": out_info["opened"] and out_info["frames"] > 0,
        "input_info": info,
        "output_info": out_info,
        "frames_written": frame_count,
        "detection_frames": detection_frames,
        "total_instances": total_instances,
        "elapsed_seconds": round(elapsed, 3),
        "avg_fps": round(frame_count / elapsed, 3) if elapsed > 0 else 0,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files found in {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary = []

    print(f"Input directory: {input_dir}", flush=True)
    print(f"Output directory: {output_root}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Videos: {len(videos)}", flush=True)

    loaded_models = {}
    for task in TASKS:
        print(f"Loading {task['name']} model: {task['model']}", flush=True)
        loaded_models[task["name"]] = YOLO(str(task["model"]))

    for video in videos:
        print(f"Processing: {video.name}", flush=True)
        for task in TASKS:
            result = run_task(loaded_models[task["name"]], task, video, output_root, args.device)
            summary.append(result)
            status = "OK" if result["ok"] else "FAILED"
            skipped = " skipped=True" if result.get("skipped") else ""
            print(
                f"{status}: {task['name']} {video.name} -> {result['output']} "
                f"frames={result.get('frames_written', 0)} fps={result.get('avg_fps', 0)}{skipped}",
                flush=True,
            )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
