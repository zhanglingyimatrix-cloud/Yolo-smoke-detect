import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "data" / "videos" / "smoking_samples" / "images"
DEFAULT_VIDEO = ROOT / "data" / "videos" / "smoking_samples" / "roboflow_smoking_preview.mp4"

SAMPLE_IMAGES = [
    {
        "name": "smoking_drinking_example_1.jpg",
        "url": "https://source.roboflow.com/9T3KVbvOd0uvf5BZ762C/05KDNfWek4VM6G5uWIc4/thumb.jpg",
        "source": "Roboflow Smoking and Drinking Detection example",
    },
    {
        "name": "smoking_drinking_example_2.jpg",
        "url": "https://source.roboflow.com/9T3KVbvOd0uvf5BZ762C/0AcAu6y82TiPjzdcpNfR/thumb.jpg",
        "source": "Roboflow Smoking and Drinking Detection example",
    },
    {
        "name": "smoking_drinking_example_3.jpg",
        "url": "https://source.roboflow.com/9T3KVbvOd0uvf5BZ762C/0EKIXghXmeGJgBF5qcLX/thumb.jpg",
        "source": "Roboflow Smoking and Drinking Detection example",
    },
    {
        "name": "cigarette_detection_example_1.jpg",
        "url": "https://source.roboflow.com/kP9Y6LaUXqSJSUopszQHmEDQ7yN2/06EZnILOy4jsvQAgO5Zb/thumb.jpg",
        "source": "Roboflow Smoking Detection Dataset example",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public smoking sample images and compose an MP4.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--seconds-per-image", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    return parser.parse_args()


def download_file(url: str, output: Path) -> bool:
    if output.exists() and output.stat().st_size > 0:
        return True

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            output.write_bytes(response.read())
        return output.stat().st_size > 0
    except Exception as exc:
        print(f"WARNING: failed to download {url}: {exc}", flush=True)
        return False


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = int(src_w * scale)
    resized_h = int(src_h * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def read_image(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def animated_frame(image: np.ndarray, frame_i: int, frames_per_image: int, width: int, height: int) -> np.ndarray:
    base = letterbox(image, width, height)
    zoom = 1.0 + 0.08 * (frame_i / max(1, frames_per_image - 1))
    crop_w = int(width / zoom)
    crop_h = int(height / zoom)
    x = int((width - crop_w) * frame_i / max(1, frames_per_image - 1))
    y = int((height - crop_h) * 0.5)
    cropped = base[y : y + crop_h, x : x + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output = args.output.resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for item in SAMPLE_IMAGES:
        path = image_dir / item["name"]
        if download_file(item["url"], path):
            downloaded.append({**item, "path": str(path)})

    images = []
    for item in downloaded:
        image = read_image(item["path"])
        if image is None:
            print(f"WARNING: cannot open image {item['path']}", flush=True)
            continue
        images.append((item, image))

    if not images:
        raise SystemExit("No valid images downloaded.")

    frames_per_image = max(1, int(round(args.fps * args.seconds_per_image)))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )

    for item, image in images:
        for frame_i in range(frames_per_image):
            frame = animated_frame(image, frame_i, frames_per_image, args.width, args.height)
            cv2.putText(
                frame,
                item["source"],
                (16, args.height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)

    writer.release()
    manifest = {
        "output_video": str(output),
        "images": downloaded,
        "fps": args.fps,
        "seconds_per_image": args.seconds_per_image,
        "size": [args.width, args.height],
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
