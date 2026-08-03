import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO cigarette+smoke detail dataset.")
    parser.add_argument("--pyro-root", type=Path, default=Path("data/raw/pyro_sdis"))
    parser.add_argument("--cigarette-root", type=Path, default=Path("data/datasets/cigarette"))
    parser.add_argument("--output-root", type=Path, default=Path("data/datasets/smoking_detail"))
    parser.add_argument("--include-cigarette", action="store_true", default=True)
    parser.add_argument("--no-cigarette", dest="include_cigarette", action="store_false")
    parser.add_argument("--max-smoke-train", type=int, default=None)
    parser.add_argument("--max-smoke-val", type=int, default=None)
    return parser.parse_args()


def reset_split_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            path = root / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


def copy_cigarette_split(src_root: Path, out_root: Path, split: str) -> tuple[int, int]:
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    if not src_images.exists() or not src_labels.exists():
        return 0, 0

    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    image_count = 0
    label_count = 0

    for image in sorted(src_images.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = src_labels / f"{image.stem}.txt"
        if not label.exists():
            continue
        dst_name = f"cigdet_{image.name}"
        shutil.copy2(image, out_images / dst_name)
        shutil.copy2(label, out_labels / f"cigdet_{image.stem}.txt")
        image_count += 1
        label_count += 1

    return image_count, label_count


def normalize_smoke_annotations(text: str | None) -> str:
    lines = []
    for raw_line in (text or "").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        # Pyro-SDIS stores smoke as class 1. Keep class 1 for smoking_detail.yaml.
        _, x, y, w, h = parts
        lines.append(f"1 {x} {y} {w} {h}")
    return "\n".join(lines) + ("\n" if lines else "")


def export_pyro_split(pyro_root: Path, out_root: Path, split: str, max_images: int | None = None) -> tuple[int, int]:
    files = sorted((pyro_root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Pyro-SDIS parquet files found for split '{split}' under {pyro_root / 'data'}")

    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    image_count = 0
    smoke_label_count = 0

    for parquet_path in files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=["image", "annotations", "image_name"], batch_size=128):
            for row in batch.to_pylist():
                if max_images is not None and image_count >= max_images:
                    return image_count, smoke_label_count

                image = row["image"] or {}
                image_bytes = image.get("bytes")
                image_name = row.get("image_name") or image.get("path")
                if not image_bytes or not image_name:
                    continue

                suffix = Path(image_name).suffix.lower() or ".jpg"
                stem = Path(image_name).stem
                dst_stem = f"pyro_{stem}"
                (out_images / f"{dst_stem}{suffix}").write_bytes(image_bytes)

                label_text = normalize_smoke_annotations(row.get("annotations"))
                (out_labels / f"{dst_stem}.txt").write_text(label_text, encoding="utf-8")
                image_count += 1
                if label_text.strip():
                    smoke_label_count += 1

    return image_count, smoke_label_count


def main() -> None:
    args = parse_args()
    out = args.output_root
    reset_split_dirs(out)

    counts = {}
    if args.include_cigarette:
        for split in ("train", "val", "test"):
            counts[f"cigarette_{split}"] = copy_cigarette_split(args.cigarette_root, out, split)

    counts["smoke_train"] = export_pyro_split(args.pyro_root, out, "train", args.max_smoke_train)
    counts["smoke_val"] = export_pyro_split(args.pyro_root, out, "val", args.max_smoke_val)

    source = """Smoking detail YOLO dataset

Classes:
0 cigarette
1 smoke

Cigarette source:
CigDet, https://data.mendeley.com/datasets/6hyrr8typ7/1, CC BY 4.0

Smoke source:
pyronear/pyro-sdis, https://huggingface.co/datasets/pyronear/pyro-sdis, Apache-2.0

Notes:
Negative samples use intentionally empty YOLO label files.
Pyro-SDIS smoke labels are kept as class id 1 for configs/smoking_detail.yaml.
"""
    (out / "SOURCE_smoking_detail.txt").write_text(source, encoding="utf-8")

    for split in ("train", "val", "test"):
        images = len([p for p in (out / "images" / split).rglob("*") if p.is_file()])
        label_files = [p for p in (out / "labels" / split).rglob("*") if p.is_file()]
        labels = len(label_files)
        empty = sum(1 for p in label_files if p.stat().st_size == 0)
        print(f"{split}: images={images} labels={labels} empty_negative_labels={empty}")

    for key, value in counts.items():
        print(f"{key}: images={value[0]} labeled_images={value[1]}")


if __name__ == "__main__":
    main()
