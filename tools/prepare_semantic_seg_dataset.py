#!/usr/bin/env python3
"""Convert YOLO polygon labels into binary semantic canopy masks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_DIR = PROJECT_ROOT / "data/processed/yolo_seg_512_random_seoul_urban"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/semantic_seg_512_random_seoul_urban"


def yolo_polygons_to_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return mask

    with label_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        values = line.split()
        if len(values) < 7:
            continue
        coords = np.array([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        points = np.rint(coords).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 1)

    return mask


def render_overlay(image_path: Path, mask: np.ndarray, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 3:
            continue
        points = [tuple(map(int, point[0])) for point in contour]
        draw.polygon(points, fill=(255, 140, 0, 85))
        draw.line(points + [points[0]], fill=(255, 120, 0, 240), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path, quality=92)


def process_split(input_dir: Path, output_dir: Path, split: str, render_previews: bool) -> dict[str, float | int]:
    image_dir = input_dir / "images" / split
    label_dir = input_dir / "labels" / split
    out_image_dir = output_dir / "images" / split
    out_mask_dir = output_dir / "masks" / split
    out_preview_dir = output_dir / "previews" / split

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    if render_previews:
        out_preview_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    total_pixels = 0
    canopy_pixels = 0
    empty_count = 0

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = label_dir / f"{image_path.stem}.txt"
        mask = yolo_polygons_to_mask(label_path, width=width, height=height)

        target_image = out_image_dir / image_path.name
        target_mask = out_mask_dir / f"{image_path.stem}.png"
        shutil.copy2(image_path, target_image)
        Image.fromarray(mask, mode="L").save(target_mask)

        if render_previews:
            render_overlay(image_path, mask, out_preview_dir / image_path.name)

        total_pixels += width * height
        canopy_pixels += int(mask.sum())
        empty_count += int(mask.sum() == 0)

    return {
        "images": len(image_paths),
        "empty_masks": empty_count,
        "total_pixels": total_pixels,
        "canopy_pixels": canopy_pixels,
        "canopy_ratio": canopy_pixels / total_pixels if total_pixels else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare semantic segmentation masks from YOLO polygons.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_YOLO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ["train", "val", "test"]:
        summary[split] = process_split(input_dir, output_dir, split, render_previews=not args.no_previews)

    total_pixels = sum(item["total_pixels"] for item in summary.values())
    canopy_pixels = sum(item["canopy_pixels"] for item in summary.values())
    summary["all"] = {
        "images": sum(item["images"] for item in summary.values()),
        "empty_masks": sum(item["empty_masks"] for item in summary.values()),
        "total_pixels": total_pixels,
        "canopy_pixels": canopy_pixels,
        "canopy_ratio": canopy_pixels / total_pixels if total_pixels else 0.0,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for split, item in summary.items():
        print(
            f"{split}: {item['images']} images, {item['empty_masks']} empty, "
            f"canopy ratio {item['canopy_ratio']:.4f}"
        )
    print(f"Saved semantic dataset: {output_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
