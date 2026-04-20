#!/usr/bin/env python3
"""Prepare test inputs for pretrained tree-crown baselines."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_INDEX = PROJECT_ROOT / "data/processed/yolo_seg_512_random_seoul_urban/split_index.csv"
DEFAULT_PATCH_DIR = PROJECT_ROOT / "data/raw/random_seoul_urban/patches"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/processed/yolo_seg_512_random_seoul_urban/images"
DEFAULT_LABEL_GEOJSON_DIR = PROJECT_ROOT / "data/processed/labels_512_random_seoul_urban/geojson"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval"

TILE_RE = re.compile(r"_r(?P<row>\d+)_c(?P<col>\d+)$")


def parse_tile_row_col(tile_id: str) -> tuple[int, int]:
    match = TILE_RE.search(tile_id)
    if not match:
        raise ValueError(f"Could not parse row/col from tile_id: {tile_id}")
    return int(match.group("row")), int(match.group("col"))


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_geotiff_tile(source_patch: Path, target: Path, row: int, col: int, tile_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_patch) as src:
        row_off = row * tile_size
        col_off = col * tile_size
        if row_off + tile_size > src.height or col_off + tile_size > src.width:
            raise ValueError(
                f"Tile window r{row} c{col} exceeds patch bounds for {source_patch.name}: "
                f"{src.width}x{src.height}"
            )

        window = Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size)
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            {
                "height": tile_size,
                "width": tile_size,
                "transform": src.window_transform(window),
                "driver": "GTiff",
            }
        )

        with rasterio.open(target, "w", **profile) as dst:
            dst.write(data)


def load_split_rows(split_index: Path, split: str) -> list[dict[str, str]]:
    with split_index.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == split]
    if not rows:
        raise RuntimeError(f"No rows found for split={split!r} in {split_index}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pretrained baseline evaluation inputs.")
    parser.add_argument("--split-index", type=Path, default=DEFAULT_SPLIT_INDEX)
    parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-geojson-dir", type=Path, default=DEFAULT_LABEL_GEOJSON_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    images_png_dir = output_dir / "images_png" / args.split
    images_geotiff_dir = output_dir / "images_geotiff" / args.split
    labels_dir = output_dir / "ground_truth_geojson" / args.split
    manifest_path = output_dir / f"{args.split}_manifest.csv"

    rows = load_split_rows(args.split_index, args.split)
    manifest_rows: list[dict[str, str]] = []

    for row in rows:
        image_name = row["image"]
        image_stem = Path(image_name).stem
        patch_id = row["patch_id"]
        tile_row, tile_col = parse_tile_row_col(row["tile_id"])

        source_png = args.image_dir / args.split / image_name
        source_patch = args.patch_dir / f"{patch_id}.tif"
        source_geojson = args.label_geojson_dir / f"{image_stem}.geojson"

        target_png = images_png_dir / image_name
        target_tif = images_geotiff_dir / f"{image_stem}.tif"
        target_geojson = labels_dir / source_geojson.name

        for source in [source_png, source_patch, source_geojson]:
            if not source.exists():
                raise FileNotFoundError(source)

        copy_file(source_png, target_png)
        copy_file(source_geojson, target_geojson)
        write_geotiff_tile(source_patch, target_tif, tile_row, tile_col, args.tile_size)

        manifest_rows.append(
            {
                "image": image_name,
                "patch_id": patch_id,
                "tile_id": row["tile_id"],
                "shape_count": row["shape_count"],
                "png": str(target_png.relative_to(PROJECT_ROOT)),
                "geotiff": str(target_tif.relative_to(PROJECT_ROOT)),
                "ground_truth_geojson": str(target_geojson.relative_to(PROJECT_ROOT)),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "patch_id",
                "tile_id",
                "shape_count",
                "png",
                "geotiff",
                "ground_truth_geojson",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_instances = sum(int(row["shape_count"]) for row in manifest_rows)
    print(f"Prepared {len(manifest_rows)} {args.split} tiles with {total_instances} GT polygons")
    print(f"PNG tiles: {images_png_dir}")
    print(f"GeoTIFF tiles: {images_geotiff_dir}")
    print(f"Ground truth: {labels_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
