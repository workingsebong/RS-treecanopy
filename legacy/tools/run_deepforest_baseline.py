#!/usr/bin/env python3
"""Run the pretrained DeepForest tree detector on the test tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval/images_geotiff/test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/pretrained_baselines/deepforest"


def box_to_geometry(transform, xmin: float, ymin: float, xmax: float, ymax: float):
    left, top = transform * (xmin, ymin)
    right, bottom = transform * (xmax, ymax)
    return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def draw_preview(image_path: Path, predictions: pd.DataFrame, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    for idx, row in predictions.iterrows():
        xy = [row.xmin, row.ymin, row.xmax, row.ymax]
        draw.rectangle(xy, outline=(255, 140, 0, 255), width=3)
        score = row.get("score", None)
        if score is not None:
            label = f"{score:.2f}"
            text_xy = (row.xmin + 2, row.ymin + 2)
            bbox = draw.textbbox(text_xy, label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 150))
            draw.text(text_xy, label, fill=(255, 220, 120, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def predict_one(model, image_path: Path, min_score: float) -> pd.DataFrame:
    result = model.predict_image(path=str(image_path))
    if result is None or result.empty:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

    result = pd.DataFrame(result).copy()
    if "score" in result.columns:
        result = result[result["score"] >= min_score].copy()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pretrained DeepForest baseline.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from deepforest import main as deepforest_main

    output_dir = args.output_dir.resolve()
    csv_dir = output_dir / "csv"
    preview_dir = output_dir / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    config_args = {"score_thresh": args.min_score}
    if args.device != "auto":
        config_args.update(
            {
                "accelerator": "gpu" if args.device == "cuda" else "cpu",
                "devices": 1,
            }
        )

    model = deepforest_main.deepforest(config_args=config_args)

    all_rows = []
    all_features = []
    image_paths = sorted(args.input_dir.glob("*.tif"))
    if not image_paths:
        raise RuntimeError(f"No GeoTIFF files found in {args.input_dir}")

    for image_path in image_paths:
        predictions = predict_one(model, image_path, min_score=args.min_score)
        image_stem = image_path.stem
        predictions.insert(0, "image", image_path.name)
        predictions.to_csv(csv_dir / f"{image_stem}.csv", index=False)
        draw_preview(image_path, predictions, preview_dir / f"{image_stem}.jpg")

        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs

        for _, row in predictions.iterrows():
            geometry = box_to_geometry(transform, row.xmin, row.ymin, row.xmax, row.ymax)
            record = row.to_dict()
            record["geometry"] = geometry
            all_features.append(record)
            all_rows.append({k: v for k, v in record.items() if k != "geometry"})

        print(f"{image_path.name}: {len(predictions)} boxes")

    merged_csv = output_dir / "deepforest_predictions.csv"
    pd.DataFrame(all_rows).to_csv(merged_csv, index=False)

    merged_geojson = output_dir / "deepforest_predictions.geojson"
    if all_features:
        gdf = gpd.GeoDataFrame(all_features, geometry="geometry", crs=crs)
        gdf.to_file(merged_geojson, driver="GeoJSON")
    else:
        gpd.GeoDataFrame(columns=["image", "label", "score", "geometry"], geometry="geometry", crs=crs).to_file(
            merged_geojson,
            driver="GeoJSON",
        )

    summary = pd.DataFrame(
        [
            {
                "model": "DeepForest pretrained",
                "tiles": len(image_paths),
                "predictions": len(all_rows),
                "min_score": args.min_score,
                "csv": str(merged_csv.relative_to(PROJECT_ROOT)),
                "geojson": str(merged_geojson.relative_to(PROJECT_ROOT)),
                "preview_dir": str(preview_dir.relative_to(PROJECT_ROOT)),
            }
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(f"Saved CSV: {merged_csv}")
    print(f"Saved GeoJSON: {merged_geojson}")
    print(f"Saved previews: {preview_dir}")


if __name__ == "__main__":
    main()
