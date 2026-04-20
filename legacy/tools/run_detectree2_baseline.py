#!/usr/bin/env python3
"""Run a pretrained Detectree2 tree-crown model on the test tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import xy
from shapely.geometry import Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval/images_geotiff/test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/pretrained_baselines/detectree2"


def read_geotiff_as_bgr(image_path: Path):
    with rasterio.open(image_path) as src:
        data = src.read([1, 2, 3])
        transform = src.transform
        crs = src.crs

    image = np.transpose(data, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image[:, :, ::-1]), transform, crs


def mask_to_geometry(mask: np.ndarray, transform):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if len(contour) >= 3]
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None, None

    points = contour.reshape(-1, 2)
    xs, ys = xy(transform, points[:, 1], points[:, 0], offset="center")
    polygon = Polygon(zip(xs, ys))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None, None

    return polygon, points


def draw_preview(image_path: Path, detections: list[dict], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    for detection in detections:
        points = detection["pixel_points"]
        if points is not None and len(points) >= 3:
            xy_points = [tuple(map(float, point)) for point in points]
            draw.polygon(xy_points, fill=(255, 140, 0, 80), outline=(255, 120, 0, 240))

        xmin, ymin, xmax, ymax = detection["bbox"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 220, 255, 230), width=2)
        label = f"{detection['score']:.2f}"
        text_xy = (xmin + 2, ymin + 2)
        text_bbox = draw.textbbox(text_xy, label, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 150))
        draw.text(text_xy, label, fill=(255, 220, 120, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pretrained Detectree2 baseline.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-name", choices=["default", "paracou", "sepilok", "danum"], default="default")
    parser.add_argument("--confidence-threshold", type=float, default=0.15)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from samgeo.detectree2 import TreeCrownDelineator

    output_dir = args.output_dir.resolve()
    geojson_dir = output_dir / "geojson"
    preview_dir = output_dir / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = None if args.device == "auto" else args.device
    delineator = TreeCrownDelineator(
        model_path=str(args.model_path.resolve()) if args.model_path else None,
        model_name=args.model_name,
        device=device,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
    )
    model_label = f"Detectree2 {args.model_path.stem if args.model_path else args.model_name}"
    delineator._setup_predictor()
    predictor = delineator._predictor

    image_paths = sorted(args.input_dir.glob("*.tif"))
    if not image_paths:
        raise RuntimeError(f"No GeoTIFF files found in {args.input_dir}")

    all_rows = []
    all_features = []
    output_crs = None

    for image_path in image_paths:
        image_bgr, transform, crs = read_geotiff_as_bgr(image_path)
        output_crs = output_crs or crs
        outputs = predictor(image_bgr)
        instances = outputs["instances"].to("cpu")

        scores = instances.scores.numpy() if instances.has("scores") else np.array([])
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else np.empty((0, *image_bgr.shape[:2]))
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.empty((0, 4))

        detections = []
        features = []
        for instance_id, (score, mask, bbox) in enumerate(zip(scores, masks, boxes), start=1):
            if score < args.confidence_threshold:
                continue
            geometry, pixel_points = mask_to_geometry(mask, transform)
            if geometry is None:
                continue

            xmin, ymin, xmax, ymax = [float(value) for value in bbox]
            record = {
                "image": image_path.name,
                "instance_id": instance_id,
                "score": float(score),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "model": model_label,
            }
            features.append({**record, "geometry": geometry})
            detections.append({**record, "pixel_points": pixel_points, "bbox": (xmin, ymin, xmax, ymax)})
            all_features.append({**record, "geometry": geometry})
            all_rows.append(record)

        tile_geojson = geojson_dir / f"{image_path.stem}.geojson"
        if features:
            gpd.GeoDataFrame(features, geometry="geometry", crs=crs).to_file(tile_geojson, driver="GeoJSON")
        else:
            empty = gpd.GeoDataFrame(
                columns=["image", "instance_id", "score", "xmin", "ymin", "xmax", "ymax", "model", "geometry"],
                geometry="geometry",
                crs=crs,
            )
            empty.to_file(tile_geojson, driver="GeoJSON")

        draw_preview(image_path, detections, preview_dir / f"{image_path.stem}.jpg")
        print(f"{image_path.name}: {len(features)} masks")

    merged_csv = output_dir / "detectree2_predictions.csv"
    pd.DataFrame(all_rows).to_csv(merged_csv, index=False)

    merged_geojson = output_dir / "detectree2_predictions.geojson"
    if all_features:
        gpd.GeoDataFrame(all_features, geometry="geometry", crs=output_crs).to_file(merged_geojson, driver="GeoJSON")
    else:
        empty = gpd.GeoDataFrame(
            columns=["image", "instance_id", "score", "xmin", "ymin", "xmax", "ymax", "model", "geometry"],
            geometry="geometry",
            crs=output_crs,
        )
        empty.to_file(merged_geojson, driver="GeoJSON")

    summary = pd.DataFrame(
        [
            {
                "model": model_label,
                "tiles": len(image_paths),
                "predictions": len(all_rows),
                "confidence_threshold": args.confidence_threshold,
                "nms_threshold": args.nms_threshold,
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
