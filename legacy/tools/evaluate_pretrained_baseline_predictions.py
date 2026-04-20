#!/usr/bin/env python3
"""Evaluate pretrained baseline GeoJSON predictions against test labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval/ground_truth_geojson/test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/pretrained_baselines/comparison"
DEFAULT_PREDICTIONS = [
    PROJECT_ROOT / "outputs/pretrained_baselines/deepforest/deepforest_predictions.geojson",
    PROJECT_ROOT / "outputs/pretrained_baselines/detectree2_conf05/detectree2_predictions.geojson",
]


def read_ground_truth(gt_dir: Path) -> gpd.GeoDataFrame:
    frames = []
    for path in sorted(gt_dir.glob("*.geojson")):
        gdf = gpd.read_file(path)
        gdf["image_stem"] = path.stem
        frames.append(gdf)
    if not frames:
        raise RuntimeError(f"No ground-truth GeoJSON files found in {gt_dir}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def read_predictions(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        gdf["image_stem"] = pd.Series(dtype=str)
        return gdf
    if "image" not in gdf.columns:
        raise ValueError(f"Prediction file has no image column: {path}")
    gdf["image_stem"] = gdf["image"].map(lambda value: Path(str(value)).stem)
    return gdf


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    return gdf


def greedy_match(gt: gpd.GeoDataFrame, pred: gpd.GeoDataFrame, threshold: float) -> tuple[int, float]:
    pairs = []
    for gt_idx, gt_geom in enumerate(gt.geometry):
        if gt_geom is None or gt_geom.is_empty:
            continue
        for pred_idx, pred_geom in enumerate(pred.geometry):
            if pred_geom is None or pred_geom.is_empty:
                continue
            intersection = gt_geom.intersection(pred_geom).area
            if intersection <= 0:
                continue
            union = gt_geom.union(pred_geom).area
            if union <= 0:
                continue
            iou = intersection / union
            if iou >= threshold:
                pairs.append((iou, gt_idx, pred_idx))

    pairs.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matched_ious = []
    for iou, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matched_ious.append(iou)

    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return len(matched_ious), mean_iou


def metric_row(model: str, image_stem: str, gt_count: int, pred_count: int, tp: int, mean_iou: float, threshold: float):
    fp = pred_count - tp
    fn = gt_count - tp
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "model": model,
        "image_stem": image_stem,
        "iou_threshold": threshold,
        "gt": gt_count,
        "pred": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
    }


def evaluate_one(model: str, gt_all: gpd.GeoDataFrame, pred_all: gpd.GeoDataFrame, thresholds: list[float]):
    rows = []
    if gt_all.crs is None:
        gt_all = gt_all.set_crs("EPSG:4326")
    if pred_all.crs is None:
        pred_all = pred_all.set_crs(gt_all.crs)
    pred_all = pred_all.to_crs(gt_all.crs)

    gt_metric = clean_geometries(gt_all.to_crs("EPSG:5186"))
    pred_metric = (
        clean_geometries(pred_all.to_crs("EPSG:5186"))
        if not pred_all.empty
        else pred_all.set_crs(gt_all.crs).to_crs("EPSG:5186")
    )

    image_stems = sorted(gt_metric["image_stem"].unique())
    for threshold in thresholds:
        total_gt = total_pred = total_tp = 0
        matched_ious = []
        for image_stem in image_stems:
            gt = gt_metric[gt_metric["image_stem"] == image_stem]
            pred = pred_metric[pred_metric["image_stem"] == image_stem]
            tp, mean_iou = greedy_match(gt, pred, threshold)
            row = metric_row(model, image_stem, len(gt), len(pred), tp, mean_iou, threshold)
            rows.append(row)
            total_gt += len(gt)
            total_pred += len(pred)
            total_tp += tp
            if tp:
                matched_ious.append(mean_iou)

        rows.append(
            metric_row(
                model,
                "__overall__",
                total_gt,
                total_pred,
                total_tp,
                sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
                threshold,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained baseline predictions.")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--prediction", type=Path, nargs="*", default=DEFAULT_PREDICTIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_all = read_ground_truth(args.gt_dir)
    rows = []
    for prediction_path in args.prediction:
        pred_all = read_predictions(prediction_path)
        model = prediction_path.parent.name
        rows.extend(evaluate_one(model, gt_all, pred_all, args.thresholds))

    metrics = pd.DataFrame(rows)
    metrics_path = output_dir / "pretrained_baseline_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    overall = metrics[metrics["image_stem"] == "__overall__"].copy()
    overall_path = output_dir / "pretrained_baseline_overall.csv"
    overall.to_csv(overall_path, index=False)

    print(overall.to_string(index=False))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved overall: {overall_path}")


if __name__ == "__main__":
    main()
