#!/usr/bin/env python
"""Run ArcGIS pretrained tree baselines with ArcGIS Pro Python.

Run this script with the ArcGIS Pro Python environment, not the WSL conda env.
It expects ArcGIS Pro, Image Analyst, and the ArcGIS deep learning libraries.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval/images_geotiff/test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/pretrained_baselines"


def safe_name(value: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "tile"
    return cleaned[:max_len]


def ensure_file_gdb(arcpy, output_dir: Path, gdb_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    gdb_path = output_dir / gdb_name
    if not arcpy.Exists(str(gdb_path)):
        arcpy.management.CreateFileGDB(str(output_dir), gdb_name)
    return gdb_path


def add_source_tile(arcpy, feature_class: str, source_tile: str) -> None:
    fields = {field.name for field in arcpy.ListFields(feature_class)}
    if "source_tile" not in fields:
        arcpy.management.AddField(feature_class, "source_tile", "TEXT", field_length=255)
    arcpy.management.CalculateField(feature_class, "source_tile", repr(source_tile), "PYTHON3")


def export_geojson(arcpy, feature_class: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arcpy.conversion.FeaturesToJSON(
        feature_class,
        str(output_path),
        "FORMATTED",
        "NO_Z_VALUES",
        "NO_M_VALUES",
        "GEOJSON",
    )


def detect_tile(
    arcpy,
    detect_objects,
    image_path: Path,
    output_fc: str,
    model_path: Path,
    model_arguments: str,
    run_nms: str,
    max_overlap_ratio: float,
) -> None:
    confidence_field = "Confidence"
    class_field = "Class"
    processing_mode = "PROCESS_AS_MOSAICKED_IMAGE"

    try:
        detect_objects(
            str(image_path),
            output_fc,
            str(model_path),
            model_arguments,
            run_nms,
            confidence_field,
            class_field,
            max_overlap_ratio,
            processing_mode,
            "NO_PIXELSPACE",
        )
    except TypeError:
        # Older ArcGIS Pro versions do not expose the use_pixelspace argument.
        detect_objects(
            str(image_path),
            output_fc,
            str(model_path),
            model_arguments,
            run_nms,
            confidence_field,
            class_field,
            max_overlap_ratio,
            processing_mode,
        )


def run_model(
    arcpy,
    detect_objects,
    *,
    model_key: str,
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    model_arguments: str,
    run_nms: str,
    max_overlap_ratio: float,
    export_json: bool,
) -> None:
    model_output_dir = output_dir / model_key
    gdb_path = ensure_file_gdb(arcpy, model_output_dir, f"{model_key}.gdb")
    geojson_dir = model_output_dir / "geojson"

    input_tiles = sorted(input_dir.glob("*.tif"))
    if not input_tiles:
        raise RuntimeError(f"No GeoTIFF tiles found in {input_dir}")

    feature_classes: list[str] = []
    for image_path in input_tiles:
        fc_name = safe_name(image_path.stem)
        output_fc = str(gdb_path / fc_name)
        if arcpy.Exists(output_fc):
            arcpy.management.Delete(output_fc)

        print(f"[{model_key}] {image_path.name}")
        detect_tile(
            arcpy,
            detect_objects,
            image_path,
            output_fc,
            model_path,
            model_arguments,
            run_nms,
            max_overlap_ratio,
        )
        add_source_tile(arcpy, output_fc, image_path.name)
        feature_classes.append(output_fc)

        if export_json:
            export_geojson(arcpy, output_fc, geojson_dir / f"{image_path.stem}.geojson")

    merged_fc = str(gdb_path / f"{model_key}_merged")
    if arcpy.Exists(merged_fc):
        arcpy.management.Delete(merged_fc)
    arcpy.management.Merge(feature_classes, merged_fc)

    if export_json:
        export_geojson(arcpy, merged_fc, model_output_dir / f"{model_key}_merged.geojson")

    print(f"[{model_key}] saved: {gdb_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ArcGIS pretrained tree baselines.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tree-seg-model", type=Path, help="ArcGIS Tree Segmentation .dlpk path")
    parser.add_argument("--tree-det-model", type=Path, help="ArcGIS Tree Detection .dlpk path")
    parser.add_argument("--tree-seg-args", default="")
    parser.add_argument("--tree-det-args", default="")
    parser.add_argument("--run-nms", default="NMS", choices=["NMS", "NO_NMS"])
    parser.add_argument("--max-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--processor", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-geojson", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tree_seg_model and not args.tree_det_model:
        raise SystemExit("Pass at least one model: --tree-seg-model and/or --tree-det-model")

    import arcpy
    from arcpy.ia import DetectObjectsUsingDeepLearning

    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.env.overwriteOutput = True
    arcpy.env.processorType = args.processor
    if args.processor == "GPU":
        arcpy.env.gpuId = args.gpu_id

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tree_seg_model:
        run_model(
            arcpy,
            DetectObjectsUsingDeepLearning,
            model_key="arcgis_tree_segmentation",
            model_path=args.tree_seg_model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_arguments=args.tree_seg_args,
            run_nms=args.run_nms,
            max_overlap_ratio=args.max_overlap_ratio,
            export_json=not args.no_geojson,
        )

    if args.tree_det_model:
        run_model(
            arcpy,
            DetectObjectsUsingDeepLearning,
            model_key="arcgis_tree_detection",
            model_path=args.tree_det_model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_arguments=args.tree_det_args,
            run_nms=args.run_nms,
            max_overlap_ratio=args.max_overlap_ratio,
            export_json=not args.no_geojson,
        )


if __name__ == "__main__":
    main()
