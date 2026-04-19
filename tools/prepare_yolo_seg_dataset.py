#!/usr/bin/env python3
"""Prepare a patch-separated YOLO segmentation dataset from saved labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGES = PROJECT_ROOT / "data/processed/labeling_candidates_512_random_seoul_urban/images"
DEFAULT_LABEL_JSON = PROJECT_ROOT / "data/processed/labels_512_random_seoul_urban/json"
DEFAULT_LABEL_YOLO = PROJECT_ROOT / "data/processed/labels_512_random_seoul_urban/yolo_seg"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/yolo_seg_512_random_seoul_urban"

# Keep every tile from the same source patch in one split. This gives a small
# but useful spatial separation without pretending the pilot labels are large.
SPLIT_BY_PATCH = {
    "train": {
        "p03_yeongdeungpo_mullae",
        "p05_mapo_hongdae_yeonnam",
        "p06_gangnam_teheran",
    },
    "val": {
        "p01_guro_digital",
    },
    "test": {
        "p02_songpa_jamsil",
        "p04_seongdong_wangsimni",
    },
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_for_patch(patch_id: str) -> str:
    for split, patch_ids in SPLIT_BY_PATCH.items():
        if patch_id in patch_ids:
            return split
    raise ValueError(f"No split configured for patch: {patch_id}")


def read_label_records(label_json_dir: Path, image_dir: Path, yolo_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_path in sorted(label_json_dir.glob("*.json")):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        image_name = Path(doc.get("image") or f"{json_path.stem}.png").name
        image_path = image_dir / image_name
        label_path = yolo_dir / f"{Path(image_name).stem}.txt"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for label: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing YOLO label for image: {label_path}")

        metadata = doc.get("metadata", {})
        patch_id = metadata.get("patch_id", "")
        split = split_for_patch(patch_id)
        shape_count = len(doc.get("shapes", []))
        records.append(
            {
                "image": image_path.name,
                "label": label_path.name,
                "json": json_path.name,
                "split": split,
                "patch_id": patch_id,
                "pool_name": metadata.get("pool_name", ""),
                "tile_id": metadata.get("tile_id", ""),
                "shape_count": shape_count,
                "is_empty": shape_count == 0,
                "image_path": str(image_path),
                "label_path": str(label_path),
            }
        )
    return records


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists. Use --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    for split in SPLIT_BY_PATCH:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_records(records: list[dict[str, Any]], output_root: Path) -> None:
    for record in records:
        split = record["split"]
        image_src = Path(record["image_path"])
        label_src = Path(record["label_path"])
        shutil.copy2(image_src, output_root / "images" / split / image_src.name)
        shutil.copy2(label_src, output_root / "labels" / split / label_src.name)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    by_patch: dict[str, dict[str, Any]] = {}
    for split in SPLIT_BY_PATCH:
        split_records = [record for record in records if record["split"] == split]
        by_split[split] = {
            "tiles": len(split_records),
            "empty_tiles": sum(1 for record in split_records if record["is_empty"]),
            "polygons": sum(int(record["shape_count"]) for record in split_records),
            "patches": sorted({record["patch_id"] for record in split_records}),
        }
    for patch_id in sorted({record["patch_id"] for record in records}):
        patch_records = [record for record in records if record["patch_id"] == patch_id]
        by_patch[patch_id] = {
            "split": patch_records[0]["split"],
            "tiles": len(patch_records),
            "empty_tiles": sum(1 for record in patch_records if record["is_empty"]),
            "polygons": sum(int(record["shape_count"]) for record in patch_records),
        }
    return {
        "dataset": "yolo_seg_512_random_seoul_urban",
        "class_names": ["tree"],
        "split_strategy": "source patch separated",
        "split_by_patch": {split: sorted(patch_ids) for split, patch_ids in SPLIT_BY_PATCH.items()},
        "totals": {
            "tiles": len(records),
            "empty_tiles": sum(1 for record in records if record["is_empty"]),
            "polygons": sum(int(record["shape_count"]) for record in records),
        },
        "by_split": by_split,
        "by_patch": by_patch,
    }


def write_dataset_yaml(output_root: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: tree",
            "",
        ]
    )
    (output_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO segmentation dataset for tree crowns.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--label-json", type=Path, default=DEFAULT_LABEL_JSON)
    parser.add_argument("--label-yolo", type=Path, default=DEFAULT_LABEL_YOLO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output.resolve()
    records = read_label_records(
        label_json_dir=args.label_json.resolve(),
        image_dir=args.images.resolve(),
        yolo_dir=args.label_yolo.resolve(),
    )
    if not records:
        raise RuntimeError("No label JSON files found.")

    prepare_output(output_root, overwrite=args.overwrite)
    copy_records(records, output_root)
    summary = summarize(records)
    write_csv(output_root / "split_index.csv", records)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_dataset_yaml(output_root)

    print(f"Prepared YOLO segmentation dataset: {output_root}")
    print(json.dumps(summary["totals"], ensure_ascii=False))
    for split, values in summary["by_split"].items():
        print(f"{split}: {values['tiles']} tiles, {values['polygons']} polygons, {values['empty_tiles']} empty")


if __name__ == "__main__":
    main()
