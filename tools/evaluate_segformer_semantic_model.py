#!/usr/bin/env python3
"""Evaluate any SegFormer semantic canopy model on the current dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from train_segformer_semantic import CanopyDataset, make_contact_sheet, save_preview_panel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/processed/semantic_seg_512_random_seoul_urban"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/segformer/tree_canopy_semantic_tcd_mit_b2_zeroshot"


def update_counts(counts: dict[str, int], pred: np.ndarray, label: np.ndarray) -> None:
    pred_pos = pred == 1
    label_pos = label == 1
    counts["tp"] += int((pred_pos & label_pos).sum())
    counts["fp"] += int((pred_pos & ~label_pos).sum())
    counts["fn"] += int((~pred_pos & label_pos).sum())
    counts["tn"] += int((~pred_pos & ~label_pos).sum())


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def collect_items(model, dataset: CanopyDataset, device: torch.device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    items = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[:, 1]
        argmax = torch.argmax(logits, dim=1)
        items.append(
            {
                "stem": batch["stem"][0],
                "image_path": Path(batch["image_path"][0]),
                "mask_path": Path(batch["mask_path"][0]),
                "label": labels[0].cpu().numpy().astype(np.uint8),
                "prob": probs[0].cpu().numpy(),
                "argmax": argmax[0].cpu().numpy().astype(np.uint8),
            }
        )

    return items


def evaluate_items(items, threshold: float | None) -> dict[str, float | int]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for item in items:
        if threshold is None:
            pred = item["argmax"]
        else:
            pred = (item["prob"] >= threshold).astype(np.uint8)
        update_counts(counts, pred, item["label"])
    return metrics_from_counts(counts)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(items, output_dir: Path, split: str, threshold: float | None, suffix: str) -> None:
    pred_dir = output_dir / f"{suffix}_pred_masks" / split
    preview_dir = output_dir / f"{suffix}_preview" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_paths = []

    for item in items:
        if threshold is None:
            pred = item["argmax"].astype(np.uint8)
        else:
            pred = (item["prob"] >= threshold).astype(np.uint8)
        pred_path = pred_dir / f"{item['stem']}.png"
        Image.fromarray(pred * 255, mode="L").save(pred_path)
        preview_path = preview_dir / f"{item['stem']}.jpg"
        save_preview_panel(item["image_path"], item["mask_path"], pred, preview_path)
        preview_paths.append(preview_path)

    title = f"SegFormer {split} predictions ({suffix})"
    make_contact_sheet(preview_paths, output_dir / f"{split}_{suffix}_preview_sheet.jpg", title)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SegFormer semantic canopy model.")
    parser.add_argument("--model-id", default="restor/tcd-segformer-mit-b2")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    val_dataset = CanopyDataset(args.dataset_dir.resolve(), "val", image_size=args.image_size, augment=False)
    test_dataset = CanopyDataset(args.dataset_dir.resolve(), "test", image_size=args.image_size, augment=False)

    val_items = collect_items(model, val_dataset, device)
    test_items = collect_items(model, test_dataset, device)

    rows = []
    for split, items in [("val", val_items), ("test", test_items)]:
        metrics = evaluate_items(items, threshold=None)
        rows.append({"split": split, "mode": "argmax", "threshold": "", **metrics})

    thresholds = np.arange(args.start, args.stop + args.step / 2, args.step)
    for threshold in thresholds:
        for split, items in [("val", val_items), ("test", test_items)]:
            metrics = evaluate_items(items, threshold=float(threshold))
            rows.append({"split": split, "mode": "threshold", "threshold": float(threshold), **metrics})

    val_threshold_rows = [row for row in rows if row["split"] == "val" and row["mode"] == "threshold"]
    best_val = max(val_threshold_rows, key=lambda row: (row["iou"], row["dice"]))
    best_threshold = float(best_val["threshold"])
    test_at_best = evaluate_items(test_items, threshold=best_threshold)
    test_argmax = evaluate_items(test_items, threshold=None)

    save_outputs(test_items, output_dir, "test", threshold=None, suffix="argmax")
    save_outputs(test_items, output_dir, "test", threshold=best_threshold, suffix=f"threshold_{best_threshold:.2f}")
    save_outputs(val_items, output_dir, "val", threshold=best_threshold, suffix=f"threshold_{best_threshold:.2f}")

    write_csv(output_dir / "metrics.csv", rows)
    summary = {
        "model_id": args.model_id,
        "dataset": str(args.dataset_dir.resolve().relative_to(PROJECT_ROOT)),
        "image_size": args.image_size,
        "device": str(device),
        "test_argmax": test_argmax,
        "best_val_threshold": best_threshold,
        "best_val": best_val,
        "test_at_best_val_threshold": test_at_best,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved output: {output_dir}")


if __name__ == "__main__":
    main()
