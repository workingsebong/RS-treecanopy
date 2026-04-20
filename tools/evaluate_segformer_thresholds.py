#!/usr/bin/env python3
"""Sweep probability thresholds for a trained SegFormer canopy model."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs/segformer/tree_canopy_semantic_random_b0_e20"


@torch.no_grad()
def collect_probs(model, dataset: CanopyDataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    items = []
    model.eval()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[:, 1]
        items.append(
            {
                "stem": batch["stem"][0],
                "image_path": Path(batch["image_path"][0]),
                "mask_path": Path(batch["mask_path"][0]),
                "prob": probs[0].cpu().numpy(),
                "label": labels[0].cpu().numpy().astype(np.uint8),
            }
        )
    return items


def metric_for_threshold(items, threshold: float) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for item in items:
        pred = item["prob"] >= threshold
        label = item["label"] == 1
        tp += int((pred & label).sum())
        fp += int((pred & ~label).sum())
        fn += int((~pred & label).sum())
        tn += int((~pred & ~label).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "threshold": threshold,
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_threshold_outputs(items, threshold: float, output_dir: Path, split: str) -> None:
    pred_dir = output_dir / "threshold_pred_masks" / split
    preview_dir = output_dir / "threshold_preview" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_paths = []
    for item in items:
        pred = (item["prob"] >= threshold).astype(np.uint8)
        pred_path = pred_dir / f"{item['stem']}.png"
        Image.fromarray(pred * 255, mode="L").save(pred_path)
        preview_path = preview_dir / f"{item['stem']}.jpg"
        save_preview_panel(item["image_path"], item["mask_path"], pred, preview_path)
        preview_paths.append(preview_path)
    make_contact_sheet(preview_paths, output_dir / f"{split}_threshold_preview_sheet.jpg", f"SegFormer {split}, threshold {threshold:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SegFormer probability thresholds.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dir = args.run_dir.resolve()
    model = SegformerForSemanticSegmentation.from_pretrained(run_dir / "best_model").to(device)
    val_dataset = CanopyDataset(args.dataset_dir.resolve(), "val", image_size=args.image_size, augment=False)
    test_dataset = CanopyDataset(args.dataset_dir.resolve(), "test", image_size=args.image_size, augment=False)

    val_items = collect_probs(model, val_dataset, device)
    test_items = collect_probs(model, test_dataset, device)
    thresholds = np.arange(args.start, args.stop + args.step / 2, args.step)

    rows = []
    for threshold in thresholds:
        val_metrics = metric_for_threshold(val_items, float(threshold))
        test_metrics = metric_for_threshold(test_items, float(threshold))
        rows.append({"split": "val", **val_metrics})
        rows.append({"split": "test", **test_metrics})

    write_csv(run_dir / "threshold_metrics.csv", rows)
    val_rows = [row for row in rows if row["split"] == "val"]
    best_val = max(val_rows, key=lambda row: (row["iou"], row["dice"]))
    best_threshold = float(best_val["threshold"])
    best_test = metric_for_threshold(test_items, best_threshold)
    save_threshold_outputs(val_items, best_threshold, run_dir, "val")
    save_threshold_outputs(test_items, best_threshold, run_dir, "test")

    summary = {
        "selected_by": "best val IoU",
        "best_threshold": best_threshold,
        "best_val": best_val,
        "test_at_best_val_threshold": best_test,
    }
    with (run_dir / "threshold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved threshold metrics: {run_dir / 'threshold_metrics.csv'}")


if __name__ == "__main__":
    main()
