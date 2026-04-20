#!/usr/bin/env python3
"""Train a SegFormer binary semantic segmentation baseline for tree canopy."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/processed/semantic_seg_512_random_seoul_urban"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/segformer/tree_canopy_semantic_mit_b0_e20"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class CanopyDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, augment: bool = False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.image_paths = sorted((root / "images" / split).glob("*.png"))
        self.mask_paths = [root / "masks" / split / image_path.name for image_path in self.image_paths]
        if not self.image_paths:
            raise RuntimeError(f"No images found for split={split!r} in {root}")
        missing = [path for path in self.mask_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(missing[0])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            turns = random.randint(0, 3)
            if turns:
                image = image.rotate(90 * turns)
                mask = mask.rotate(90 * turns)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.int64)
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        mask_tensor = torch.from_numpy(mask_arr)

        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
            "stem": image_path.stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_random_b0(num_labels: int = 2) -> SegformerForSemanticSegmentation:
    config = SegformerConfig(
        num_channels=3,
        num_labels=num_labels,
        id2label={0: "background", 1: "canopy"},
        label2id={"background": 0, "canopy": 1},
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=256,
    )
    return SegformerForSemanticSegmentation(config)


def build_model(pretrained_model: str | None, random_fallback: bool) -> tuple[SegformerForSemanticSegmentation, str]:
    if pretrained_model:
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model,
                num_labels=2,
                id2label={0: "background", 1: "canopy"},
                label2id={"background": 0, "canopy": 1},
                ignore_mismatched_sizes=True,
            )
            return model, pretrained_model
        except Exception as exc:
            if not random_fallback:
                raise
            print(f"Could not load pretrained model {pretrained_model!r}: {exc}")
            print("Falling back to randomly initialized SegFormer-B0 config.")

    return build_random_b0(), "random_segformer_b0"


def canopy_ratio(dataset: CanopyDataset) -> float:
    positives = 0
    total = 0
    for path in dataset.mask_paths:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
        positives += int(mask.sum())
        total += int(mask.size)
    return positives / total if total else 0.0


def make_class_weights(train_dataset: CanopyDataset, max_canopy_weight: float, device: torch.device) -> torch.Tensor:
    ratio = canopy_ratio(train_dataset)
    canopy_weight = (1 - ratio) / max(ratio, 1e-6)
    canopy_weight = min(max_canopy_weight, max(1.0, canopy_weight))
    return torch.tensor([1.0, canopy_weight], dtype=torch.float32, device=device)


def batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    return pixel_values, labels


def forward_loss(model: nn.Module, pixel_values: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor):
    outputs = model(pixel_values=pixel_values)
    logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    return loss, logits


def update_confusion(counts: dict[str, int], preds: torch.Tensor, labels: torch.Tensor) -> None:
    pred_pos = preds == 1
    label_pos = labels == 1
    counts["tp"] += int((pred_pos & label_pos).sum().item())
    counts["fp"] += int((pred_pos & ~label_pos).sum().item())
    counts["fn"] += int((~pred_pos & label_pos).sum().item())
    counts["tn"] += int((~pred_pos & ~label_pos).sum().item())


def metrics_from_counts(counts: dict[str, int], loss: float, batches: int) -> dict[str, float]:
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
        "loss": loss / max(1, batches),
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    total_loss = 0.0
    batches = 0

    for batch in loader:
        pixel_values, labels = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            loss, logits = forward_loss(model, pixel_values, labels, class_weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)
        update_confusion(counts, preds.cpu(), labels.cpu())
        total_loss += float(loss.detach().cpu().item())
        batches += 1

    return metrics_from_counts(counts, total_loss, batches)


def overlay_mask(image: Image.Image, mask: np.ndarray, fill: tuple[int, int, int, int], outline: tuple[int, int, int, int]):
    out = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 3:
            continue
        points = [tuple(map(int, point[0])) for point in contour]
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=2)
    return Image.alpha_composite(out, layer).convert("RGB")


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = load_font(18)
    header_h = 32
    canvas = Image.new("RGB", (image.width, image.height + header_h), (24, 26, 28))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    return canvas


def load_font(size: int) -> ImageFont.ImageFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_preview_panel(original_path: Path, gt_mask_path: Path, pred_mask: np.ndarray, output_path: Path) -> None:
    image = Image.open(original_path).convert("RGB")
    gt = (np.asarray(Image.open(gt_mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    pred = cv2.resize(pred_mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)

    panels = [
        add_title(image, "Original"),
        add_title(overlay_mask(image, gt, (0, 255, 140, 75), (0, 255, 140, 240)), "Ground truth"),
        add_title(overlay_mask(image, pred, (255, 140, 0, 85), (255, 120, 0, 245)), "SegFormer pred"),
    ]
    gap = 10
    width = sum(panel.width for panel in panels) + gap * (len(panels) + 1)
    height = max(panel.height for panel in panels) + gap * 2
    canvas = Image.new("RGB", (width, height), (238, 240, 242))
    x = gap
    for panel in panels:
        canvas.paste(panel, (x, gap))
        x += panel.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def make_contact_sheet(image_paths: list[Path], output_path: Path, title: str) -> None:
    if not image_paths:
        return
    font = load_font(20)
    thumb_w = 420
    gap = 14
    title_h = 44
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        scale = thumb_w / image.width
        thumbs.append(image.resize((thumb_w, int(round(image.height * scale))), Image.Resampling.LANCZOS))
    cols = 1
    width = thumb_w + gap * 2
    height = title_h + sum(thumb.height for thumb in thumbs) + gap * (len(thumbs) + 1)
    sheet = Image.new("RGB", (width, height), (238, 240, 242))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), title, fill=(24, 26, 28), font=font)
    y = title_h + gap
    for thumb in thumbs:
        sheet.paste(thumb, (gap, y))
        y += thumb.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


@torch.no_grad()
def save_predictions(model: nn.Module, dataset: CanopyDataset, output_dir: Path, device: torch.device) -> None:
    model.eval()
    pred_dir = output_dir / "pred_masks" / dataset.split
    preview_dir = output_dir / "preview" / dataset.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_paths = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in loader:
        pixel_values, labels = batch_to_device(batch, device)
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        stem = batch["stem"][0]
        pred_path = pred_dir / f"{stem}.png"
        Image.fromarray(pred * 255, mode="L").save(pred_path)
        preview_path = preview_dir / f"{stem}.jpg"
        save_preview_panel(Path(batch["image_path"][0]), Path(batch["mask_path"][0]), pred, preview_path)
        preview_paths.append(preview_path)

    make_contact_sheet(preview_paths, output_dir / f"{dataset.split}_preview_sheet.jpg", f"SegFormer {dataset.split} predictions")


def write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SegFormer canopy semantic segmentation baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained-model", default="nvidia/mit-b0")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--no-random-fallback", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-canopy-weight", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_dataset = CanopyDataset(args.dataset_dir.resolve(), "train", image_size=args.image_size, augment=True)
    val_dataset = CanopyDataset(args.dataset_dir.resolve(), "val", image_size=args.image_size, augment=False)
    test_dataset = CanopyDataset(args.dataset_dir.resolve(), "test", image_size=args.image_size, augment=False)

    pretrained_model = None if args.random_init else args.pretrained_model
    model, model_source = build_model(pretrained_model, random_fallback=not args.no_random_fallback)
    model.to(device)

    class_weights = make_class_weights(train_dataset, args.max_canopy_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Model source: {model_source}")
    print(f"Device: {device}")
    print(f"Class weights: background={class_weights[0].item():.3f}, canopy={class_weights[1].item():.3f}")

    rows = []
    best_val_iou = -1.0
    best_epoch = 0
    best_dir = output_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, class_weights, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, class_weights, optimizer=None)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        write_metrics(output_dir / "metrics.csv", rows)

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch
            model.save_pretrained(best_dir)

        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train loss {train_metrics['loss']:.4f} IoU {train_metrics['iou']:.4f} Dice {train_metrics['dice']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} IoU {val_metrics['iou']:.4f} Dice {val_metrics['dice']:.4f}"
        )

    model = SegformerForSemanticSegmentation.from_pretrained(best_dir).to(device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_metrics = run_epoch(model, test_loader, device, class_weights, optimizer=None)
    save_predictions(model, train_dataset, output_dir, device)
    save_predictions(model, val_dataset, output_dir, device)
    save_predictions(model, test_dataset, output_dir, device)

    summary = {
        "model": "SegFormer binary canopy semantic segmentation",
        "model_source": model_source,
        "dataset": str(args.dataset_dir.resolve().relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "class_weights": {"background": float(class_weights[0].item()), "canopy": float(class_weights[1].item())},
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "test_metrics": test_metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved output: {output_dir}")


if __name__ == "__main__":
    main()
