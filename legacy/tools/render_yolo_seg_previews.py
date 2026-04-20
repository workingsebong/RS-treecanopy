#!/usr/bin/env python3
"""Render orange YOLO segmentation previews."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "outputs/yolo/tree_crown_512_yolo11n_seg_e50/weights/best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "data/processed/yolo_seg_512_random_seoul_urban/images/test"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/yolo/tree_crown_512_yolo11n_seg_e50_test_predict_orange"

PALETTE_RGB = np.array(
    [
        [255, 150, 0],
        [255, 190, 40],
        [255, 112, 40],
        [255, 210, 90],
        [230, 95, 0],
        [255, 165, 80],
    ],
    dtype=np.float32,
)


def render_result(result, output_path: Path, alpha: float, draw_boxes: bool) -> int:
    image = result.orig_img.copy()
    masks = result.masks.data.cpu().numpy() if result.masks is not None else np.empty((0, *image.shape[:2]))
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))

    rendered = image.astype(np.float32)
    for idx, mask in enumerate(masks):
        mask_bool = mask > 0.5
        if mask_bool.shape[:2] != image.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        color_bgr = PALETTE_RGB[idx % len(PALETTE_RGB)][::-1]
        rendered[mask_bool] = rendered[mask_bool] * (1 - alpha) + color_bgr * alpha

        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, color_bgr.tolist(), 2)

    rendered_u8 = np.clip(rendered, 0, 255).astype(np.uint8)

    if draw_boxes:
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(rendered_u8, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.rectangle(rendered_u8, (x1, y1), (x2, y2), (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered_u8)
    return len(masks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render orange YOLO-seg prediction previews.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--no-boxes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.jpg"):
        old_file.unlink()

    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )
    for result in results:
        output_path = output_dir / f"{Path(result.path).stem}.jpg"
        count = render_result(result, output_path, alpha=args.alpha, draw_boxes=not args.no_boxes)
        print(f"{output_path.name}: {count} masks")

    print(f"Saved previews: {output_dir}")


if __name__ == "__main__":
    main()
