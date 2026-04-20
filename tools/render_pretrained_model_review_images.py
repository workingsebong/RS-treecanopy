#!/usr/bin/env python3
"""Render visual review sheets for tree-crown model comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from PIL import Image, ImageDraw, ImageFont
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval/images_png/test"
DEFAULT_GEOTIFF_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval/images_geotiff/test"
DEFAULT_GT_DIR = PROJECT_ROOT / "data/processed/pretrained_baseline_eval/ground_truth_geojson/test"
DEFAULT_YOLO_DIR = PROJECT_ROOT / "outputs/yolo/tree_crown_512_yolo11n_seg_e50_test_predict_orange"
DEFAULT_DEEPFOREST_DIR = PROJECT_ROOT / "outputs/pretrained_baselines/deepforest/preview"
DEFAULT_DETECTREE2_DIR = PROJECT_ROOT / "outputs/pretrained_baselines/detectree2_conf05/preview"
DEFAULT_DETECTREE2_LOOSE_DIR = PROJECT_ROOT / "outputs/pretrained_baselines/detectree2/preview"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/pretrained_baselines/review"


def load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def ellipsize(text: str, max_chars: int = 44) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def geometry_to_pixel_rings(geometry, transform):
    if geometry is None or geometry.is_empty:
        return []

    polygons = []
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
    else:
        return []

    inv_transform = ~transform
    rings = []
    for polygon in polygons:
        coords = []
        for x, y in polygon.exterior.coords:
            col, row = inv_transform * (x, y)
            coords.append((float(col), float(row)))
        if len(coords) >= 3:
            rings.append(coords)
    return rings


def render_gt_overlay(image_path: Path, geotiff_path: Path, gt_path: Path) -> Image.Image:
    image = load_image(image_path)
    draw = ImageDraw.Draw(image, "RGBA")

    if not gt_path.exists():
        return image

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        raster_crs = src.crs

    gdf = gpd.read_file(gt_path)
    if gdf.empty:
        return image
    if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    for geometry in gdf.geometry:
        for ring in geometry_to_pixel_rings(geometry, transform):
            draw.polygon(ring, fill=(0, 255, 140, 70))
            draw.line(ring + [ring[0]], fill=(0, 255, 140, 245), width=3)

    return image


def add_header(image: Image.Image, title: str, subtitle: str | None = None) -> Image.Image:
    font = load_font(18)
    sub_font = load_font(13)
    header_h = 46 if subtitle else 36
    canvas = Image.new("RGB", (image.width, image.height + header_h), (24, 26, 28))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, image.width, header_h], fill=(24, 26, 28))
    draw.text((10, 6), ellipsize(title), fill=(255, 255, 255), font=font)
    if subtitle:
        draw.text((10, 28), ellipsize(subtitle, 54), fill=(210, 216, 222), font=sub_font)
    return canvas


def resize_panel(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    height = int(round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_contact_sheet(items: list[tuple[str, Image.Image]], output_path: Path, title: str, cols: int = 4) -> None:
    thumb_w = 360
    gap = 14
    title_h = 44
    font = load_font(20)
    thumbs = [(name, resize_panel(image, thumb_w)) for name, image in items]
    if not thumbs:
        return

    rows = (len(thumbs) + cols - 1) // cols
    thumb_h = max(image.height for _, image in thumbs)
    width = cols * thumb_w + (cols + 1) * gap
    height = title_h + rows * thumb_h + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), (238, 240, 242))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), title, fill=(24, 26, 28), font=font)

    for idx, (_, image) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = gap + col * (thumb_w + gap)
        y = title_h + gap + row * (thumb_h + gap)
        sheet.paste(image, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_vertical_sheet(items: list[tuple[str, Image.Image]], output_path: Path, title: str) -> None:
    if not items:
        return

    gap = 14
    title_h = 44
    font = load_font(20)
    width = max(image.width for _, image in items) + gap * 2
    height = title_h + sum(image.height for _, image in items) + gap * (len(items) + 1)
    sheet = Image.new("RGB", (width, height), (238, 240, 242))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), title, fill=(24, 26, 28), font=font)

    y = title_h + gap
    for _, image in items:
        sheet.paste(image, (gap, y))
        y += image.height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def hstack_panels(panels: list[Image.Image], output_path: Path, gap: int = 10) -> None:
    if not panels:
        return
    max_h = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + gap * (len(panels) + 1)
    height = max_h + gap * 2
    canvas = Image.new("RGB", (width, height), (238, 240, 242))
    x = gap
    for panel in panels:
        canvas.paste(panel, (x, gap))
        x += panel.width + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render model comparison review images.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--geotiff-dir", type=Path, default=DEFAULT_GEOTIFF_DIR)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO_DIR)
    parser.add_argument("--deepforest-dir", type=Path, default=DEFAULT_DEEPFOREST_DIR)
    parser.add_argument("--detectree2-dir", type=Path, default=DEFAULT_DETECTREE2_DIR)
    parser.add_argument("--detectree2-loose-dir", type=Path, default=DEFAULT_DETECTREE2_LOOSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    panel_dir = output_dir / "comparison_panels"
    sheet_dir = output_dir / "model_contact_sheets"
    panel_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG test images found in {args.image_dir}")

    contact_items = {
        "original": [],
        "ground_truth": [],
        "yolo11n_seg": [],
        "deepforest": [],
        "detectree2_conf05": [],
        "detectree2_conf015": [],
    }
    all_model_rows = []

    for image_path in image_paths:
        stem = image_path.stem
        geotiff_path = require(args.geotiff_dir / f"{stem}.tif")
        gt_path = require(args.gt_dir / f"{stem}.geojson")
        original = load_image(image_path)
        gt = render_gt_overlay(image_path, geotiff_path, gt_path)
        yolo = load_image(require(args.yolo_dir / f"{stem}.jpg"))
        deepforest = load_image(require(args.deepforest_dir / f"{stem}.jpg"))
        detectree2 = load_image(require(args.detectree2_dir / f"{stem}.jpg"))
        detectree2_loose = load_image(require(args.detectree2_loose_dir / f"{stem}.jpg"))

        labeled = {
            "original": add_header(original, "Original", stem),
            "ground_truth": add_header(gt, "Ground truth", "manual crown polygons"),
            "yolo11n_seg": add_header(yolo, "YOLO11n-seg", "trained on pilot labels"),
            "deepforest": add_header(deepforest, "DeepForest", "pretrained, boxes"),
            "detectree2_conf05": add_header(detectree2, "Detectree2", "pretrained, conf 0.5"),
            "detectree2_conf015": add_header(detectree2_loose, "Detectree2", "pretrained, conf 0.15"),
        }

        for key, panel in labeled.items():
            contact_items[key].append((stem, panel))

        comparison_panels = [
            resize_panel(labeled["original"], 360),
            resize_panel(labeled["ground_truth"], 360),
            resize_panel(labeled["yolo11n_seg"], 360),
            resize_panel(labeled["deepforest"], 360),
            resize_panel(labeled["detectree2_conf05"], 360),
        ]
        hstack_panels(comparison_panels, panel_dir / f"{stem}_comparison.jpg")
        all_model_rows.append((stem, hstack_for_grid(comparison_panels)))

    sheet_titles = {
        "original": "Original test tiles",
        "ground_truth": "Ground truth crown polygons",
        "yolo11n_seg": "YOLO11n-seg predictions",
        "deepforest": "DeepForest pretrained predictions",
        "detectree2_conf05": "Detectree2 pretrained predictions, confidence 0.5",
        "detectree2_conf015": "Detectree2 pretrained predictions, confidence 0.15",
    }
    for key, items in contact_items.items():
        make_contact_sheet(items, sheet_dir / f"{key}_contact_sheet.jpg", sheet_titles[key])

    make_vertical_sheet(all_model_rows, output_dir / "all_models_comparison_sheet.jpg", "Model comparison by test tile")

    print(f"Saved comparison panels: {panel_dir}")
    print(f"Saved model contact sheets: {sheet_dir}")
    print(f"Saved overview sheet: {output_dir / 'all_models_comparison_sheet.jpg'}")


def hstack_for_grid(panels: list[Image.Image], gap: int = 8) -> Image.Image:
    max_h = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, max_h), (238, 240, 242))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


if __name__ == "__main__":
    main()
