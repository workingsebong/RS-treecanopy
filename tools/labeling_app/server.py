#!/usr/bin/env python3
"""Local polygon labeling app for tree crown tiles."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import struct
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clean_points(points: Any, width: int, height: int) -> list[list[float]]:
    cleaned: list[list[float]] = []
    if not isinstance(points, list):
        return cleaned
    for point in points:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        x = float_or_default(point[0], 0.0)
        y = float_or_default(point[1], 0.0)
        x = min(max(x, 0.0), max(width - 1, 0))
        y = min(max(y, 0.0), max(height - 1, 0))
        cleaned.append([round(x, 3), round(y, 3)])
    return cleaned


class LabelStore:
    def __init__(
        self,
        image_dir: Path,
        selected_csv: Path,
        label_root: Path,
    ) -> None:
        self.image_dir = image_dir
        self.selected_csv = selected_csv
        self.label_root = label_root
        self.json_dir = label_root / "json"
        self.yolo_dir = label_root / "yolo_seg"
        self.geojson_dir = label_root / "geojson"
        self.classes_path = label_root / "classes.txt"
        for path in [self.json_dir, self.yolo_dir, self.geojson_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.classes_path.write_text("tree\n", encoding="utf-8")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        rows = load_csv_rows(self.selected_csv)
        image_names = [p.name for p in sorted(self.image_dir.glob("*.png"))]
        metadata: dict[str, dict[str, Any]] = {}
        for image_name in image_names:
            matched: dict[str, str] | None = None
            for row in rows:
                source_name = row.get("file_name", "")
                if image_name == source_name or image_name.endswith("_" + source_name):
                    matched = row
                    break
            if matched is None:
                matched = {}
            metadata[image_name] = dict(matched)
            metadata[image_name]["label_image_name"] = image_name
            metadata[image_name]["source_file_name"] = matched.get("file_name", image_name)
        return metadata

    def image_path(self, image_name: str) -> Path:
        safe_name = Path(image_name).name
        path = self.image_dir / safe_name
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(safe_name)
        return path

    def label_json_path(self, image_name: str) -> Path:
        return self.json_dir / f"{Path(image_name).stem}.json"

    def yolo_path(self, image_name: str) -> Path:
        return self.yolo_dir / f"{Path(image_name).stem}.txt"

    def geojson_path(self, image_name: str) -> Path:
        return self.geojson_dir / f"{Path(image_name).stem}.geojson"

    def list_tiles(self) -> list[dict[str, Any]]:
        tiles: list[dict[str, Any]] = []
        for idx, path in enumerate(sorted(self.image_dir.glob("*.png")), start=1):
            width, height = read_png_size(path)
            label_doc = self.load_labels(path.name, width, height)
            meta = self.metadata.get(path.name, {})
            tiles.append(
                {
                    "index": idx,
                    "image": path.name,
                    "source_file_name": meta.get("source_file_name", path.name),
                    "tile_id": meta.get("tile_id", ""),
                    "selection_rank": int_or_default(meta.get("selection_rank"), idx),
                    "score_rank": int_or_default(meta.get("score_rank"), idx),
                    "width": width,
                    "height": height,
                    "valid_width": int_or_default(meta.get("valid_width"), width),
                    "valid_height": int_or_default(meta.get("valid_height"), height),
                    "is_edge": str(meta.get("is_edge", "False")).lower() == "true",
                    "saved_shapes": len(label_doc.get("shapes", [])),
                    "updated_at": label_doc.get("updated_at"),
                }
            )
        return sorted(tiles, key=lambda item: item["selection_rank"])

    def empty_doc(self, image_name: str, width: int, height: int) -> dict[str, Any]:
        meta = self.metadata.get(image_name, {})
        return {
            "image": image_name,
            "source_file_name": meta.get("source_file_name", image_name),
            "image_width": width,
            "image_height": height,
            "class_name": "tree",
            "shapes": [],
            "metadata": meta,
        }

    def load_labels(self, image_name: str, width: int, height: int) -> dict[str, Any]:
        path = self.label_json_path(image_name)
        if not path.exists():
            return self.empty_doc(image_name, width, height)
        try:
            with path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError:
            return self.empty_doc(image_name, width, height)
        doc.setdefault("image", image_name)
        doc.setdefault("image_width", width)
        doc.setdefault("image_height", height)
        doc.setdefault("class_name", "tree")
        doc.setdefault("shapes", [])
        doc.setdefault("metadata", self.metadata.get(image_name, {}))
        return doc

    def save_labels(self, image_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        image_path = self.image_path(image_name)
        default_width, default_height = read_png_size(image_path)
        width = int_or_default(payload.get("image_width"), default_width)
        height = int_or_default(payload.get("image_height"), default_height)
        shapes = []
        for idx, shape in enumerate(payload.get("shapes", []), start=1):
            if not isinstance(shape, dict):
                continue
            points = clean_points(shape.get("points"), width, height)
            if len(points) < 3:
                continue
            shapes.append(
                {
                    "id": str(shape.get("id") or f"tree_{idx:04d}"),
                    "label": "tree",
                    "points": points,
                }
            )

        doc = {
            "image": image_name,
            "source_file_name": self.metadata.get(image_name, {}).get("source_file_name", image_name),
            "image_width": width,
            "image_height": height,
            "class_name": "tree",
            "shapes": shapes,
            "metadata": self.metadata.get(image_name, {}),
            "updated_at": utc_now(),
        }

        json_path = self.label_json_path(image_name)
        json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        yolo_path = self.write_yolo(image_name, doc)
        geojson_path = self.write_geojson(image_name, doc)
        return {
            "ok": True,
            "shape_count": len(shapes),
            "json_path": str(json_path),
            "yolo_path": str(yolo_path),
            "geojson_path": str(geojson_path),
            "updated_at": doc["updated_at"],
        }

    def write_yolo(self, image_name: str, doc: dict[str, Any]) -> Path:
        width = max(int_or_default(doc.get("image_width"), 1), 1)
        height = max(int_or_default(doc.get("image_height"), 1), 1)
        lines = []
        for shape in doc.get("shapes", []):
            values = ["0"]
            for x, y in shape.get("points", []):
                values.append(f"{float(x) / width:.6f}")
                values.append(f"{float(y) / height:.6f}")
            lines.append(" ".join(values))
        path = self.yolo_path(image_name)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def write_geojson(self, image_name: str, doc: dict[str, Any]) -> Path:
        meta = self.metadata.get(image_name, {})
        width = max(int_or_default(meta.get("valid_width"), doc.get("image_width", 1)), 1)
        height = max(int_or_default(meta.get("valid_height"), doc.get("image_height", 1)), 1)
        west = float_or_default(meta.get("west"), 0.0)
        south = float_or_default(meta.get("south"), 0.0)
        east = float_or_default(meta.get("east"), float(doc.get("image_width", width)))
        north = float_or_default(meta.get("north"), float(doc.get("image_height", height)))
        has_geo = bool(meta.get("west") and meta.get("south") and meta.get("east") and meta.get("north"))

        features = []
        for idx, shape in enumerate(doc.get("shapes", []), start=1):
            ring = []
            for x, y in shape.get("points", []):
                px = min(max(float(x), 0.0), float(width))
                py = min(max(float(y), 0.0), float(height))
                if has_geo:
                    lon = west + (px / width) * (east - west)
                    lat = north - (py / height) * (north - south)
                    ring.append([lon, lat])
                else:
                    ring.append([px, py])
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": shape.get("id", f"tree_{idx:04d}"),
                        "label": "tree",
                        "image": image_name,
                        "source_file_name": doc.get("source_file_name", image_name),
                        "tile_id": meta.get("tile_id", ""),
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring],
                    },
                }
            )

        collection = {
            "type": "FeatureCollection",
            "name": Path(image_name).stem,
            "features": features,
            "properties": {
                "coordinate_space": "EPSG:4326" if has_geo else "pixel",
                "generated_at": utc_now(),
            },
        }
        path = self.geojson_path(image_name)
        path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
        return path


class LabelingHandler(SimpleHTTPRequestHandler):
    store: LabelStore

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[labeling_app] {self.address_string()} - {format % args}")

    def send_json(self, data: dict[str, Any] | list[Any], status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"ok": False, "error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self.serve_file(STATIC_ROOT / "index.html")
        if path == "/api/health":
            return self.send_json({"ok": True, "tiles": len(self.store.list_tiles())})
        if path == "/api/tiles":
            return self.send_json({"tiles": self.store.list_tiles()})
        if path == "/api/labels":
            params = parse_qs(parsed.query)
            image_name = params.get("image", [""])[0]
            try:
                image_path = self.store.image_path(image_name)
            except FileNotFoundError:
                return self.send_error_json("Unknown image", HTTPStatus.NOT_FOUND)
            width, height = read_png_size(image_path)
            return self.send_json(self.store.load_labels(image_path.name, width, height))
        if path.startswith("/images/"):
            image_name = unquote(path.removeprefix("/images/"))
            try:
                image_path = self.store.image_path(image_name)
            except FileNotFoundError:
                return self.send_error_json("Unknown image", HTTPStatus.NOT_FOUND)
            return self.serve_file(image_path)
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            return self.serve_file((STATIC_ROOT / rel).resolve())
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/labels":
            return self.send_error_json("Unknown endpoint", HTTPStatus.NOT_FOUND)
        params = parse_qs(parsed.query)
        image_name = params.get("image", [""])[0]
        try:
            image_path = self.store.image_path(image_name)
        except FileNotFoundError:
            return self.send_error_json("Unknown image", HTTPStatus.NOT_FOUND)

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return self.send_error_json("Invalid JSON payload")

        result = self.store.save_labels(image_path.name, payload)
        self.send_json(result)

    def serve_file(self, path: Path) -> None:
        resolved = path.resolve()
        allowed_roots = [STATIC_ROOT.resolve(), self.store.image_dir.resolve()]
        if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not resolved.exists() or not resolved.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RS-treecanopy labeling app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--images",
        type=Path,
        default=PROJECT_ROOT / "data/processed/labeling_candidates_512_random_seoul_urban/images",
        help="Directory containing PNG candidate tiles.",
    )
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=PROJECT_ROOT / "data/processed/labeling_candidates_512_random_seoul_urban/selected_tiles.csv",
        help="CSV with tile metadata for selected candidates.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data/processed/labels_512_random_seoul_urban",
        help="Output directory for JSON, YOLO-seg, and GeoJSON labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.images.resolve()
    if not image_dir.exists():
        raise SystemExit(f"Image directory does not exist: {image_dir}")
    LabelingHandler.store = LabelStore(
        image_dir=image_dir,
        selected_csv=args.selected_csv.resolve(),
        label_root=args.labels.resolve(),
    )
    server = ThreadingHTTPServer((args.host, args.port), LabelingHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Labeling app running at {url}")
    print(f"Images: {image_dir}")
    print(f"Labels: {args.labels.resolve()}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeling app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
