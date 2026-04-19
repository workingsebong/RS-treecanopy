#!/usr/bin/env python3
"""Build a random urban V-World sample set for tree crown labeling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import array_bounds, from_bounds
from rasterio.windows import Window
import requests
from shapely.geometry import box

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AREA_NAME = "random_seoul_urban"
DEFAULT_SEED = 20260419
DEFAULT_PATCH_COUNT = 6
DEFAULT_PATCH_WIDTH_LON = 0.005
DEFAULT_PATCH_HEIGHT_LAT = 0.0045
DEFAULT_TILE_SIZE = 512
DEFAULT_TOP_N = 36
DEFAULT_MAX_CANDIDATES_PER_PATCH = 8
DEFAULT_EXCLUDE_OVERLAP_THRESHOLD = 0.05
DEFAULT_MAX_PATCH_SAMPLING_ATTEMPTS = 500
ZOOM = 19
VWORLD_TILE_SIZE = 256
DEFAULT_EXCLUDE_FILENAMES = ("UPIS_C_UQ153.shp", "UPIS_C_UQ142.shp")

TILE_URL = "https://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/{z}/{y}/{x}.jpeg"
CAPABILITIES_URL = "https://api.vworld.kr/req/wmts/1.0.0/{key}/WMTSCapabilities.xml"

# Hand-picked urban pools avoid obvious mountains and wide water. The final
# patch inside each pool is random and reproducible via seed.
URBAN_POOLS = [
    {
        "name": "mapo_hongdae_yeonnam",
        "min_lon": 126.912,
        "min_lat": 37.548,
        "max_lon": 126.936,
        "max_lat": 37.566,
    },
    {
        "name": "seodaemun_ehwa_sinchon",
        "min_lon": 126.936,
        "min_lat": 37.555,
        "max_lon": 126.957,
        "max_lat": 37.566,
    },
    {
        "name": "jung_euljiro_chungmuro",
        "min_lon": 126.985,
        "min_lat": 37.557,
        "max_lon": 127.010,
        "max_lat": 37.570,
    },
    {
        "name": "seongdong_wangsimni",
        "min_lon": 127.030,
        "min_lat": 37.545,
        "max_lon": 127.055,
        "max_lat": 37.565,
    },
    {
        "name": "gangnam_teheran",
        "min_lon": 127.025,
        "min_lat": 37.497,
        "max_lon": 127.055,
        "max_lat": 37.510,
    },
    {
        "name": "songpa_jamsil",
        "min_lon": 127.075,
        "min_lat": 37.500,
        "max_lon": 127.105,
        "max_lat": 37.515,
    },
    {
        "name": "yeongdeungpo_mullae",
        "min_lon": 126.885,
        "min_lat": 37.505,
        "max_lon": 126.910,
        "max_lat": 37.525,
    },
    {
        "name": "guro_digital",
        "min_lon": 126.880,
        "min_lat": 37.475,
        "max_lon": 126.905,
        "max_lat": 37.495,
    },
    {
        "name": "dongdaemun_jangan",
        "min_lon": 127.055,
        "min_lat": 37.565,
        "max_lon": 127.080,
        "max_lat": 37.585,
    },
]


@dataclass
class PatchSpec:
    patch_id: str
    pool_name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    exclude_overlap_ratio: float = 0.0
    sampling_attempts: int = 1


@dataclass
class ExclusionMask:
    geometry: Any
    source_paths: list[Path]
    source_details: list[dict[str, Any]]
    feature_count: int
    bounds_wgs84: tuple[float, float, float, float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lon_to_tile_x(lon: float, zoom: int) -> int:
    return int((lon + 180) / 360 * 2**zoom)


def lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_r = math.radians(lat)
    return int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * 2**zoom)


def tile_to_lon(x: int, zoom: int) -> float:
    return x / 2**zoom * 360 - 180


def tile_to_lat(y: int, zoom: int) -> float:
    n = math.pi - 2 * math.pi * y / 2**zoom
    return math.degrees(math.atan(math.sinh(n)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_api_key() -> str:
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("VWORLD_API_KEY")
    if not key or key == "YOUR_VWORLD_API_KEY":
        raise RuntimeError("Set a valid VWORLD_API_KEY in .env before running this script.")
    return key


def find_default_exclusion_paths() -> list[Path]:
    paths: list[Path] = []
    data_raw = PROJECT_ROOT / "data/raw"
    if not data_raw.exists():
        return paths
    for filename in DEFAULT_EXCLUDE_FILENAMES:
        matches = sorted(data_raw.rglob(filename))
        if matches:
            paths.append(matches[0])
    return paths


def resolve_exclusion_paths(paths: list[Path] | None) -> list[Path]:
    if paths is None:
        paths = find_default_exclusion_paths()
    resolved: list[Path] = []
    for path in paths:
        full_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not full_path.exists():
            raise FileNotFoundError(f"Exclusion geometry does not exist: {full_path}")
        resolved.append(full_path.resolve())
    return resolved


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return gdf
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def load_exclusion_mask(paths: list[Path] | None) -> ExclusionMask | None:
    source_paths = resolve_exclusion_paths(paths)
    if not source_paths:
        return None

    frames: list[gpd.GeoDataFrame] = []
    source_details: list[dict[str, Any]] = []
    for path in source_paths:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"Exclusion geometry has no CRS: {path}")
        source_details.append(
            {
                "path": str(path),
                "feature_count": int(len(gdf)),
                "source_crs": str(gdf.crs),
            }
        )
        gdf = clean_geometries(gdf)
        if gdf.empty:
            continue
        frames.append(gdf[["geometry"]].to_crs("EPSG:4326"))

    if not frames:
        return None

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    if hasattr(combined.geometry, "union_all"):
        geometry = combined.geometry.union_all()
    else:
        geometry = combined.geometry.unary_union
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    return ExclusionMask(
        geometry=geometry,
        source_paths=source_paths,
        source_details=source_details,
        feature_count=int(len(combined)),
        bounds_wgs84=tuple(float(v) for v in geometry.bounds),
    )


def geometry_overlap_ratio(bounds: tuple[float, float, float, float], mask: ExclusionMask | None) -> float:
    if mask is None:
        return 0.0
    west, south, east, north = bounds
    geom = box(west, south, east, north)
    if geom.is_empty or geom.area <= 0 or not geom.intersects(mask.geometry):
        return 0.0
    return float(min(1.0, geom.intersection(mask.geometry).area / geom.area))


def overlaps_exclusion(ratio: float, threshold: float) -> bool:
    if threshold <= 0:
        return ratio > 0
    return ratio >= threshold


def write_exclusion_mask(mask: ExclusionMask | None, output_path: Path) -> str | None:
    if mask is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"name": ["parks_and_urban_natural_parks"]},
        geometry=[mask.geometry],
        crs="EPSG:4326",
    )
    gdf.to_file(output_path, driver="GeoJSON")
    return str(output_path)


def exclusion_mask_metadata(
    mask: ExclusionMask | None,
    overlap_threshold: float,
    output_geojson: str | None,
) -> dict[str, Any]:
    if mask is None:
        return {
            "enabled": False,
            "reason": "No exclusion geometry paths were found or provided.",
        }
    west, south, east, north = mask.bounds_wgs84
    return {
        "enabled": True,
        "target_crs": "EPSG:4326",
        "source_paths": [str(path) for path in mask.source_paths],
        "source_details": mask.source_details,
        "feature_count_after_cleaning": mask.feature_count,
        "dissolve_method": "union of valid geometries",
        "overlap_threshold": overlap_threshold,
        "excluded_if": "patch/tile bbox overlap ratio with mask is greater than or equal to the threshold",
        "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "output_geojson": output_geojson,
    }


def sample_patches(args: argparse.Namespace, exclusion_mask: ExclusionMask | None) -> list[PatchSpec]:
    rng = random.Random(args.seed)
    pools = URBAN_POOLS.copy()
    if args.patch_count <= len(pools):
        selected_pools = rng.sample(pools, args.patch_count)
    else:
        selected_pools = [rng.choice(pools) for _ in range(args.patch_count)]

    patches: list[PatchSpec] = []
    for idx, pool in enumerate(selected_pools, start=1):
        accepted_patch: PatchSpec | None = None
        for attempt in range(1, args.max_patch_sampling_attempts + 1):
            min_center_lon = pool["min_lon"] + args.patch_width_lon / 2
            max_center_lon = pool["max_lon"] - args.patch_width_lon / 2
            min_center_lat = pool["min_lat"] + args.patch_height_lat / 2
            max_center_lat = pool["max_lat"] - args.patch_height_lat / 2
            if min_center_lon > max_center_lon:
                center_lon = (pool["min_lon"] + pool["max_lon"]) / 2
            else:
                center_lon = rng.uniform(min_center_lon, max_center_lon)
            if min_center_lat > max_center_lat:
                center_lat = (pool["min_lat"] + pool["max_lat"]) / 2
            else:
                center_lat = rng.uniform(min_center_lat, max_center_lat)

            min_lon = center_lon - args.patch_width_lon / 2
            max_lon = center_lon + args.patch_width_lon / 2
            min_lat = center_lat - args.patch_height_lat / 2
            max_lat = center_lat + args.patch_height_lat / 2
            overlap_ratio = geometry_overlap_ratio((min_lon, min_lat, max_lon, max_lat), exclusion_mask)
            if overlaps_exclusion(overlap_ratio, args.exclude_overlap_threshold):
                continue
            accepted_patch = PatchSpec(
                patch_id=f"p{idx:02d}_{pool['name']}",
                pool_name=pool["name"],
                min_lon=round(min_lon, 9),
                min_lat=round(min_lat, 9),
                max_lon=round(max_lon, 9),
                max_lat=round(max_lat, 9),
                exclude_overlap_ratio=round(overlap_ratio, 6),
                sampling_attempts=attempt,
            )
            break
        if accepted_patch is None:
            raise RuntimeError(
                f"Could not sample a patch from pool {pool['name']} outside the exclusion mask "
                f"after {args.max_patch_sampling_attempts} attempts."
            )
        patches.append(accepted_patch)
    return patches


def get_capabilities_metadata(session: requests.Session, api_key: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "capabilities_url_template": CAPABILITIES_URL.replace("{key}", "{key}"),
    }
    try:
        resp = session.get(CAPABILITIES_URL.format(key=api_key), timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {
            "wmts": "http://www.opengis.net/wmts/1.0",
            "ows": "http://www.opengis.net/ows/1.1",
        }
        service = root.find("ows:ServiceIdentification", ns)
        if service is not None:
            for key_name in ["Title", "Abstract", "ServiceType", "ServiceTypeVersion", "Fees", "AccessConstraints"]:
                el = service.find(f"ows:{key_name}", ns)
                metadata[f"service_{key_name.lower()}"] = (el.text or "").strip() if el is not None else None
        contents = root.find("wmts:Contents", ns)
        if contents is not None:
            for layer in contents.findall("wmts:Layer", ns):
                identifier = layer.find("ows:Identifier", ns)
                if identifier is None or identifier.text != "Satellite":
                    continue
                wgs84_bbox = layer.find("ows:WGS84BoundingBox", ns)
                lower = wgs84_bbox.find("ows:LowerCorner", ns).text.split() if wgs84_bbox is not None else None
                upper = wgs84_bbox.find("ows:UpperCorner", ns).text.split() if wgs84_bbox is not None else None
                resource = layer.find("wmts:ResourceURL", ns)
                resource_template = resource.attrib.get("template") if resource is not None else None
                if resource_template:
                    resource_template = resource_template.replace(api_key, "{key}")
                metadata["satellite_layer"] = {
                    "title": (layer.find("ows:Title", ns).text or "").strip(),
                    "identifier": "Satellite",
                    "format": (layer.find("wmts:Format", ns).text or "").strip(),
                    "tile_matrix_set": (layer.find("wmts:TileMatrixSetLink/wmts:TileMatrixSet", ns).text or "").strip(),
                    "wgs84_bbox": {
                        "west": float(lower[0]),
                        "south": float(lower[1]),
                        "east": float(upper[0]),
                        "north": float(upper[1]),
                    } if lower and upper else None,
                    "resource_url_template": resource_template,
                }
                break
    except Exception as exc:  # noqa: BLE001
        metadata["capabilities_error"] = str(exc)
    return metadata


def download_patch(
    patch: PatchSpec,
    raw_patch_dir: Path,
    session: requests.Session,
    api_key: str,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    x_min = lon_to_tile_x(patch.min_lon, ZOOM)
    x_max = lon_to_tile_x(patch.max_lon, ZOOM)
    y_min = lat_to_tile_y(patch.max_lat, ZOOM)
    y_max = lat_to_tile_y(patch.min_lat, ZOOM)
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1
    total = n_x * n_y
    canvas = np.zeros((n_y * VWORLD_TILE_SIZE, n_x * VWORLD_TILE_SIZE, 3), dtype=np.uint8)
    failed_tiles: list[dict[str, Any]] = []
    sample_headers = None
    sample_exif_keys = None
    sample_info_keys = None

    print(f"{patch.patch_id}: {n_x} x {n_y} = {total} V-World tiles")
    downloaded = 0
    for row, y in enumerate(range(y_min, y_max + 1)):
        for col, x in enumerate(range(x_min, x_max + 1)):
            url = TILE_URL.format(key=api_key, z=ZOOM, y=y, x=x)
            try:
                resp = session.get(url, timeout=10)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type.lower():
                    raise ValueError(f"non-image response ({content_type}): {resp.text[:160]}")
                pil_img = Image.open(BytesIO(resp.content))
                if sample_headers is None:
                    sample_headers = {
                        k: v for k, v in resp.headers.items()
                        if k.lower() in {"date", "last-modified", "etag", "cache-control", "content-type"}
                    }
                    sample_exif_keys = list(pil_img.getexif().keys())
                    sample_info_keys = sorted(pil_img.info.keys())
                img = np.array(pil_img.convert("RGB"))
                r0, r1 = row * VWORLD_TILE_SIZE, (row + 1) * VWORLD_TILE_SIZE
                c0, c1 = col * VWORLD_TILE_SIZE, (col + 1) * VWORLD_TILE_SIZE
                canvas[r0:r1, c0:c1] = img
            except Exception as exc:  # noqa: BLE001
                failed_tiles.append({"x": x, "y": y, "row": row, "col": col, "error": str(exc)})
            downloaded += 1
            if downloaded % 20 == 0 or downloaded == total:
                print(f"  {downloaded}/{total} tiles done", end="\r")
    print()

    successful_tiles = total - len(failed_tiles)
    if successful_tiles == 0:
        raise RuntimeError(f"All tile downloads failed for {patch.patch_id}.")

    west = tile_to_lon(x_min, ZOOM)
    east = tile_to_lon(x_max + 1, ZOOM)
    north = tile_to_lat(y_min, ZOOM)
    south = tile_to_lat(y_max + 1, ZOOM)
    transform = from_bounds(west, south, east, north, canvas.shape[1], canvas.shape[0])

    tif_path = raw_patch_dir / "patches" / f"{patch.patch_id}.tif"
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=canvas.shape[0],
        width=canvas.shape[1],
        count=3,
        dtype=np.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="lzw",
    ) as dst:
        for band in range(3):
            dst.write(canvas[:, :, band], band + 1)

    metadata = {
        "patch": asdict(patch),
        "provider": "V-World",
        "service": "OGC WMTS",
        "layer": "Satellite",
        "zoom": ZOOM,
        "tile_size": VWORLD_TILE_SIZE,
        "actual_tile_bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "tile_range": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "tile_grid": {"n_x": n_x, "n_y": n_y, "total": total},
        "download": {
            "downloaded_at_utc": utc_now(),
            "successful_tiles": successful_tiles,
            "failed_tiles": len(failed_tiles),
            "failures_preview": failed_tiles[:50],
        },
        "output": {
            "path": str(tif_path),
            "width": int(canvas.shape[1]),
            "height": int(canvas.shape[0]),
            "bands": 3,
            "dtype": "uint8",
            "crs": "EPSG:4326",
        },
        "sample_tile_response": {
            "http_headers": sample_headers,
            "jpeg_exif_keys": sample_exif_keys,
            "pil_info_keys": sample_info_keys,
        },
        "available_source_metadata": source_metadata,
        "acquisition_metadata": {
            "acquisition_date": None,
            "sensor": None,
            "note": "V-World WMTS Satellite tiles and WMTSCapabilities did not expose per-tile acquisition date or sensor metadata. Sampled JPEG tiles had no EXIF keys.",
        },
    }
    metadata_path = raw_patch_dir / "metadata" / f"{patch.patch_id}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def split_patches(
    patch_metadata: list[dict[str, Any]],
    tile_root: Path,
    tile_size: int,
    exclusion_mask: ExclusionMask | None,
    exclude_overlap_threshold: float,
) -> pd.DataFrame:
    image_dir = tile_root / "images"
    preview_dir = tile_root / "preview"
    image_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old_file in image_dir.glob("*.png"):
        old_file.unlink()

    records: list[dict[str, Any]] = []
    for patch_meta in patch_metadata:
        patch_id = patch_meta["patch"]["patch_id"]
        pool_name = patch_meta["patch"]["pool_name"]
        tif_path = Path(patch_meta["output"]["path"])
        with rasterio.open(tif_path) as src:
            n_cols = math.ceil(src.width / tile_size)
            n_rows = math.ceil(src.height / tile_size)
            for row in range(n_rows):
                for col in range(n_cols):
                    x_off = col * tile_size
                    y_off = row * tile_size
                    valid_width = min(tile_size, src.width - x_off)
                    valid_height = min(tile_size, src.height - y_off)
                    window = Window(x_off, y_off, valid_width, valid_height)
                    data = src.read([1, 2, 3], window=window)
                    data = np.moveaxis(data, 0, -1)
                    tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    tile[:valid_height, :valid_width, :] = data.astype(np.uint8)
                    tile_name = f"{patch_id}_tile{tile_size}_r{row:02d}_c{col:02d}.png"
                    Image.fromarray(tile).save(image_dir / tile_name)

                    tile_transform = rasterio.windows.transform(window, src.transform)
                    west, south, east, north = array_bounds(valid_height, valid_width, tile_transform)
                    exclude_overlap_ratio = geometry_overlap_ratio((west, south, east, north), exclusion_mask)
                    records.append(
                        {
                            "patch_id": patch_id,
                            "pool_name": pool_name,
                            "tile_id": f"{patch_id}_r{row:02d}_c{col:02d}",
                            "file_name": tile_name,
                            "row": row,
                            "col": col,
                            "x_off": x_off,
                            "y_off": y_off,
                            "tile_size": tile_size,
                            "valid_width": valid_width,
                            "valid_height": valid_height,
                            "is_edge": valid_width < tile_size or valid_height < tile_size,
                            "crs": src.crs.to_string() if src.crs else None,
                            "west": west,
                            "south": south,
                            "east": east,
                            "north": north,
                            "exclude_overlap_ratio": exclude_overlap_ratio,
                            "excluded_by_geometry": overlaps_exclusion(
                                exclude_overlap_ratio,
                                exclude_overlap_threshold,
                            ),
                        }
                    )
    index = pd.DataFrame(records)
    index.to_csv(tile_root / "tile_index.csv", index=False)
    summary = {
        "tile_size": tile_size,
        "n_patches": int(index["patch_id"].nunique()) if not index.empty else 0,
        "n_tiles": int(len(index)),
        "edge_tiles": int(index["is_edge"].sum()) if not index.empty else 0,
        "geometry_excluded_tiles": int(index["excluded_by_geometry"].sum()) if not index.empty else 0,
        "exclude_overlap_threshold": exclude_overlap_threshold,
        "image_dir": str(image_dir),
    }
    (tile_root / "tile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return index


def score_tile(image_path: Path, valid_width: int, valid_height: int) -> dict[str, float | bool]:
    arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    valid = arr[:valid_height, :valid_width, :]
    r = valid[..., 0]
    g = valid[..., 1]
    b = valid[..., 2]
    brightness = valid.mean(axis=2)
    hsv = rgb_to_hsv(valid / 255.0)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    green_mask = (hue > 0.20) & (hue < 0.42) & (saturation > 0.18) & (value > 0.20)
    exg = 2 * g - r - b
    grad_y, grad_x = np.gradient(brightness)
    texture = np.sqrt(grad_x**2 + grad_y**2)

    vegetation_ratio = float(green_mask.mean())
    green_texture_mean = float(texture[green_mask].mean()) if green_mask.any() else 0.0
    texture_mean = float(texture.mean())
    dark_ratio = float((brightness < 35).mean())
    bright_ratio = float((brightness > 245).mean())
    mean_exg = float(exg.mean())
    forest_like = vegetation_ratio > 0.88 or (vegetation_ratio > 0.78 and texture_mean < 5.0)

    score = (
        vegetation_ratio * 100
        + min(green_texture_mean, 18) * 1.25
        + min(texture_mean, 18) * 0.35
        + max(mean_exg, 0) * 0.02
        - dark_ratio * 6
        - bright_ratio * 4
    )
    if forest_like:
        score -= 25

    return {
        "vegetation_ratio": vegetation_ratio,
        "green_texture_mean": green_texture_mean,
        "texture_mean": texture_mean,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "mean_exg": mean_exg,
        "forest_like": forest_like,
        "candidate_score": float(score),
    }


def select_candidates(
    tile_index: pd.DataFrame,
    tile_root: Path,
    candidate_root: Path,
    top_n: int,
    max_per_patch: int,
) -> pd.DataFrame:
    image_dir = tile_root / "images"
    output_image_dir = candidate_root / "images"
    preview_dir = candidate_root / "preview"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_image_dir.glob("*.png"):
        old_file.unlink()

    scores = []
    for _, rec in tile_index.iterrows():
        metrics = score_tile(
            image_dir / rec["file_name"],
            valid_width=int(rec["valid_width"]),
            valid_height=int(rec["valid_height"]),
        )
        scores.append({**rec.to_dict(), **metrics})

    scores_df = pd.DataFrame(scores).sort_values("candidate_score", ascending=False).reset_index(drop=True)
    scores_df["score_rank"] = range(1, len(scores_df) + 1)
    scores_df.to_csv(candidate_root / "all_tile_scores.csv", index=False)

    selected_rows: list[dict[str, Any]] = []
    per_patch: dict[str, int] = {}
    for _, row in scores_df.iterrows():
        excluded_by_geometry = bool(row.get("excluded_by_geometry", False))
        if bool(row["is_edge"]) or bool(row["forest_like"]) or excluded_by_geometry:
            continue
        patch_id = str(row["patch_id"])
        if per_patch.get(patch_id, 0) >= max_per_patch:
            continue
        row_dict = row.to_dict()
        selected_rows.append(row_dict)
        per_patch[patch_id] = per_patch.get(patch_id, 0) + 1
        if len(selected_rows) >= top_n:
            break

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        raise RuntimeError("No eligible candidate tiles were selected.")
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected.to_csv(candidate_root / "selected_tiles.csv", index=False)

    for _, rec in selected.iterrows():
        src = image_dir / rec["file_name"]
        dst = output_image_dir / f"rank{int(rec['selection_rank']):02d}_{rec['file_name']}"
        shutil.copy2(src, dst)

    write_selected_contact_sheet(selected, image_dir, candidate_root / "preview" / "selected_contact_sheet.png")
    return selected


def write_selected_contact_sheet(selected: pd.DataFrame, image_dir: Path, output_path: Path) -> None:
    n_cols = 6
    n_rows = int(math.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.55))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, rec) in zip(axes.ravel(), selected.iterrows()):
        img = Image.open(image_dir / rec["file_name"])
        ax.imshow(img)
        ax.set_title(
            f"#{int(rec['selection_rank'])} {rec['tile_id']}\n"
            f"veg={rec['vegetation_ratio']:.2f}, score={rec['candidate_score']:.1f}",
            fontsize=7,
        )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random urban V-World sampling for 512px tree crown labels.")
    parser.add_argument("--area-name", default=DEFAULT_AREA_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patch-count", type=int, default=DEFAULT_PATCH_COUNT)
    parser.add_argument("--patch-width-lon", type=float, default=DEFAULT_PATCH_WIDTH_LON)
    parser.add_argument("--patch-height-lat", type=float, default=DEFAULT_PATCH_HEIGHT_LAT)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-candidates-per-patch", type=int, default=DEFAULT_MAX_CANDIDATES_PER_PATCH)
    parser.add_argument(
        "--exclude-geometry",
        type=Path,
        nargs="*",
        default=None,
        help="Park/forest polygons to exclude. Defaults to UPIS_C_UQ153.shp and UPIS_C_UQ142.shp under data/raw.",
    )
    parser.add_argument(
        "--no-exclude-geometry",
        action="store_true",
        help="Disable park/forest exclusion geometry even when default UPIS shapefiles exist.",
    )
    parser.add_argument(
        "--exclude-overlap-threshold",
        type=float,
        default=DEFAULT_EXCLUDE_OVERLAP_THRESHOLD,
        help="Reject a random patch or tile when this fraction of its bbox overlaps the exclusion mask.",
    )
    parser.add_argument(
        "--max-patch-sampling-attempts",
        type=int,
        default=DEFAULT_MAX_PATCH_SAMPLING_ATTEMPTS,
        help="Maximum random retries per urban pool when sampled patches overlap the exclusion mask.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.exclude_overlap_threshold < 0 or args.exclude_overlap_threshold > 1:
        raise ValueError("--exclude-overlap-threshold must be between 0 and 1.")

    api_key = get_api_key()
    raw_root = PROJECT_ROOT / "data/raw" / args.area_name
    tile_root = PROJECT_ROOT / f"data/processed/tiles_{args.tile_size}_{args.area_name}"
    candidate_root = PROJECT_ROOT / f"data/processed/labeling_candidates_{args.tile_size}_{args.area_name}"
    label_root = PROJECT_ROOT / f"data/processed/labels_{args.tile_size}_{args.area_name}"
    label_root.mkdir(parents=True, exist_ok=True)
    (label_root / "classes.txt").write_text("tree\n", encoding="utf-8")

    exclusion_mask = None if args.no_exclude_geometry else load_exclusion_mask(args.exclude_geometry)
    exclusion_mask_geojson = write_exclusion_mask(exclusion_mask, raw_root / "exclude_mask.geojson")
    exclusion_metadata = exclusion_mask_metadata(
        exclusion_mask,
        overlap_threshold=args.exclude_overlap_threshold,
        output_geojson=exclusion_mask_geojson,
    )
    if exclusion_mask is None:
        print("Exclusion mask: disabled")
    else:
        print(
            "Exclusion mask: "
            f"{exclusion_mask.feature_count} features, threshold={args.exclude_overlap_threshold:.2f}, "
            f"geojson={exclusion_mask_geojson}"
        )

    session = requests.Session()
    source_metadata = get_capabilities_metadata(session, api_key)
    patches = sample_patches(args, exclusion_mask)
    write_csv(raw_root / "sampling_patches.csv", [asdict(p) for p in patches])
    print("Selected random patches:")
    for patch in patches:
        print(
            f"  {patch.patch_id}: {patch.min_lon}, {patch.min_lat}, {patch.max_lon}, {patch.max_lat} "
            f"(exclude_overlap={patch.exclude_overlap_ratio:.4f}, attempts={patch.sampling_attempts})"
        )

    patch_metadata = []
    for patch in patches:
        patch_metadata.append(download_patch(patch, raw_root, session, api_key, source_metadata))

    dataset_metadata = {
        "area_name": args.area_name,
        "seed": args.seed,
        "patch_count": args.patch_count,
        "patch_width_lon": args.patch_width_lon,
        "patch_height_lat": args.patch_height_lat,
        "exclude_overlap_threshold": args.exclude_overlap_threshold,
        "provider": "V-World",
        "service": "OGC WMTS",
        "layer": "Satellite",
        "zoom": ZOOM,
        "tile_size_for_labeling": args.tile_size,
        "exclusion_mask": exclusion_metadata,
        "patches": [metadata["patch"] for metadata in patch_metadata],
        "source_metadata": source_metadata,
        "acquisition_metadata": {
            "acquisition_date": None,
            "sensor": None,
            "note": "V-World WMTS does not expose per-tile acquisition date or sensor metadata. Use NGII orthoimage source data and orthoimage achievement metadata when exact acquisition/production metadata is required.",
        },
        "generated_at_utc": utc_now(),
    }
    (raw_root / "dataset_metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tile_index = split_patches(
        patch_metadata,
        tile_root,
        args.tile_size,
        exclusion_mask=exclusion_mask,
        exclude_overlap_threshold=args.exclude_overlap_threshold,
    )
    selected = select_candidates(
        tile_index,
        tile_root,
        candidate_root,
        top_n=args.top_n,
        max_per_patch=args.max_candidates_per_patch,
    )

    print("\nDone.")
    print(f"Raw patches: {raw_root}")
    excluded_tiles = int(tile_index["excluded_by_geometry"].sum()) if not tile_index.empty else 0
    print(f"Tiles: {tile_root} ({len(tile_index)} tiles, {excluded_tiles} geometry-excluded)")
    print(f"Candidates: {candidate_root} ({len(selected)} selected)")
    print(f"Labels: {label_root}")


if __name__ == "__main__":
    main()
