"""
Pick a compact validation AOI from within the held-out TEST spatial sectors (never
seen in training or annotation) for independent manual counting, per brief section 16.
Exports the AOI polygon + a real high-res image crop to actually count from.

Usage:
    python scripts/select_validation_aoi.py --block pilot --target-ha 1.5
"""
import argparse
import hashlib
import subprocess

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from common import load_config, get_block, resolve_path, PROJECT_ROOT


def sector_split(sector: str, train=0.70, val=0.15) -> str:
    h = int(hashlib.md5(sector.encode()).hexdigest(), 16) / 16**32
    if h < train:
        return "train"
    if h < train + val:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--target-ha", type=float, default=1.5)
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)
    tile_index = pd.read_csv(PROJECT_ROOT / cfg["paths"]["tiles_dir"] / args.block / "tile_index.csv")

    col_bin = pd.cut(tile_index["col_off"], 4, labels=False)
    row_bin = pd.cut(tile_index["row_off"], 4, labels=False)
    tile_index["sector"] = row_bin.astype(str) + "_" + col_bin.astype(str)
    tile_index["split"] = tile_index["sector"].apply(sector_split)

    test_tiles = tile_index[tile_index["split"] == "test"].copy()
    print(f"{len(test_tiles)} tiles in held-out TEST sectors")

    # world-space side length for target area
    px_size = 0.0213  # m/px, from block GSD
    side_m = (args.target_ha * 10_000) ** 0.5

    # find a test tile, then grow a square AOI centered there, snapped to real coords
    seed_tile = test_tiles.iloc[len(test_tiles) // 2]
    cx = seed_tile["transform_c"] + seed_tile["width"] * px_size / 2
    cy = seed_tile["transform_f"] - seed_tile["height"] * px_size / 2
    half = side_m / 2
    minx, maxx = cx - half, cx + half
    miny, maxy = cy - half, cy + half

    aoi = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame([{"aoi_id": f"{args.block}_validation_aoi", "area_ha": aoi.area / 10_000}],
                            geometry=[aoi], crs=cfg["crs"])

    out_dir = PROJECT_ROOT / "outputs" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    aoi_path = out_dir / f"{args.block}_validation_aoi.gpkg"
    gdf.to_file(aoi_path, driver="GPKG")

    print(f"\nAOI: {minx:.2f}, {miny:.2f} -> {maxx:.2f}, {maxy:.2f}  ({aoi.area/10_000:.2f} ha)")
    print(f"Saved boundary: {aoi_path}")

    # crop a real, full-resolution image from the source orthomosaic for manual counting
    rgb_path = resolve_path(block["rgb"])
    crop_path = out_dir / f"{args.block}_validation_aoi_crop.tif"
    gdal_translate = r"C:\Program Files\QGIS 3.44.7\bin\gdal_translate.exe"
    subprocess.run([
        gdal_translate, "-projwin", str(minx), str(maxy), str(maxx), str(miny),
        str(rgb_path), str(crop_path)
    ], check=True)
    print(f"Saved full-res crop for counting: {crop_path}")

    png_path = out_dir / f"{args.block}_validation_aoi_crop.png"
    subprocess.run([gdal_translate, "-of", "PNG", str(crop_path), str(png_path)], check=True)
    print(f"Saved viewable PNG: {png_path}")


if __name__ == "__main__":
    main()
