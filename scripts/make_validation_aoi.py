"""
Carve out a small validation AOI for manual palm counting, to compute real counting
accuracy (brief section 16) against the model's output in the same area.

Picked from a spatial sector assigned to the TEST split (never trained/validated on),
using the SAME sector_split() hash as build_yolo_dataset.py, so the comparison also
reflects true out-of-sample generalization, not an area the model partially saw.

Usage:
    python scripts/make_validation_aoi.py --block pilot --size-m 150
"""
import argparse
import sys

import rasterio
import geopandas as gpd
from shapely.geometry import box as shp_box

sys.path.insert(0, str(__file__.rsplit("\\", 1)[0]))
from build_yolo_dataset import sector_split
from common import load_config, get_block, resolve_path, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--size-m", type=float, default=150, help="AOI side length in metres")
    ap.add_argument("--sectors", type=int, default=4, help="must match sample_tiles_for_annotation.py")
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)
    rgb_path = resolve_path(block["rgb"])

    with rasterio.open(rgb_path) as ds:
        minx, miny, maxx, maxy = ds.bounds
        width_m, height_m = maxx - minx, maxy - miny

    # find a TEST-split sector (same convention as build_yolo_dataset.py: "row_col" string)
    chosen = None
    for row_bin in range(args.sectors):
        for col_bin in range(args.sectors):
            sector = f"{row_bin}_{col_bin}"
            if sector_split(sector) == "test":
                cell_w, cell_h = width_m / args.sectors, height_m / args.sectors
                cell_minx = minx + col_bin * cell_w
                cell_maxy = maxy - row_bin * cell_h  # row 0 = top = maxy
                # center of this cell, avoiding the very edge of the raster
                cx = cell_minx + cell_w / 2
                cy = cell_maxy - cell_h / 2
                chosen = (sector, cx, cy)
                break
        if chosen:
            break

    if not chosen:
        raise SystemExit("No test-split sector found — check sector_split() / sectors count")

    sector, cx, cy = chosen
    half = args.size_m / 2
    aoi = shp_box(cx - half, cy - half, cx + half, cy + half)

    gdf = gpd.GeoDataFrame([{"aoi_id": f"{args.block}_validation_aoi", "sector": sector,
                              "size_m": args.size_m, "split": "test"}],
                            geometry=[aoi], crs=cfg["crs"])
    out_path = PROJECT_ROOT / "outputs" / "qc" / f"{args.block}_validation_aoi.gpkg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")

    print(f"Validation AOI: {args.size_m}x{args.size_m} m, centered at ({cx:.1f}, {cy:.1f})")
    print(f"  from spatial sector '{sector}' (assigned to TEST split — model never trained/validated on this area)")
    print(f"  bounds: minx={cx-half:.1f} miny={cy-half:.1f} maxx={cx+half:.1f} maxy={cy+half:.1f}")
    print(f"Written: {out_path}")
    print(f"\nNext: open in QGIS alongside {rgb_path}, zoom to the AOI polygon, and manually count every "
          f"visible palm crown inside it. Then run:")
    print(f"  scripts/count_accuracy.py --block {args.block} --actual-count <your count> --aoi {out_path}")


if __name__ == "__main__":
    main()
