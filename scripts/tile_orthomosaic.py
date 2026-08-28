"""
Tile a block's RGB orthomosaic into fixed-size, overlapping image chips for
YOLO annotation/training/inference. Writes:
  tiles/<block_id>/images/<tile_id>.jpg
  tiles/<block_id>/tile_index.csv        (pixel offset + affine transform per tile -> geo-referencing)
  tiles/<block_id>/tile_footprints.gpkg  (tile boundaries, for visual QC in QGIS)

Usage:
    python scripts/tile_orthomosaic.py --block pilot
    python scripts/tile_orthomosaic.py --block pilot --size 1280 --overlap 256
"""
import argparse
import csv

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box

from common import load_config, get_block, resolve_path, PROJECT_ROOT


def make_offsets(total, size, stride):
    """Window start offsets covering [0, total), last window shifted back to stay full-size."""
    if total <= size:
        return [0]
    offs = list(range(0, total - size + 1, stride))
    if offs[-1] != total - size:
        offs.append(total - size)
    return offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--size", type=int, default=None, help="override tiling.size_px")
    ap.add_argument("--overlap", type=int, default=None, help="override tiling.overlap_px")
    ap.add_argument("--limit", type=int, default=None, help="stop after N tiles (debug)")
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)
    size = args.size or cfg["tiling"]["size_px"]
    overlap = args.overlap or cfg["tiling"]["overlap_px"]
    stride = size - overlap
    skip_frac = cfg["tiling"]["nodata_skip_fraction"]

    rgb_path = resolve_path(block["rgb"])
    out_dir = PROJECT_ROOT / cfg["paths"]["tiles_dir"] / block["block_id"]
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    index_rows = []
    footprints = []

    with rasterio.open(rgb_path) as ds:
        row_offs = make_offsets(ds.height, size, stride)
        col_offs = make_offsets(ds.width, size, stride)
        total = len(row_offs) * len(col_offs)
        print(f"Block '{block['block_id']}': {ds.width}x{ds.height}px -> "
              f"{len(col_offs)} x {len(row_offs)} = {total} candidate tiles "
              f"(size={size}, overlap={overlap}, stride={stride})")

        for r in row_offs:
            for c in col_offs:
                if args.limit and kept >= args.limit:
                    break
                window = Window(c, r, size, size)
                data = ds.read(window=window, boundless=True, fill_value=0)  # (bands, h, w)

                rgb = data[:3]
                black_frac = float(np.mean(np.all(rgb == 0, axis=0)))
                if black_frac >= skip_frac:
                    skipped += 1
                    continue

                tile_id = f"{block['block_id']}_r{r}_c{c}"
                chip = np.moveaxis(rgb, 0, -1)  # (h, w, bands) RGB
                chip_bgr = cv2.cvtColor(chip, cv2.COLOR_RGB2BGR)
                out_path = img_dir / f"{tile_id}.jpg"
                cv2.imwrite(str(out_path), chip_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                tile_transform = ds.window_transform(window)
                index_rows.append({
                    "tile_id": tile_id,
                    "block_id": block["block_id"],
                    "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                    "col_off": c, "row_off": r, "width": size, "height": size,
                    "transform_a": tile_transform.a, "transform_b": tile_transform.b,
                    "transform_c": tile_transform.c, "transform_d": tile_transform.d,
                    "transform_e": tile_transform.e, "transform_f": tile_transform.f,
                    "crs": str(ds.crs),
                })

                minx, maxy = tile_transform.c, tile_transform.f
                maxx = minx + size * tile_transform.a
                miny = maxy + size * tile_transform.e  # e is negative
                footprints.append({"tile_id": tile_id, "geometry": box(minx, miny, maxx, maxy)})

                kept += 1
                if kept % 200 == 0:
                    print(f"  ... {kept} tiles written", flush=True)
            if args.limit and kept >= args.limit:
                break

    index_path = out_dir / "tile_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    if footprints:
        gdf = gpd.GeoDataFrame(footprints, crs=cfg["crs"])
        gdf.to_file(out_dir / "tile_footprints.gpkg", driver="GPKG")

    print(f"\nDone. kept={kept} skipped(nodata)={skipped}", flush=True)
    print(f"images:  {img_dir}", flush=True)
    print(f"index:   {index_path}", flush=True)
    print(f"footprints: {out_dir / 'tile_footprints.gpkg'}", flush=True)


if __name__ == "__main__":
    main()
