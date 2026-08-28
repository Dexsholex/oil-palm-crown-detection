"""
Draw the model's detections as boxes on a real image crop, for visual QC / client
presentation (brief section 6: "visual detection of individual crowns").

Usage:
    python scripts/visualize_detections.py --block pilot --aoi outputs/qc/pilot_validation_aoi.gpkg --imagery outputs/qc/pilot_validation_aoi_imagery.tif
"""
import argparse

import cv2
import numpy as np
import rasterio
import geopandas as gpd

from common import load_config, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--imagery", required=True, help="georeferenced image crop to draw on")
    ap.add_argument("--aoi", default=None, help="optional AOI polygon to restrict/outline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    inv_path = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "inventory" / f"{args.block}_palm_inventory.gpkg"
    crowns = gpd.read_file(inv_path, layer="crowns")

    with rasterio.open(PROJECT_ROOT / args.imagery) as ds:
        img = np.moveaxis(ds.read()[:3], 0, -1).copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bounds = ds.bounds
        transform = ds.transform

        in_view = crowns.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
        print(f"{len(in_view)} detections fall within this image's extent")

        for _, row in in_view.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            col1, row1 = ~transform * (minx, maxy)
            col2, row2 = ~transform * (maxx, miny)
            x1, y1, x2, y2 = int(col1), int(row1), int(col2), int(row2)
            conf = row["Confidence"]
            color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

        if args.aoi:
            aoi = gpd.read_file(PROJECT_ROOT / args.aoi).to_crs(ds.crs)
            for geom in aoi.geometry:
                coords = [(~transform * (x, y)) for x, y in geom.exterior.coords]
                pts = np.array(coords, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img_bgr, [pts], isClosed=True, color=(255, 0, 0), thickness=6)

    out_path = PROJECT_ROOT / (args.out or f"outputs/qc/{args.block}_detections_overlay.png")
    cv2.imwrite(str(out_path), img_bgr)
    print(f"Written: {out_path}  ({len(in_view)} boxes drawn, green=conf>=0.5, orange=lower)")


if __name__ == "__main__":
    main()
