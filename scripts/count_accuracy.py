"""
Compute counting accuracy (brief section 16) for a validation area with a known
ground-truth palm count: manually count palms in a QGIS polygon/window, spatially
filter the predicted inventory to that same window, and compare.

Usage:
    python scripts/count_accuracy.py --block pilot --actual-count 842
    # or restrict to a specific validation polygon (GeoPackage/Shapefile, single feature):
    python scripts/count_accuracy.py --block pilot --actual-count 210 --aoi validation_area.gpkg
"""
import argparse

import geopandas as gpd

from common import load_config, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--actual-count", type=int, required=True)
    ap.add_argument("--aoi", default=None, help="optional polygon file to restrict comparison to")
    args = ap.parse_args()

    cfg = load_config()
    gpkg_path = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "inventory" / f"{args.block}_palm_inventory.gpkg"
    centroids = gpd.read_file(gpkg_path, layer="centroids")

    if args.aoi:
        aoi = gpd.read_file(args.aoi).to_crs(centroids.crs)
        centroids = gpd.clip(centroids, aoi)

    predicted = len(centroids)
    actual = args.actual_count
    error_pct = abs(predicted - actual) / actual * 100
    accuracy_pct = 100 - error_pct

    print(f"Block:            {args.block}")
    if args.aoi:
        print(f"AOI:              {args.aoi}")
    print(f"Predicted count:  {predicted}")
    print(f"Actual count:     {actual}")
    print(f"Counting error:   {error_pct:.2f}%")
    print(f"Counting accuracy:{accuracy_pct:.2f}%")

    target = 90.0
    if accuracy_pct >= target:
        print(f"\n[PASS] >= {target}% production target")
    else:
        print(f"\n[BELOW TARGET] < {target}% — review false positives/negatives before scaling")


if __name__ == "__main__":
    main()
