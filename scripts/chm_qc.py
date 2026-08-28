"""
OPTIONAL QC utility (brief section 13): derive CHM = DSM - DTM and check whether
detected crowns sit on plausible canopy height, to spot obvious vegetation false
positives (e.g. a "crown" detection sitting over a CHM of ~0.5m is very likely a
weed clump, not a mature palm which is typically several metres tall).

This does NOT run automatically as part of the production pipeline — only use it
if error analysis on the pilot block shows CHM would meaningfully cut false
positives. Requires dsm/dtm paths to be set in configs/blocks.yaml for the block.

Usage:
    python scripts/chm_qc.py --block pilot --min-height-m 1.5
"""
import argparse

import numpy as np
import rasterio
from rasterio.sample import sample_gen
from pyproj import Transformer
import geopandas as gpd

from common import load_config, get_block, resolve_path, PROJECT_ROOT


def sample_raster(ds, coords, src_crs):
    """Sample a raster at coords given in src_crs, reprojecting if the raster differs.

    Elevation products don't always ship in the project CRS — block_003's DTM arrived in
    EPSG:4326 (lat/lon degrees) while its RGB/DSM were EPSG:32631. Sampling that with UTM
    coordinates silently returns nodata for every point (the coords fall far outside the
    raster's degree-valued bounds), which would look like "no valid CHM data" rather than a
    CRS bug. Transforming the sample points is far cheaper than warping a multi-GB raster.
    """
    if ds.crs is not None and str(ds.crs) != str(src_crs):
        transformer = Transformer.from_crs(src_crs, ds.crs, always_xy=True)
        coords = [transformer.transform(x, y) for x, y in coords]
        print(f"  [note] reprojected sample points {src_crs} -> {ds.crs} for {ds.name}", flush=True)
    return np.array([v[0] for v in sample_gen(ds, coords)], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--min-height-m", type=float, default=1.5,
                     help="crowns whose centroid CHM is below this are flagged as likely FPs")
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)
    if not block.get("dsm") or not block.get("dtm"):
        raise SystemExit(f"Block '{args.block}' has no dsm/dtm configured in blocks.yaml — skip CHM QC")

    dsm_path, dtm_path = resolve_path(block["dsm"]), resolve_path(block["dtm"])
    gpkg_path = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "inventory" / f"{args.block}_palm_inventory.gpkg"
    centroids = gpd.read_file(gpkg_path, layer="centroids")

    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        coords = [(pt.x, pt.y) for pt in centroids.geometry]
        dsm_vals = sample_raster(dsm_ds, coords, centroids.crs)
        dtm_vals = sample_raster(dtm_ds, coords, centroids.crs)
        dsm_nodata, dtm_nodata = dsm_ds.nodata, dtm_ds.nodata

    # Drop points where either surface has no data, so nodata sentinels (-32767, -9999)
    # can't silently poison the CHM difference.
    invalid = np.isnan(dsm_vals) | np.isnan(dtm_vals)
    if dsm_nodata is not None:
        invalid |= dsm_vals == dsm_nodata
    if dtm_nodata is not None:
        invalid |= dtm_vals == dtm_nodata

    chm = dsm_vals - dtm_vals
    chm[invalid] = np.nan
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"{n_invalid} / {len(chm)} crowns have no DSM/DTM coverage — left unflagged (QA_Status='no_chm_data')", flush=True)
    centroids["CHM_m"] = chm
    # NaN < x is False, so points with no CHM coverage would otherwise be silently marked
    # "ok" — label them distinctly instead so they're not mistaken for height-verified crowns.
    centroids["QA_Status"] = np.where(
        np.isnan(chm), "no_chm_data",
        np.where(chm < args.min_height_m, "flagged_low_height", "ok"),
    )

    n_flagged = int((centroids["QA_Status"] == "flagged_low_height").sum())
    print(f"{n_flagged} / {len(centroids)} crowns flagged with CHM < {args.min_height_m} m "
          f"(likely weed/understory false positives — review before removing)", flush=True)

    out_path = gpkg_path.parent / f"{args.block}_palm_inventory_chm_qc.gpkg"
    centroids.to_file(out_path, layer="centroids_chm", driver="GPKG")
    print(f"Written: {out_path} (inspect 'flagged_low_height' in QGIS before deciding to filter)", flush=True)


if __name__ == "__main__":
    main()
