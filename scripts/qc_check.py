"""
Data QC step: verify a block's RGB orthomosaic (and DSM/DTM if present) before
it enters the pipeline — CRS, resolution, band count, nodata coverage, alignment.

Usage:
    python scripts/qc_check.py --block pilot
"""
import argparse
import sys

import numpy as np
import rasterio

from common import load_config, get_block, resolve_path


def check_raster(path, label, expected_crs=None, crs_is_fatal=True):
    print(f"\n--- {label}: {path} ---")
    if not path.exists():
        print(f"  [FAIL] file not found")
        return False

    ok = True
    with rasterio.open(path) as ds:
        print(f"  size: {ds.width} x {ds.height} px, bands: {ds.count}, dtype: {ds.dtypes[0]}")
        gsd_x, gsd_y = ds.res
        print(f"  pixel size: {gsd_x:.4f} x {gsd_y:.4f} m ({gsd_x*100:.2f} cm/px)")
        print(f"  CRS: {ds.crs}")
        if expected_crs and str(ds.crs) != expected_crs:
            if crs_is_fatal:
                print(f"  [FAIL] CRS mismatch — expected {expected_crs}")
                ok = False
            else:
                print(f"  [WARN] CRS is {ds.crs}, not the project CRS {expected_crs} — "
                      f"pixel size/footprint printed below are in this raster's own units "
                      f"and will look wrong; chm_qc.py reprojects sample points to handle it")

        area_ha = (ds.width * gsd_x) * (ds.height * gsd_y) / 10_000
        print(f"  approx footprint: {area_ha:,.1f} ha")

        # Sample a grid of small windows at NATIVE resolution spread across the raster.
        # Deliberately NOT using out_shape decimation / embedded overviews here: overview
        # pyramids can be stale (seen in practice — coarsest overview showed 68% "black"
        # while the real full-res pixels underneath were normal vegetation values), which
        # would silently produce a false nodata-gap warning. Small native-res windows always
        # read the base image band, bypassing the overview pyramid entirely.
        grid = 12
        win_size = 64
        samples = []
        for gy in range(grid):
            for gx in range(grid):
                row = min(int(ds.height * gy / grid), ds.height - win_size)
                col = min(int(ds.width * gx / grid), ds.width - win_size)
                w = ds.read(window=rasterio.windows.Window(col, row, win_size, win_size))
                samples.append(w[:3] if w.shape[0] >= 3 else w)
        small = np.concatenate(samples, axis=2)  # stack along width

        nodata_val = ds.nodata
        if nodata_val is not None:
            nodata_frac = float(np.mean(small[0] == nodata_val))
        else:
            # heuristic: treat all-bands-zero as nodata/black border
            nodata_frac = float(np.mean(np.all(small == 0, axis=0)))
        print(f"  estimated nodata/black-border fraction: {nodata_frac:.2%}")
        if nodata_frac > 0.5:
            print(f"  [WARN] large nodata fraction — check mosaic extent/clipping")

        for b in range(min(ds.count, 3)):
            band = small[b]
            valid = band[band != nodata_val] if nodata_val is not None else band
            if valid.size == 0:
                print(f"  band {b+1}: all nodata in sample")
                continue
            print(f"  band {b+1} (nodata excluded): min={valid.min()} max={valid.max()} mean={valid.mean():.1f}")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, help="block_id from configs/blocks.yaml")
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)

    print(f"=== QC report: block '{block['block_id']}' ===")
    all_ok = True
    all_ok &= check_raster(resolve_path(block["rgb"]), "RGB orthomosaic", cfg["crs"])

    # DSM/DTM CRS is checked too (warn, not fail): chm_qc.py reprojects sample points when
    # these differ from the project CRS, so a mismatch is handled — but it must still be
    # visible here rather than passing silently, since a wrong-CRS elevation raster otherwise
    # only shows up as nonsense pixel sizes / footprints buried in the report.
    if block.get("dsm"):
        check_raster(resolve_path(block["dsm"]), "DSM", cfg["crs"], crs_is_fatal=False)
    else:
        print("\n--- DSM: not provided (optional) ---")

    if block.get("dtm"):
        check_raster(resolve_path(block["dtm"]), "DTM", cfg["crs"], crs_is_fatal=False)
    else:
        print("\n--- DTM: not provided (optional) ---")

    print(f"\n=== QC {'PASSED' if all_ok else 'FAILED'} ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
