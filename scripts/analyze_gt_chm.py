"""
Empirically establish this plantation's actual palm-crown height range by sampling
CHM (DSM-DTM) at the centroid of every HUMAN-ANNOTATED ground-truth crown box.

Why: a generic "typical oil palm height" number from literature isn't defensible for
THIS specific plantation (age mix, variety, terrain). Every ground-truth box was
confirmed by a human to be a real palm crown, so sampling CHM at those exact points
gives a real, data-derived height distribution to use as the plausibility band for
QC-flagging the model's own predictions later (chm_qc.py) — not a guessed threshold.

Usage:
    python scripts/analyze_gt_chm.py --block pilot
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from common import load_config, get_block, resolve_path, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)
    if not block.get("dsm") or not block.get("dtm"):
        raise SystemExit("dsm/dtm not set in blocks.yaml for this block")

    label_dir = PROJECT_ROOT / cfg["paths"]["annotations_dir"] / "to_label" / args.block
    tile_index = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["tiles_dir"] / args.block / "tile_index.csv"
    ).set_index("tile_id")

    points = []
    for txt in label_dir.glob("*.txt"):
        if txt.name == "classes.txt":
            continue
        tile_id = txt.stem
        if tile_id not in tile_index.index:
            continue
        row = tile_index.loc[tile_id]
        transform = Affine(row.transform_a, row.transform_b, row.transform_c,
                            row.transform_d, row.transform_e, row.transform_f)
        for line in open(txt):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls, cx, cy = int(parts[0]), float(parts[1]), float(parts[2])
            px, py = cx * row.width, cy * row.height
            wx, wy = transform * (px, py)
            points.append((wx, wy))

    print(f"Sampling CHM at {len(points)} human-confirmed ground-truth crown centroids...")

    dsm_path, dtm_path = resolve_path(block["dsm"]), resolve_path(block["dtm"])
    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(dtm_path) as dtm_ds:
        dsm_vals, dtm_vals, out_of_bounds = [], [], 0
        for wx, wy in points:
            try:
                dv = next(dsm_ds.sample([(wx, wy)]))[0]
                tv = next(dtm_ds.sample([(wx, wy)]))[0]
                if dv == dsm_ds.nodata or tv == dtm_ds.nodata or np.isnan(dv) or np.isnan(tv):
                    out_of_bounds += 1
                    continue
                dsm_vals.append(dv)
                dtm_vals.append(tv)
            except IndexError:
                out_of_bounds += 1

    dsm_vals, dtm_vals = np.array(dsm_vals), np.array(dtm_vals)
    chm = dsm_vals - dtm_vals

    print(f"\nValid samples: {len(chm)} (excluded {out_of_bounds} outside DSM/DTM coverage or nodata)")
    print(f"\n=== Empirical ground-truth palm CHM (height-above-ground) distribution ===")
    print(f"  min:    {chm.min():.2f} m")
    print(f"  p5:     {np.percentile(chm, 5):.2f} m")
    print(f"  p25:    {np.percentile(chm, 25):.2f} m")
    print(f"  median: {np.median(chm):.2f} m")
    print(f"  mean:   {chm.mean():.2f} m")
    print(f"  p75:    {np.percentile(chm, 75):.2f} m")
    print(f"  p95:    {np.percentile(chm, 95):.2f} m")
    print(f"  max:    {chm.max():.2f} m")
    print(f"  std:    {chm.std():.2f} m")

    negative = int((chm < 0).sum())
    if negative:
        print(f"\n[NOTE] {negative} points have negative CHM (DSM<DTM) — likely DSM/DTM misalignment "
              f"noise at that pixel, not a real measurement; harmless for setting a threshold band.")

    p5 = np.percentile(chm, 5)
    print(f"\nSuggested chm_qc.py --min-height-m: {max(0.5, p5 * 0.7):.2f} "
          f"(70% of the p5 empirical value, as a conservative floor — flags things clearly "
          f"shorter than almost every confirmed real palm, without being so strict it flags "
          f"legitimate young/small palms)")


if __name__ == "__main__":
    main()
