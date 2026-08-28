"""
Select a representative/stratified subset of tiles for manual annotation,
instead of annotating the whole block. Stratifies on two cheap RGB-only proxies:

  - vegetation "greenness" (Excess Green Index)  -> spans bare soil/road <-> dense canopy
  - texture / edge density (Laplacian variance)   -> spans smooth weed mats <-> structured
                                                       palm crowns / crowded canopy

plus a spatial-sector bucket (coarse grid over the block) so the sample isn't
clustered in one corner. This deliberately does NOT try to pre-classify
"palm vs weed" — it just makes sure annotators see a spread of conditions
(Scenario A/B/C from the project brief), per section 9/14 of the brief.

Usage:
    python scripts/sample_tiles_for_annotation.py --block pilot --n-tiles 150
"""
import argparse
import shutil
import csv

import cv2
import numpy as np
import pandas as pd

from common import load_config, PROJECT_ROOT


def excess_green(img_bgr):
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    total = r + g + b + 1e-6
    rn, gn, bn = r / total, g / total, b / total
    exg = 2 * gn - rn - bn
    return float(np.mean(exg))


def texture_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--n-tiles", type=int, default=150,
                     help="target number of tiles to sample for annotation")
    ap.add_argument("--sectors", type=int, default=4,
                     help="grid sectors per axis for spatial spread (default 4x4=16 sectors)")
    ap.add_argument("--bins", type=int, default=3, help="quantile bins per proxy metric")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config()
    tiles_dir = PROJECT_ROOT / cfg["paths"]["tiles_dir"] / args.block
    index_path = tiles_dir / "tile_index.csv"
    if not index_path.exists():
        raise SystemExit(f"No tile_index.csv at {index_path} — run tile_orthomosaic.py first")

    df = pd.read_csv(index_path)

    out_dir_check = PROJECT_ROOT / cfg["paths"]["annotations_dir"] / "to_label" / args.block
    already_sampled = {p.stem for p in out_dir_check.glob("*.jpg")} if out_dir_check.exists() else set()
    if already_sampled:
        before = len(df)
        df = df[~df["tile_id"].isin(already_sampled)]
        print(f"Excluding {before - len(df)} tiles already sampled in a previous batch "
              f"({len(already_sampled)} found in {out_dir_check})")

    print(f"Scoring {len(df)} tiles (greenness + texture)...", flush=True)

    exg_vals, tex_vals = [], []
    for _, row in df.iterrows():
        img = cv2.imread(str(PROJECT_ROOT / row["image_path"]))
        exg_vals.append(excess_green(img))
        tex_vals.append(texture_score(img))
    df["exg"] = exg_vals
    df["texture"] = tex_vals

    df["exg_bin"] = pd.qcut(df["exg"], args.bins, labels=False, duplicates="drop")
    df["tex_bin"] = pd.qcut(df["texture"], args.bins, labels=False, duplicates="drop")

    col_bin = pd.cut(df["col_off"], args.sectors, labels=False)
    row_bin = pd.cut(df["row_off"], args.sectors, labels=False)
    df["sector"] = row_bin.astype(str) + "_" + col_bin.astype(str)

    df["stratum"] = (
        df["exg_bin"].astype(str) + "_" + df["tex_bin"].astype(str) + "_" + df["sector"]
    )

    strata = df["stratum"].unique()
    per_stratum = max(1, args.n_tiles // len(strata))
    print(f"{len(strata)} strata (greenness x texture x spatial sector), "
          f"~{per_stratum} tiles/stratum target")

    # Explicit loop rather than groupby(...).apply(...): on some pandas versions,
    # group_keys=False silently drops the grouping column ('stratum') from the result
    # when a group's sample returns exactly 1 row (as happens whenever strata > n_tiles),
    # which broke the stratum-count print below with a KeyError.
    sampled_parts = []
    for _, g in df.groupby("stratum"):
        sampled_parts.append(g.sample(min(len(g), per_stratum), random_state=args.seed))
    sampled = pd.concat(sampled_parts)
    if len(sampled) < args.n_tiles:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(min(len(remaining), args.n_tiles - len(sampled)),
                                  random_state=args.seed)
        sampled = pd.concat([sampled, extra])

    out_dir = PROJECT_ROOT / cfg["paths"]["annotations_dir"] / "to_label" / args.block
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in sampled.iterrows():
        src = PROJECT_ROOT / row["image_path"]
        shutil.copy2(src, out_dir / src.name)

    manifest_path = out_dir.parent / f"{args.block}_annotation_sample_manifest.csv"
    if manifest_path.exists():
        prior = pd.read_csv(manifest_path)
        combined = pd.concat([prior, sampled], ignore_index=True).drop_duplicates(subset="tile_id")
        combined.to_csv(manifest_path, index=False)
        print(f"Appended to existing manifest ({len(prior)} prior + {len(sampled)} new = {len(combined)} total)", flush=True)
    else:
        sampled.to_csv(manifest_path, index=False)

    print(f"\nSampled {len(sampled)} tiles across {sampled['stratum'].nunique()} strata", flush=True)
    print(f"Copied to: {out_dir}", flush=True)
    print(f"Manifest:  {manifest_path}", flush=True)
    print("\nNext: open this folder in an annotation tool (e.g. labelImg) with class list "
          "'oil_palm' and YOLO output format, save .txt labels next to each .jpg.")


if __name__ == "__main__":
    main()
