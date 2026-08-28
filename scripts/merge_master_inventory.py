"""
Merge every processed block's centroid inventory into one master plantation-wide
GeoPackage + CSV. Run after process_block.py has been run for all blocks in
configs/blocks.yaml.

Usage:
    python scripts/merge_master_inventory.py
    python scripts/merge_master_inventory.py --blocks pilot block_002 block_003
"""
import argparse

import geopandas as gpd
import pandas as pd

from common import load_config, load_blocks, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=None,
                     help="block_ids to merge (default: all blocks in blocks.yaml)")
    args = ap.parse_args()

    cfg = load_config()
    block_ids = args.blocks or [b["block_id"] for b in load_blocks()]
    inv_dir = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "inventory"

    centroids_list, crowns_list = [], []
    missing = []
    for block_id in block_ids:
        gpkg = inv_dir / f"{block_id}_palm_inventory.gpkg"
        if not gpkg.exists():
            missing.append(block_id)
            continue
        crowns_list.append(gpd.read_file(gpkg, layer="crowns"))

        # Prefer the CHM-QC'd centroids (carries QA_Status + CHM_m) when available for
        # this block, so that flag isn't silently dropped from the merged master file.
        chm_gpkg = inv_dir / f"{block_id}_palm_inventory_chm_qc.gpkg"
        if chm_gpkg.exists():
            centroids_list.append(gpd.read_file(chm_gpkg, layer="centroids_chm"))
        else:
            centroids_list.append(gpd.read_file(gpkg, layer="centroids"))

    if missing:
        print(f"[WARN] no inventory found for: {missing} — run process_block.py for these first")
    if not centroids_list:
        raise SystemExit("No block inventories found — nothing to merge")

    all_centroids = gpd.GeoDataFrame(pd.concat(centroids_list, ignore_index=True), crs=cfg["crs"])
    all_crowns = gpd.GeoDataFrame(pd.concat(crowns_list, ignore_index=True), crs=cfg["crs"])

    out_path = inv_dir / "master_palm_inventory.gpkg"
    all_crowns.to_file(out_path, layer="crowns", driver="GPKG")
    all_centroids.to_file(out_path, layer="centroids", driver="GPKG")
    all_centroids.drop(columns="geometry").to_csv(inv_dir / "master_palm_inventory.csv", index=False)

    print(f"\nMerged {len(centroids_list)} block(s): {[b for b in block_ids if b not in missing]}")
    print(f"Total palms: {len(all_centroids)}")
    for block_id, gdf in zip([b for b in block_ids if b not in missing], centroids_list):
        print(f"  {block_id}: {len(gdf)}")
    print(f"\nWritten: {out_path}")
    print(f"CSV:     {inv_dir / 'master_palm_inventory.csv'}")


if __name__ == "__main__":
    main()
