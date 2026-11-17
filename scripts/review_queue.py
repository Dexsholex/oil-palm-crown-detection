"""
Build a review queue for one block: the detections the model is least sure about, and the
boxes it nearly kept, ranked so a person checks the most uncertain cases first.

Two kinds of item:
- "doubtful detection": a counted crown with confidence below inference.review_high_conf
  (0.5). The reviewer decides whether it is a palm; if not, it comes out of the count.
- "possible miss": a box from the review_candidates layer (confidence between
  inference.review_conf, 0.10, and the counting threshold, 0.25). If it is a palm, it goes
  into the count.

Items are ranked by how close their confidence is to the counting threshold, closest first.
scripts/evaluate_review_queue.py measures, on the labelled test tiles, how many of the
model's mistakes a reviewer fixes by working down this ranking.

Outputs, in outputs/review/<block>/ (client imagery: keep private):
- <block>_review_queue.gpkg, layer review_queue: one box per item with Review_Rank, Kind,
  Ref_ID (Tree_ID or Cand_ID), Confidence, Easting, Northing and an empty Decision column.
  Open it in QGIS over the orthomosaic and write "palm" or "not palm" in Decision.
- <block>_review_queue.csv: the same table without geometry.
- chips/: one picture per item (the item in orange or cyan, counted crowns around it in
  green), named by rank, and sheets/: the chips 48 to a page for a quick first pass.

Needs the block's inventory from infer_pipeline.py (a version that writes the
review_candidates layer; without it only doubtful detections are queued).

Usage:
    python scripts/review_queue.py --block block_003
    python scripts/review_queue.py --block block_003 --max-items 500
    python scripts/review_queue.py --inventory path/to/x_palm_inventory.gpkg --imagery path/to/rgb.tif --block x
"""
import argparse
import math

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from common import load_config, get_block, resolve_path, PROJECT_ROOT

COLOURS = {"doubtful detection": (0, 140, 255), "possible miss": (255, 200, 0)}  # BGR: orange, cyan


def build_queue(crowns: gpd.GeoDataFrame, candidates: gpd.GeoDataFrame, conf_t: float, high_conf: float):
    doubtful = crowns[crowns["Confidence"] < high_conf]
    parts = [gpd.GeoDataFrame({"Kind": "doubtful detection", "Ref_ID": doubtful["Tree_ID"],
                               "Confidence": doubtful["Confidence"]}, geometry=doubtful.geometry)]
    if candidates is not None and len(candidates):
        parts.append(gpd.GeoDataFrame({"Kind": "possible miss", "Ref_ID": candidates["Cand_ID"],
                                       "Confidence": candidates["Confidence"]}, geometry=candidates.geometry))
    queue = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crowns.crs)
    queue["_uncertainty"] = (queue["Confidence"] - conf_t).abs()
    queue = queue.sort_values(["_uncertainty", "Ref_ID"]).drop(columns="_uncertainty").reset_index(drop=True)
    queue.insert(0, "Review_Rank", np.arange(1, len(queue) + 1))
    centroids = queue.geometry.centroid
    queue["Easting"] = centroids.x
    queue["Northing"] = centroids.y
    queue["Decision"] = ""
    queue["Note"] = ""
    return queue


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    lo, hi = np.percentile(rgb[rgb > 0], (2, 98)) if (rgb > 0).any() else (0, 1)
    return np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def make_chips(queue, crowns, imagery_path, out_dir, chip_m: float, px: int = 320):
    chips_dir = out_dir / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with rasterio.open(imagery_path) as ds:
        q = queue.to_crs(ds.crs) if queue.crs != ds.crs else queue
        c = crowns.to_crs(ds.crs) if crowns.crs != ds.crs else crowns
        c_sindex = c.sindex
        for _, item in q.iterrows():
            cx, cy = item.geometry.centroid.x, item.geometry.centroid.y
            half = chip_m / 2
            bounds = (cx - half, cy - half, cx + half, cy + half)
            win = from_bounds(*bounds, transform=ds.transform)
            data = ds.read([1, 2, 3], window=win, out_shape=(3, px, px), boundless=True, fill_value=0)
            img = cv2.cvtColor(np.ascontiguousarray(to_uint8(np.moveaxis(data, 0, -1))), cv2.COLOR_RGB2BGR)
            scale = px / chip_m

            def to_px(x, y):
                return int(round((x - bounds[0]) * scale)), int(round((bounds[3] - y) * scale))

            for j in c_sindex.query(item.geometry.buffer(half * 1.5), predicate="intersects"):
                g = c.geometry.iloc[j]
                if item["Kind"] == "doubtful detection" and c["Tree_ID"].iloc[j] == item["Ref_ID"]:
                    continue
                x1, y1, x2, y2 = g.bounds
                cv2.rectangle(img, to_px(x1, y2), to_px(x2, y1), (60, 200, 60), 2)
            x1, y1, x2, y2 = item.geometry.bounds
            cv2.rectangle(img, to_px(x1, y2), to_px(x2, y1), COLOURS[item["Kind"]], 2)
            label = f"#{item['Review_Rank']} {item['Kind']} {item['Confidence']:.2f}"
            cv2.rectangle(img, (0, 0), (px, 20), (0, 0, 0), -1)
            cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            name = f"{item['Review_Rank']:05d}_{item['Kind'].replace(' ', '_')}.png"
            cv2.imwrite(str(chips_dir / name), img)
            names.append(f"chips/{name}")
    return names


def make_sheets(out_dir, chip_names, per_sheet: int = 48, cols: int = 8):
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    n_sheets = math.ceil(len(chip_names) / per_sheet)
    for s in range(n_sheets):
        imgs = [cv2.imread(str(out_dir / n)) for n in chip_names[s * per_sheet:(s + 1) * per_sheet]]
        h, w = imgs[0].shape[:2]
        rows = math.ceil(len(imgs) / cols)
        sheet = np.full((rows * (h + 4), cols * (w + 4), 3), 255, np.uint8)
        for k, im in enumerate(imgs):
            r, col = divmod(k, cols)
            sheet[r * (h + 4):r * (h + 4) + h, col * (w + 4):col * (w + 4) + w] = im
        cv2.imwrite(str(sheets_dir / f"sheet_{s + 1:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return n_sheets


def main():
    cfg = load_config()
    inf = cfg["inference"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--inventory", default=None, help="default outputs/inventory/<block>_palm_inventory.gpkg")
    ap.add_argument("--imagery", default=None, help="orthomosaic for the chips; default the block's rgb in blocks.yaml")
    ap.add_argument("--high-conf", type=float, default=inf.get("review_high_conf", 0.5))
    ap.add_argument("--max-items", type=int, default=None, help="keep only the first N items")
    ap.add_argument("--chip-m", type=float, default=8.0, help="chip width on the ground, metres")
    ap.add_argument("--no-chips", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_root = PROJECT_ROOT / cfg["paths"]["outputs_dir"]
    inv_path = resolve_path(args.inventory) if args.inventory else out_root / "inventory" / f"{args.block}_palm_inventory.gpkg"
    out_dir = resolve_path(args.out_dir) if args.out_dir else out_root / "review" / args.block
    out_dir.mkdir(parents=True, exist_ok=True)

    crowns = gpd.read_file(inv_path, layer="crowns")
    if "review_candidates" in set(gpd.list_layers(inv_path)["name"]):
        candidates = gpd.read_file(inv_path, layer="review_candidates")
    else:
        candidates = None
        print("No review_candidates layer in this inventory: re-run infer_pipeline.py to add possible "
              "misses. Queuing doubtful detections only.", flush=True)

    queue = build_queue(crowns, candidates, inf["conf_thresh"], args.high_conf)
    if args.max_items:
        queue = queue.head(args.max_items)

    if not args.no_chips and len(queue):
        imagery = resolve_path(args.imagery) if args.imagery else resolve_path(get_block(args.block)["rgb"])
        queue["Chip"] = make_chips(queue, crowns, imagery, out_dir, args.chip_m)
        n_sheets = make_sheets(out_dir, list(queue["Chip"]))
    else:
        queue["Chip"] = ""
        n_sheets = 0

    gpkg_path = out_dir / f"{args.block}_review_queue.gpkg"
    queue.to_file(gpkg_path, layer="review_queue", driver="GPKG")
    queue.drop(columns="geometry").to_csv(out_dir / f"{args.block}_review_queue.csv", index=False)

    n_doubt = int((queue["Kind"] == "doubtful detection").sum())
    print(f"Block '{args.block}': {len(crowns)} counted crowns. Review queue: {len(queue)} items "
          f"({n_doubt} doubtful detections, {len(queue) - n_doubt} possible misses), "
          f"{100 * len(queue) / max(len(crowns), 1):.1f} per 100 counted crowns.", flush=True)
    print(f"Queue: {gpkg_path} (layer review_queue; fill Decision with 'palm' or 'not palm')", flush=True)
    if n_sheets:
        print(f"Pictures: {out_dir / 'chips'} and {n_sheets} contact sheet(s) in {out_dir / 'sheets'}", flush=True)


if __name__ == "__main__":
    main()
