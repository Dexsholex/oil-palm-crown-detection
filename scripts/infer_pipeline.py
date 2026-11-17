"""
Run the trained YOLO model over every tile of a block, convert pixel boxes back
to the project CRS using each tile's affine transform, merge/de-duplicate
detections created by tile overlap (geospatial NMS), and produce the final
geospatial palm inventory (crown polygons + centroids).

Requires tile_orthomosaic.py to have been run for the block first (for
tile_index.csv — the pixel->world affine transform per tile).

Boxes the model finds with confidence between inference.review_conf (0.10) and the counting
threshold conf_thresh (0.25) are not counted. They are saved in a separate layer,
review_candidates, as possible misses for scripts/review_queue.py. The count and the crowns
and centroids layers are exactly what a run at conf_thresh alone would give.

Usage:
    python scripts/infer_pipeline.py --block pilot --weights models/oil_palm_crown/weights/best.pt
"""
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from shapely.geometry import box as shp_box
from ultralytics import YOLO

from common import load_config, get_block, resolve_path, PROJECT_ROOT


def geo_nms(gdf: gpd.GeoDataFrame, iou_thresh: float, centroid_dist_thresh: float = 3.5) -> gpd.GeoDataFrame:
    """Greedy NMS in geo space to drop duplicate detections from overlapping tiles.

    Merges on IoU>=iou_thresh OR centroid distance < centroid_dist_thresh (metres).
    The distance rule catches cross-tile duplicates of the SAME physical crown whose
    boxes differ enough (partial visibility near a tile edge) that IoU alone misses
    them.

    IMPORTANT: candidates are found via a DISTANCE buffer on the CENTROID, not box
    intersection. An earlier version used `sindex.query(geom, predicate="intersects")`
    on the box geometry to find candidates, which silently caps how far apart two
    merge-candidates can ever be (bounded by their combined box half-extents) — a
    partially-cropped edge box in one tile and a full box in the neighbouring tile
    for the SAME tree can easily NOT overlap at all, so the distance rule never got
    a chance to fire for exactly the cases it exists to catch.

    A later version buffered the BOX geometry by centroid_dist_thresh before querying
    — correct, but querying a box-based index with an inflated box in a dense planting
    (many crowns within a few metres of each other) returned far more candidates than
    needed and was too slow at production scale (tens of thousands of detections).
    Building the spatial index on CENTROID POINTS instead, and buffering only a small
    point radius, keeps the search region tight (a circle of radius centroid_dist_thresh)
    while still being a safe superset of any real IoU-overlap candidates — two boxes of
    this size can't have meaningful IoU without their centroids being well within that
    radius anyway.
    """
    gdf = gdf.sort_values("confidence", ascending=False).reset_index(drop=True)
    geoms = gdf.geometry.values
    centroid_series = gdf.geometry.centroid
    centroids = centroid_series.values
    centroid_sindex = centroid_series.sindex
    keep = np.ones(len(gdf), dtype=bool)

    for i in range(len(gdf)):
        if not keep[i]:
            continue
        candidate_idx = centroid_sindex.query(centroids[i].buffer(centroid_dist_thresh), predicate="intersects")
        for j in candidate_idx:
            if j <= i or not keep[j]:
                continue
            is_dup = centroids[i].distance(centroids[j]) < centroid_dist_thresh
            if not is_dup:
                inter = geoms[i].intersection(geoms[j]).area
                if inter <= 0:
                    continue
                union = geoms[i].area + geoms[j].area - inter
                is_dup = union > 0 and (inter / union) >= iou_thresh
            if is_dup:
                # lower-confidence box (j, since sorted desc) is suppressed
                keep[j] = False
    return gdf[keep].reset_index(drop=True)


def split_and_dedup(gdf: gpd.GeoDataFrame, conf: float, merge_iou: float):
    """Split raw detections at the counting threshold and remove duplicates.

    Returns (crowns, candidates). crowns: detections with confidence >= conf after geo_nms,
    exactly as if inference had run at conf. candidates: lower-confidence detections that do
    not duplicate a crown or each other (possible misses for review; never counted).

    Re-running geo_nms over crowns + candidates removes no crown: every crown has a higher
    confidence than every candidate, and greedy NMS already left no duplicate pair among the
    crowns, so only candidates can be suppressed.
    """
    crowns = geo_nms(gdf[gdf["confidence"] >= conf], merge_iou)
    low = gdf[gdf["confidence"] < conf]
    if len(low) == 0:
        return crowns, low.reset_index(drop=True)
    both = pd.concat([crowns.assign(_cand=False), low.assign(_cand=True)], ignore_index=True)
    both = geo_nms(gpd.GeoDataFrame(both, crs=gdf.crs), merge_iou)
    assert int((~both["_cand"]).sum()) == len(crowns), "a candidate suppressed a counted crown"
    candidates = both[both["_cand"]].drop(columns="_cand").reset_index(drop=True)
    return crowns, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--merge-iou", type=float, default=None)
    ap.add_argument("--review-conf", type=float, default=None,
                    help="also keep boxes down to this confidence as possible misses for review "
                         "(not counted); default inference.review_conf in the config")
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)
    inf_cfg = cfg["inference"]
    conf = args.conf or inf_cfg["conf_thresh"]
    iou = args.iou or inf_cfg["iou_thresh"]
    merge_iou = args.merge_iou or inf_cfg["merge_iou_thresh"]
    review_conf = min(args.review_conf or inf_cfg.get("review_conf", conf), conf)

    tiles_dir = PROJECT_ROOT / cfg["paths"]["tiles_dir"] / args.block
    index_path = tiles_dir / "tile_index.csv"
    if not index_path.exists():
        raise SystemExit(f"Missing {index_path} — run tile_orthomosaic.py first")
    tile_index = pd.read_csv(index_path).set_index("tile_id")

    model = YOLO(args.weights)
    image_dir = tiles_dir / "images"
    print(f"Running inference on tiles in {image_dir} ...", flush=True)

    records = []
    results = model.predict(
        source=str(image_dir),
        conf=review_conf,
        iou=iou,
        imgsz=cfg["model"]["imgsz"],
        device=cfg["model"]["device"],
        stream=True,
        verbose=False,
    )

    n_tiles_done = 0
    for result in results:
        tile_id = result.path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if tile_id not in tile_index.index:
            continue
        row = tile_index.loc[tile_id]
        transform = Affine(row.transform_a, row.transform_b, row.transform_c,
                            row.transform_d, row.transform_e, row.transform_f)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            n_tiles_done += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            wx1, wy1 = transform * (x1, y1)
            wx2, wy2 = transform * (x2, y2)
            minx, maxx = sorted((wx1, wx2))
            miny, maxy = sorted((wy1, wy2))
            records.append({
                "tile_id": tile_id,
                "confidence": float(c),
                "geometry": shp_box(minx, miny, maxx, maxy),
            })
        n_tiles_done += 1
        if n_tiles_done % 200 == 0:
            print(f"  ... {n_tiles_done} tiles inferred, {len(records)} raw detections so far", flush=True)

    n_counted = sum(r["confidence"] >= conf for r in records)
    print(f"Inference done: {n_tiles_done} tiles, {n_counted} raw (pre-dedup) detections at confidence "
          f">= {conf}, plus {len(records) - n_counted} between {review_conf} and {conf} for review", flush=True)
    if not n_counted:
        print("No detections found.", flush=True)
        return

    gdf = gpd.GeoDataFrame(records, crs=cfg["crs"])
    deduped, candidates = split_and_dedup(gdf, conf, merge_iou)
    print(f"After cross-tile duplicate removal (IoU >= {merge_iou}): {len(deduped)} crowns "
          f"(+ {len(candidates)} possible misses for review, not counted)", flush=True)

    out_dir = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path, csv_path = write_inventory(deduped, candidates, args.block, out_dir)

    print(f"\n=== Block '{args.block}' total palm count: {len(deduped)} ===", flush=True)
    print(f"Geospatial inventory: {gpkg_path}  (layers: crowns, centroids, review_candidates)", flush=True)
    print(f"CSV table: {csv_path}", flush=True)


def write_inventory(deduped: gpd.GeoDataFrame, candidates: gpd.GeoDataFrame, block_id: str, out_dir):
    """Write the crowns and centroids layers (the inventory), the review_candidates layer, and the CSV."""
    deduped = deduped.copy()
    deduped["Tree_ID"] = [f"{block_id}_{i+1:06d}" for i in range(len(deduped))]
    deduped["Crown_ID"] = deduped["Tree_ID"]
    deduped["Block_ID"] = block_id
    deduped["Confidence"] = deduped["confidence"]
    deduped["Crown_Area_m2"] = deduped.geometry.area
    centroids = deduped.geometry.centroid
    deduped["Easting"] = centroids.x
    deduped["Northing"] = centroids.y
    deduped["QA_Status"] = ""

    crowns_out = deduped[["Tree_ID", "Crown_ID", "Block_ID", "Confidence",
                           "Crown_Area_m2", "Easting", "Northing", "QA_Status", "geometry"]]
    points_out = crowns_out.copy()
    points_out["geometry"] = centroids

    gpkg_path = out_dir / f"{block_id}_palm_inventory.gpkg"
    crowns_out.to_file(gpkg_path, layer="crowns", driver="GPKG")
    points_out.to_file(gpkg_path, layer="centroids", driver="GPKG")

    if len(candidates):
        cand = candidates.copy()
        cand["Cand_ID"] = [f"{block_id}_c{i+1:06d}" for i in range(len(cand))]
        cand["Block_ID"] = block_id
        cand["Confidence"] = cand["confidence"]
        cand_centroids = cand.geometry.centroid
        cand["Easting"] = cand_centroids.x
        cand["Northing"] = cand_centroids.y
        cand[["Cand_ID", "Block_ID", "Confidence", "Easting", "Northing", "geometry"]].to_file(
            gpkg_path, layer="review_candidates", driver="GPKG")

    csv_path = out_dir / f"{block_id}_palm_inventory.csv"
    crowns_out.drop(columns="geometry").to_csv(csv_path, index=False)
    return gpkg_path, csv_path


if __name__ == "__main__":
    main()
