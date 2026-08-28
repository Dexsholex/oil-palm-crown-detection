"""
Turn annotated tiles (image.jpg + image.txt YOLO labels in annotations/to_label/<block>/)
into a train/val/test YOLO dataset, split by SPATIAL SECTOR (not randomly per-tile) so
train/val/test come from spatially separated parts of the plantation — see brief section 15
("do not randomly split adjacent tiles"). An image with a present-but-empty .txt is a valid
annotated background/negative tile; an image with no .txt at all is treated as un-annotated
and skipped.

Supports combining annotations from MULTIPLE blocks into one training set (e.g. pilot +
block_003) — each block's sector grid is independent, so sector strings are prefixed with
the block_id before hashing to keep the split decision per-block (sector "1_1" in pilot and
"1_1" in block_003 are unrelated physical locations and must not be forced into a correlated
split just because they share a label).

Usage:
    python scripts/build_yolo_dataset.py --blocks pilot
    python scripts/build_yolo_dataset.py --blocks pilot block_003
"""
import argparse
import hashlib
import shutil

import pandas as pd
import yaml

from common import load_config, PROJECT_ROOT


def sector_split(sector: str, train=0.70, val=0.15) -> str:
    """Deterministic hash-based split so the same sector always lands in the same split."""
    h = int(hashlib.md5(sector.encode()).hexdigest(), 16) / 16**32
    if h < train:
        return "train"
    if h < train + val:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True, nargs="+",
                     help="one or more block_ids to combine, e.g. --blocks pilot block_003")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    cfg = load_config()
    out_root = PROJECT_ROOT / cfg["paths"]["yolo_dataset_dir"]

    # Clear stale content first — rebuilding with a different set/composition of blocks
    # than a previous run must not leave old images/labels lying around in train/val/test.
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = out_root / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    n_boxes = {"train": 0, "val": 0, "test": 0}
    missing = 0
    per_block_totals = {}

    for block_id in args.blocks:
        label_dir = PROJECT_ROOT / cfg["paths"]["annotations_dir"] / "to_label" / block_id
        manifest_path = label_dir.parent / f"{block_id}_annotation_sample_manifest.csv"
        if not manifest_path.exists():
            raise SystemExit(f"Missing {manifest_path} — run sample_tiles_for_annotation.py "
                              f"for block '{block_id}' first")

        df = pd.read_csv(manifest_path)
        block_boxes = 0

        for _, row in df.iterrows():
            tile_id = row["tile_id"]
            img_src = label_dir / f"{tile_id}.jpg"
            lbl_src = label_dir / f"{tile_id}.txt"
            if not img_src.exists() or not lbl_src.exists():
                missing += 1
                continue

            prefixed_sector = f"{block_id}_{row['sector']}"
            split = sector_split(prefixed_sector, args.train_frac, args.val_frac)
            shutil.copy2(img_src, out_root / split / "images" / img_src.name)
            shutil.copy2(lbl_src, out_root / split / "labels" / lbl_src.name)
            counts[split] += 1
            with open(lbl_src) as f:
                n = sum(1 for line in f if line.strip())
            n_boxes[split] += n
            block_boxes += n

        per_block_totals[block_id] = block_boxes

    data_yaml = {
        "path": str(out_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: c for i, c in enumerate(cfg["classes"])},
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"blocks combined: {args.blocks}")
    print(f"boxes per block: {per_block_totals}")
    print(f"tiles: {counts}  (skipped {missing} un-annotated tiles — no .txt found)")
    print(f"crown boxes: {n_boxes}  total={sum(n_boxes.values())}")
    print(f"data.yaml written to {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
