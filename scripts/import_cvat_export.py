"""
Import a CVAT "Ultralytics YOLO" export (zip) back into annotations/to_label/<block>/,
matching CVAT's label .txt files to our existing tile .jpg files by filename. Only the
.txt files are taken from the export — the .jpg tiles already on disk (the ones we
uploaded to CVAT) are left untouched, so this is safe to re-run as annotation continues.

Usage:
    python scripts/import_cvat_export.py --block pilot --zip path\to\task_export.zip
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from common import load_config, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()

    cfg = load_config()
    label_dir = PROJECT_ROOT / cfg["paths"]["annotations_dir"] / "to_label" / args.block
    if not label_dir.exists():
        raise SystemExit(f"{label_dir} doesn't exist — run sample_tiles_for_annotation.py first")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(tmp_path)

        txt_files = list(tmp_path.rglob("labels/**/*.txt"))
        if not txt_files:
            # fall back: some exports put labels directly under labels/ with no subset dir
            txt_files = [p for p in tmp_path.rglob("*.txt") if p.parent.name != "" and "label" in str(p.parent).lower()]
        if not txt_files:
            raise SystemExit("No label .txt files found in the export zip — check the export format was "
                              "'Ultralytics YOLO' (Detection) and the task actually has annotations.")

        data_yaml = next(tmp_path.rglob("data.yaml"), None)
        if data_yaml:
            print(f"Export class list ({data_yaml}):")
            print(data_yaml.read_text())

        matched, unmatched = 0, []
        for txt in txt_files:
            tile_id = txt.stem
            dst_img = label_dir / f"{tile_id}.jpg"
            if not dst_img.exists():
                unmatched.append(tile_id)
                continue
            shutil.copy2(txt, label_dir / f"{tile_id}.txt")
            matched += 1

        print(f"\nMatched {matched} label files to existing tiles in {label_dir}")
        if unmatched:
            print(f"[WARN] {len(unmatched)} label files had no matching .jpg tile — "
                  f"filenames may have been altered by CVAT export. Examples: {unmatched[:5]}")

        total_boxes = sum(
            1 for txt in label_dir.glob("*.txt") if txt.name != "classes.txt"
            for line in open(txt) if line.strip()
        )
        n_annotated_tiles = sum(
            1 for txt in label_dir.glob("*.txt") if txt.name != "classes.txt"
        )
        print(f"Total annotated tiles so far: {n_annotated_tiles}")
        print(f"Total crown boxes so far: {total_boxes}")


if __name__ == "__main__":
    main()
