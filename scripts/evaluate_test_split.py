"""
Score a model on the held-out test split built by build_yolo_dataset.py: standard detection
metrics, plus counting accuracy — the model's palm count vs. the hand-labelled count on every
test tile, at the same confidence/NMS settings the production pipeline uses.

Counts are reported twice. "Whole tile" includes palms cut off at the tile border, which the
annotators mostly did not box but the model does detect. "Interior" counts only boxes whose
centre lies more than half the tile overlap from the border — the part of each tile a block
run keeps, since neighbouring tiles see the border palms whole and duplicates are merged.

Usage:
    python scripts/evaluate_test_split.py --weights models/oil_palm_crown-current/weights/best.pt
    python scripts/evaluate_test_split.py --weights models/oil_palm_crown-4/weights/best.pt --name baseline
    python scripts/evaluate_test_split.py --weights ... --no-val    # counts only, skip Ultralytics val
"""
import argparse
import csv

import numpy as np
from ultralytics import YOLO

from common import load_config, resolve_path, PROJECT_ROOT


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def count_matches(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> int:
    """Greedy one-to-one matching, highest-confidence prediction first."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    ious = iou_matrix(pred[np.argsort(-conf)], gt)
    used, matched = set(), 0
    for row in ious:
        for j in np.argsort(-row):
            if row[j] < thresh:
                break
            if j not in used:
                used.add(j)
                matched += 1
                break
    return matched


def centre_inside(boxes: np.ndarray, w: int, h: int, margin: float) -> int:
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    return int(((cx >= margin) & (cx < w - margin) & (cy >= margin) & (cy < h - margin)).sum())


def summarise(label: str, rows: list, manual_key: str, model_key: str):
    for block in ["all"] + sorted({r["block"] for r in rows}):
        sel = [r for r in rows if block == "all" or r["block"] == block]
        manual = np.array([r[manual_key] for r in sel])
        model_n = np.array([r[model_key] for r in sel])
        diff = model_n.sum() - manual.sum()
        print(f"  {label} {block}: {len(sel)} tiles, manual {manual.sum()} vs model {model_n.sum()} "
              f"({100 * diff / manual.sum():+.1f}%), counting accuracy "
              f"{100 * (1 - abs(diff) / manual.sum()):.1f}%, mean absolute error "
              f"{np.abs(model_n - manual).mean():.2f} palms/tile", flush=True)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default="final", help="label used in output file names")
    ap.add_argument("--dataset", default=cfg["paths"]["yolo_dataset_dir"])
    ap.add_argument("--device", default=str(cfg["model"]["device"]))
    ap.add_argument("--no-val", action="store_true", help="skip Ultralytics val, report counts only")
    args = ap.parse_args()

    dataset = resolve_path(args.dataset)
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write a data.yaml with this machine's dataset path, so a data.yaml from another machine still works.
    data_yaml = out_dir / "data_test.yaml"
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(cfg["classes"]))
    data_yaml.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\n"
                         f"test: test/images\nnames:\n{names}", encoding="utf-8")

    imgsz = cfg["model"]["imgsz"]
    conf_t, iou_t = cfg["inference"]["conf_thresh"], cfg["inference"]["iou_thresh"]
    margin = cfg["tiling"]["overlap_px"] / 2
    model = YOLO(str(resolve_path(args.weights)))

    if not args.no_val:
        m = model.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=4, device=args.device,
                      plots=False, verbose=False, project=str(out_dir), name=f"val_{args.name}",
                      exist_ok=True).results_dict
        print(f"Test split ({args.name}): precision {m['metrics/precision(B)']:.3f}, "
              f"recall {m['metrics/recall(B)']:.3f}, mAP50 {m['metrics/mAP50(B)']:.3f}, "
              f"mAP50-95 {m['metrics/mAP50-95(B)']:.3f}", flush=True)

    rows = []
    for img in sorted((dataset / "test" / "images").glob("*.jpg")):
        r = model.predict(str(img), imgsz=imgsz, conf=conf_t, iou=iou_t, device=args.device, verbose=False)[0]
        pred, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        h, w = r.orig_shape
        lab = np.loadtxt(dataset / "test" / "labels" / f"{img.stem}.txt", ndmin=2)
        if lab.size:
            gt = np.stack([(lab[:, 1] - lab[:, 3] / 2) * w, (lab[:, 2] - lab[:, 4] / 2) * h,
                           (lab[:, 1] + lab[:, 3] / 2) * w, (lab[:, 2] + lab[:, 4] / 2) * h], axis=1)
        else:
            gt = np.zeros((0, 4))
        rows.append({"tile": img.stem, "block": img.stem.rsplit("_r", 1)[0],
                     "manual_count": len(gt), "model_count": len(pred),
                     "matched": count_matches(pred, conf, gt),
                     "manual_interior": centre_inside(gt, w, h, margin),
                     "model_interior": centre_inside(pred, w, h, margin)})

    csv_path = out_dir / f"test_counts_{args.name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(r["matched"] for r in rows)
    n_pred = sum(r["model_count"] for r in rows)
    n_gt = sum(r["manual_count"] for r in rows)
    print(f"At conf {conf_t} (pipeline setting), IoU match 0.5, whole tiles: "
          f"precision {matched / max(n_pred, 1):.3f}, recall {matched / max(n_gt, 1):.3f}", flush=True)
    summarise("whole tile", rows, "manual_count", "model_count")
    summarise(f"interior ({margin:.0f} px border ignored)", rows, "manual_interior", "model_interior")
    print(f"Per-tile counts: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
