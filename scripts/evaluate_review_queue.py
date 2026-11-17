"""
Measure how well a confidence-ranked review queue finds the model's mistakes, on the labelled
test split (the same 79 held-out tiles as evaluate_test_split.py).

The production pipeline keeps detections at confidence >= 0.25. Its mistakes are false
detections (boxes with no hand label) and misses (hand labels with no box). A review queue
asks a person to check the cases the model is least sure about:

- "doubtful detections": kept boxes with confidence below --high-conf (default 0.5). If one has
  no hand label, the reviewer deletes it and fixes a false detection.
- "possible misses": boxes the pipeline threw away, with confidence between --review-conf
  (default 0.10) and 0.25. If one matches a hand label the pipeline missed, the reviewer adds
  it and fixes a miss.

Items are ranked by how close their confidence is to the 0.25 cut-off (closest first). The
script reports how many mistakes a reviewer fixes after checking the first 10%, 25%, 50% and
all of the queue, against checking the same number of kept detections picked at random, and
which mistakes no queue item points at.

Only tile interiors count (box centre more than half the tile overlap from the border), as in
the interior count: annotators mostly skipped palms cut off by the border, so edge boxes
would be scored as false detections when they are real palms.

Usage:
    python scripts/evaluate_review_queue.py --weights models/oil_palm_crown-current/weights/best.pt
    python scripts/evaluate_review_queue.py --weights ... --device cpu --dataset path/to/yolo_dataset
"""
import argparse
import csv

import numpy as np
from ultralytics import YOLO

from common import load_config, resolve_path, PROJECT_ROOT
from evaluate_test_split import iou_matrix


def greedy_match(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, thresh: float = 0.5):
    """One-to-one matching at IoU >= thresh, highest-confidence prediction first.
    Returns (pred_matched, gt_matched) boolean arrays."""
    pred_hit = np.zeros(len(pred), dtype=bool)
    gt_hit = np.zeros(len(gt), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return pred_hit, gt_hit
    ious = iou_matrix(pred, gt)
    for i in np.argsort(-conf):
        for j in np.argsort(-ious[i]):
            if ious[i, j] < thresh:
                break
            if not gt_hit[j]:
                gt_hit[j] = pred_hit[i] = True
                break
    return pred_hit, gt_hit


def interior(boxes: np.ndarray, w: int, h: int, margin: float) -> np.ndarray:
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    return (cx >= margin) & (cx < w - margin) & (cy >= margin) & (cy < h - margin)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default=cfg["paths"]["yolo_dataset_dir"])
    ap.add_argument("--device", default=str(cfg["model"]["device"]))
    ap.add_argument("--review-conf", type=float, default=cfg["inference"].get("review_conf", 0.10))
    ap.add_argument("--high-conf", type=float, default=cfg["inference"].get("review_high_conf", 0.5))
    ap.add_argument("--seed", type=int, default=0, help="for the random-review comparison")
    args = ap.parse_args()

    dataset = resolve_path(args.dataset)
    imgsz = cfg["model"]["imgsz"]
    conf_t, iou_t = cfg["inference"]["conf_thresh"], cfg["inference"]["iou_thresh"]
    margin = cfg["tiling"]["overlap_px"] / 2
    model = YOLO(str(resolve_path(args.weights)))

    items = []          # one row per queue item
    kept_interior = []  # (is_false_detection,) for every kept interior detection, for the random baseline
    n_gt = n_fp = n_fn = n_kept = 0
    fp_confident = fn_unseen = 0

    for img in sorted((dataset / "test" / "images").glob("*.jpg")):
        r = model.predict(str(img), imgsz=imgsz, conf=args.review_conf, iou=iou_t,
                          device=args.device, verbose=False)[0]
        boxes, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        h, w = r.orig_shape
        lab = np.loadtxt(dataset / "test" / "labels" / f"{img.stem}.txt", ndmin=2)
        gt = (np.stack([(lab[:, 1] - lab[:, 3] / 2) * w, (lab[:, 2] - lab[:, 4] / 2) * h,
                        (lab[:, 1] + lab[:, 3] / 2) * w, (lab[:, 2] + lab[:, 4] / 2) * h], axis=1)
              if lab.size else np.zeros((0, 4)))

        kept = conf >= conf_t
        # The pipeline's own result: kept boxes matched against the hand labels.
        kept_hit, gt_hit = greedy_match(boxes[kept], conf[kept], gt)
        # Dropped boxes (review_conf <= conf < 0.25) matched against the labels the pipeline missed.
        drop = ~kept
        missed = np.flatnonzero(~gt_hit)
        drop_hit, missed_hit = greedy_match(boxes[drop], conf[drop], gt[missed])

        in_kept = interior(boxes[kept], w, h, margin)
        in_drop = interior(boxes[drop], w, h, margin)
        in_gt = interior(gt, w, h, margin)

        n_gt += int(in_gt.sum())
        n_kept += int(in_kept.sum())
        fp = in_kept & ~kept_hit
        fn = in_gt & ~gt_hit
        n_fp += int(fp.sum())
        n_fn += int(fn.sum())
        fp_confident += int((fp & (conf[kept] >= args.high_conf)).sum())
        recoverable = np.zeros(len(gt), dtype=bool)
        recoverable[missed[missed_hit]] = True
        fn_unseen += int((fn & ~recoverable).sum())
        kept_interior += list(~kept_hit[in_kept])

        for c, hit in zip(conf[kept][in_kept], kept_hit[in_kept]):
            if c < args.high_conf:
                items.append({"tile": img.stem, "kind": "doubtful detection", "confidence": float(c),
                              "fixes_error": int(not hit)})
        for c, hit in zip(conf[drop][in_drop], drop_hit[in_drop]):
            items.append({"tile": img.stem, "kind": "possible miss", "confidence": float(c),
                          "fixes_error": int(hit)})

    items.sort(key=lambda it: abs(it["confidence"] - conf_t))
    n_err = n_fp + n_fn
    fixes = np.array([it["fixes_error"] for it in items])
    print(f"Test tiles, interiors only: {n_gt} hand-labelled palms, {n_kept} kept detections "
          f"(confidence >= {conf_t})")
    print(f"Pipeline mistakes: {n_fp} false detections + {n_fn} misses = {n_err}")
    print(f"Review queue: {len(items)} items "
          f"({sum(it['kind'] == 'doubtful detection' for it in items)} doubtful detections with "
          f"confidence < {args.high_conf}, {sum(it['kind'] == 'possible miss' for it in items)} "
          f"possible misses with confidence {args.review_conf}-{conf_t}); that is "
          f"{100 * len(items) / max(n_kept, 1):.0f} items per 100 kept detections")

    rng = np.random.default_rng(args.seed)
    kept_fp = np.array(kept_interior, dtype=bool)
    print(f"{'Checked':>22} | {'mistakes fixed':>15} | {'random kept detections, same effort':>36}")
    for share in (0.10, 0.25, 0.50, 1.00):
        k = int(round(share * len(items)))
        fixed = int(fixes[:k].sum())
        rand = np.mean([kept_fp[rng.choice(len(kept_fp), size=min(k, len(kept_fp)), replace=False)].sum()
                        for _ in range(200)]) if k else 0.0
        print(f"{f'first {share:.0%} ({k} items)':>22} | {fixed:>5} of {n_err} ({100 * fixed / max(n_err, 1):4.1f}%) | "
              f"{rand:>8.1f} ({100 * rand / max(n_err, 1):4.1f}%)")
    print(f"Not pointed at by any queue item: {fp_confident} false detections with confidence >= "
          f"{args.high_conf}, {fn_unseen} misses with no box above {args.review_conf}")

    out_dir = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "review_queue_test_items.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "tile", "kind", "confidence", "fixes_error"])
        writer.writeheader()
        for rank, it in enumerate(items, 1):
            writer.writerow({"rank": rank, **it})
    print(f"Ranked queue items: {out_csv}")


if __name__ == "__main__":
    main()
