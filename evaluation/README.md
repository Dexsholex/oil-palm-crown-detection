# Test-split evaluation

How the released model (training run 9, `models/oil_palm_crown-current`) and the earlier
baseline (run 4, `models/oil_palm_crown-4`) perform on the held-out test split: 79 tiles
(51 pilot, 28 block_003) with 716 hand-labelled crowns. Test tiles come from map sectors the
model never trained on and were not used to pick the checkpoint.

Produced with:

```bash
python scripts/evaluate_test_split.py --weights models/oil_palm_crown-current/weights/best.pt --name final
python scripts/evaluate_test_split.py --weights models/oil_palm_crown-4/weights/best.pt --name baseline
```

The labelled tiles themselves are the plantation owner's imagery and are not published.

## Detection scores (Ultralytics `val`)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Released model (run 9) | 0.845 | 0.774 | 0.844 | 0.350 |
| Baseline (run 4) | 0.727 | 0.637 | 0.698 | 0.279 |

Precision and recall here are at the confidence that maximises F1.

## Counting accuracy (pipeline settings: confidence 0.25, NMS IoU 0.45)

Palm count per tile from the model vs. the hand labels, in two ways:

- **Interior** — only boxes whose centre is more than 128 px (half the 256 px tile overlap)
  from the tile border. In a block run every palm near a border is seen whole in the
  neighbouring tile and duplicates are merged, so this is what each tile contributes.
- **Whole tile** — every box, including palms cut off by the border. The annotators mostly
  did not box those partial crowns, but the model detects them, so whole-tile counts run high.

| Model | Region | Pilot (51 tiles) | block_003 (28 tiles) | All (79 tiles) | Mean abs. error per tile |
|---|---|---|---|---|---|
| Released | Interior | 321 vs 311 (+3.2%) | 198 vs 207 (−4.3%) | **519 vs 518 (+0.2%)** | 0.57 |
| Released | Whole tile | 552 vs 438 (+26.0%) | 342 vs 278 (+23.0%) | 894 vs 716 (+24.9%) | 2.56 |
| Baseline | Interior | 311 vs 311 (0.0%) | 143 vs 207 (−30.9%) | 454 vs 518 (−12.4%) | 1.11 |
| Baseline | Whole tile | 523 vs 438 (+19.4%) | 227 vs 278 (−18.3%) | 750 vs 716 (+4.7%) | 2.68 |

Counts are model vs. hand.

What this shows:

- The released model's interior count is within 0.2% of the hand count overall and within
  about 4% on each block.
- The baseline badly undercounted block_003 (dense mature canopy and newly planted palms),
  missing about 31% of interior palms. Retraining on those cases (run 9) fixed most of it.
- The baseline's whole-tile total (+4.7%) looks good only because its pilot overcount and
  block_003 undercount cancel.
- Of the released model's 301 detections with no matching label (whole tiles, IoU 0.5), 206
  touch the tile edge. Away from the edges, roughly 100 labelled palms are missed and a similar
  number of detections have no matching label — the interior total is close partly because
  these cancel.
- On the 43 validation tiles, the released model's interior count was 304 against 317 by
  hand (−4.1%).

These are tile-level checks on held-out ground. No field count over a whole block has been
done.

## Files

- `test_counts_final.csv`, `test_counts_baseline.csv` — one row per test tile: `manual_count`
  and `model_count` (whole tile), `matched` (detections matching a label at IoU ≥ 0.5),
  `manual_interior` and `model_interior`.
