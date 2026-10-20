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

## Outside imagery: an independent dataset from Peru

To see how the model does away from the plantation it was trained on, it was run on
*Oil Palm Tree Detection for Anomaly Identification* (Dominguez Meza & Rituay, Mendeley Data,
[doi:10.17632/nh7d23dgnw.1](https://doi.org/10.17632/nh7d23dgnw.1), CC BY 4.0):

- 424 straight-down drone photos (800 × 600 px) from a DJI Phantom 4 Multispectral flown at
  17–30 m, by researchers at Peruvian universities. None were used in training.
- 549 labelled palms (the dataset's healthy and anomalous classes counted together).
- Very different from the training tiles: close-ups of **mature palms with overlapping crowns**
  that fill the frame (median box 491 px wide, against roughly 150–200 px in the training
  tiles). Labels cover the full frond spread of the main palms; palms cut off at the photo
  edge are mostly unlabelled.

Scored at confidence 0.25, at several input sizes (a smaller input shrinks the palms towards the
size the model was trained on):

| Input size | Palm width as seen by the model | mAP@0.5 (strict box match) | Labelled palms found (detection centre inside the label) | Detections centred on a labelled palm | Photos with no detection |
|---|---|---|---|---|---|
| 800 px (native) | ~490 px | 0.002 | — | — | — |
| 640 px | ~390 px | 0.005 | 33% | 47% | 184 of 424 |
| 256 px | ~160 px | 0.049 | 53% | 71% | 115 of 424 |

Strict mAP at 480 px and 352 px was 0.004 and 0.009.

What this shows:

- **The model does not transfer to close-up imagery of mature, overlapping palms.** Even with
  the photos shrunk so palms match the training size, it finds only about half of them,
  against 77–83% on the held-out tiles from its own plantation.
- Palm size on screen matters a lot: shrinking the input from 640 to 256 px raised the share
  of palms found from 33% to 53%.
- The strict scores understate it somewhat: where the model does find a palm, its box often
  covers the crown core rather than the full frond spread, and in the photos checked by eye,
  several of its "false" detections were real palms at the photo edge that the dataset did not
  label.
- To use the model on imagery like this, fine-tune it on a few hundred labelled examples of
  that imagery first. Plantations imaged the same way as the training data (orthomosaics at
  about 2 cm/px, young palms) are still untested outside the original site.

## Files

- `test_counts_final.csv`, `test_counts_baseline.csv` — one row per test tile: `manual_count`
  and `model_count` (whole tile), `matched` (detections matching a label at IoU ≥ 0.5),
  `manual_interior` and `model_interior`.
