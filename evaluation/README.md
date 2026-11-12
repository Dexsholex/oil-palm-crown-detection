# Test-split evaluation

How three models perform on the held-out West African test split: 79 tiles (51 pilot,
28 block_003) with 716 hand-labelled crowns. Test tiles come from map sectors the models never
trained on and were not used to pick a checkpoint.

- **Released model** (training run 9, `models/oil_palm_crown-current`), used for the production
  count.
- **Baseline** (run 4, `models/oil_palm_crown-4`), the earlier model.
- **Trained on both sites** (`models/oil_palm_crown-joint`, added in v1.2.0): run 9 retrained on
  the West African tiles together with public drone photos from Peru. See
  [Fine-tuning on Peru photos](#fine-tuning-on-peru-photos) below.

Produced with:

```bash
python scripts/evaluate_test_split.py --weights models/oil_palm_crown-current/weights/best.pt --name final
python scripts/evaluate_test_split.py --weights models/oil_palm_crown-4/weights/best.pt --name baseline
python scripts/evaluate_test_split.py --weights models/oil_palm_crown-joint/weights/best.pt --name joint
```

The labelled tiles themselves are the plantation owner's imagery and are not published.

## Detection scores (Ultralytics `val`)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Released model (run 9) | 0.845 | 0.774 | 0.844 | 0.350 |
| Baseline (run 4) | 0.727 | 0.637 | 0.698 | 0.279 |
| Trained on both sites | 0.817 | 0.785 | 0.852 | 0.382 |

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
| Both sites | Interior | 315 vs 311 (+1.3%) | 207 vs 207 (0.0%) | **522 vs 518 (+0.8%)** | 0.43 |
| Both sites | Whole tile | 540 vs 438 (+23.3%) | 358 vs 278 (+28.8%) | 898 vs 716 (+25.4%) | 2.46 |

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
- The model trained on both sites does as well as the released model here. Its interior
  count is 522 against 518 (+0.8%) and its error per tile is a little lower (0.43 against
  0.57 palms), but on 79 tiles these differences are too small to call either model better.

These are tile-level checks on held-out ground. No field count over a whole block has been
done.

## Outside imagery: an independent dataset from Peru

To see how the model does away from the plantation it was trained on, it was run on
*Oil Palm Tree Detection for Anomaly Identification* (Dominguez Meza & Rituay, Mendeley Data,
[doi:10.17632/nh7d23dgnw.1](https://doi.org/10.17632/nh7d23dgnw.1), CC BY 4.0):

- 424 straight-down drone photos (800 × 600 px) from a DJI Phantom 4 Multispectral flown at
  17–30 m, by researchers at Peruvian universities.
- 549 labelled palms (the dataset's healthy and anomalous classes counted together).
- Very different from the training tiles: close-ups of **mature palms with overlapping crowns**
  that fill the frame (median box 491 px wide, against roughly 150–200 px in the training
  tiles). Labels cover the full frond spread of the main palms; palms cut off at the photo
  edge are mostly unlabelled.

### The released model, without fine-tuning

Scored on all 424 photos at confidence 0.25, at several input sizes (a smaller input shrinks
the palms towards the size the model was trained on):

| Input size | Palm width as seen by the model | mAP@0.5 (strict box match) | Labelled palms found (detection centre inside the label) | Detections centred on a labelled palm | Photos with no detection |
|---|---|---|---|---|---|
| 800 px (native) | ~490 px | 0.002 | — | — | — |
| 640 px | ~390 px | 0.005 | 33% | 47% | 184 of 424 |
| 256 px | ~160 px | 0.049 | 53% | 71% | 115 of 424 |

Strict mAP at 480 px and 352 px was 0.004 and 0.009.

What this shows:

- **The released model does not transfer to close-up imagery of mature, overlapping palms.**
  Even with the photos shrunk so palms match the training size, it finds only about half of
  them, against 77–83% on the held-out tiles from its own plantation.
- Palm size on screen matters a lot: shrinking the input from 640 to 256 px raised the share
  of palms found from 33% to 53%.
- The strict scores understate it somewhat: where the model does find a palm, its box often
  covers the crown core rather than the full frond spread, and in the photos checked by eye,
  several of its "false" detections were real palms at the photo edge that the dataset did not
  label.

### Fine-tuning on Peru photos

The photos were split into training (276 photos, 354 palms), validation (64, 80) and test
(84, 115). The dataset's own split is random, but neighbouring photo numbers can show the same
palm (photos 200 and 201, for example), so whole blocks of 20 consecutive numbers went to one
split, and any validation or test photo that looked almost the same as a training photo was
moved to training. The split is in `peru_split.csv`.

Two models were trained from the released model, each with two learning rates, keeping the one
with the best validation score:

- **Fine-tuned on Peru only** (`notebooks/finetune_peru.ipynb`, public data only): Peru photos
  at 320 or 640 px input; kept 320 px, learning rate 0.0002.
- **Trained on both sites** (`notebooks/finetune_joint.ipynb`): the 313 West African training
  tiles together with the Peru training photos. Each Peru photo was shrunk to 320 × 240 px so its
  palms appear about 196 px wide, close to the West African crowns, and 16 photos were packed
  into one 1280 × 960 picture (six different shuffles, 108 pictures). Kept learning rate 0.001,
  chosen by the mean of West African and Peru validation mAP@0.5 (0.935 against 0.913). Training
  stopped at epoch 92 (best at 62), 35 minutes on one Kaggle T4 GPU.

Both used run 9's recipe (AdamW, rotations up to 180°, flips, mosaic, seed 0). The released
model was scored at 256 px, its best size on the validation photos; the other two at 320 px.

**Peru test photos** (84 photos, 115 labelled palms; confidence 0.25 for the counts):

| Model | Precision | Recall | mAP@0.5 | Palms found | Count vs. 115 by hand | Photos with no detection |
|---|---|---|---|---|---|---|
| Released (run 9) | 0.142 | 0.200 | 0.061 | 47.0% | 80 (−30.4%) | 29 |
| Fine-tuned on Peru only | 0.933 | 0.957 | 0.985 | 97.4% | 159 (+38.3%) | 0 |
| Trained on both sites | 0.952 | 0.965 | 0.978 | 98.3% | 180 (+56.5%) | 0 |

**West African test tiles** (the 79 tiles above):

| Model | Precision | Recall | mAP@0.5 | Interior count vs. 518 by hand |
|---|---|---|---|---|
| Released (run 9) | 0.845 | 0.774 | 0.844 | 519 (+0.2%) |
| Fine-tuned on Peru only | 0.030 | 0.043 | 0.004 | 404 (−22.0%) |
| Trained on both sites | 0.817 | 0.785 | 0.852 | 522 (+0.8%) |

What this shows:

- **A few hundred labelled photos are enough to adapt the model to this imagery.** Both
  fine-tuned models find over 97% of the Peru test palms, against 47% before.
- **Fine-tuning on Peru alone destroys the model on its original site.** Its West African
  mAP@0.5 fell from 0.844 to 0.004. The two datasets box palms differently: Peru labels cover
  the full frond spread, West African labels the crown. After Peru-only training, the model's
  boxes on the West African tiles were about 1.8 times the labelled width (median 321–339 px
  against 180–184 px), so almost none matched a label, and its counts swung both ways (pilot
  149 against 311, block_003 255 against 207). That model is not released.
- **Training on both sites together avoids this.** The model trained on both sites matches the
  released model on the West African tiles and the Peru-only model on the Peru photos.
- **Whole-photo counts run high.** The fine-tuned models box palms cut off by the photo edge,
  which the Peru labels mostly leave out: all 47 of the Peru-only model's extra detections
  touch the photo edge, and 66 of the 67 from the model trained on both sites (14 of those 67
  are a second box on a palm already found). For per-photo counts, a higher confidence picked
  on the validation photos helps: 0.5 for the Peru-only model and 0.7 for the model trained on
  both sites bring both test counts to +12.2%, with 93.0% and 97.4% of palms found. The same happens
  at tile borders on the West African tiles, where the pipeline deals with it by overlapping
  tiles and counting interiors. Single photos have no neighbour to share an edge with, so to
  count palms over an area, stitch the photos into an orthomosaic and run the pipeline on that.
  Why the model trained on both sites boxes more of these edge palms than the Peru-only model
  has not been tested.
- **What this does not show.** The Peru test photos come from the same plantation and flights
  as the training photos, so this measures how well the model adapts to a site from 276
  labelled photos, not how it does on a new site without labels. Each result comes from one
  training run.

To use the model trained on both sites, give it palms at about the size it was trained on,
roughly 200 px across: 1280 px tiles for orthomosaics at about 2 cm/px, and a 320 px input for
800 × 600 photos like these.

## Files

- `test_counts_final.csv`, `test_counts_baseline.csv`, `test_counts_joint.csv` — one row per
  West African test tile: `manual_count` and `model_count` (whole tile), `matched` (detections
  matching a label at IoU ≥ 0.5), `manual_interior` and `model_interior`.
- `peru_split.csv` — the Peru split used by both notebooks: photo file name, photo number, the
  dataset's own split, the split used here, palms per photo, and a note for photos moved to
  training.
