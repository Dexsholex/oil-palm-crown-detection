---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - object-detection
  - yolov8
  - remote-sensing
  - uav
  - agriculture
  - oil-palm
  - tree-counting
---

# Oil palm crown detector (YOLOv8s)

Single-class object detector that finds individual oil palm crowns in drone (UAV) RGB
orthomosaic tiles. Paired with the pipeline in this repository, it turns a georeferenced
orthomosaic into a palm inventory (one point and crown box per palm). Used in production to
count 718,751 palms over 12,166.6 ha.

- **Developed by:** Oluwasola Oyinloye
- **Model type:** YOLOv8s object detector, one class (`oil_palm`), 1280 × 1280 px input
- **Weights:** `models/oil_palm_crown-current/weights/best.pt` (training run 9)
- **Code:** https://github.com/Dexsholex/oil-palm-crown-detection
- **Licence:** AGPL-3.0 (inherited from Ultralytics YOLOv8)

## Intended use

- Counting and mapping oil palms in nadir UAV RGB orthomosaics of plantations, at a resolution
  close to the training imagery (~2 cm/px).
- A starting point for fine-tuning on other plantations, or for pre-labelling new imagery so
  annotators correct boxes instead of drawing them from scratch.

**Not intended for:** satellite imagery, oblique photos, other tree species, or counts that
will be used without checking a sample against a manual count.

## Training data

- UAV RGB orthomosaics from one commercial oil palm plantation in West Africa. The imagery
  and labels are proprietary and are not released.
- 1280 × 1280 px tiles (256 px overlap), hand-annotated with one box per visible crown:

  | Source | Annotated tiles | Crown boxes |
  |---|---|---|
  | Pilot block (~85 ha, 2.13 cm/px), two stratified annotation batches covering dense/sparse palms, weeds, roads and bare soil | 286 | 2,541 |
  | block_003, targeted dense mature canopy and newly planted palms, added after those were under-detected in production | 149 | 1,500 |
  | **Total** | **435** | **4,041** |

- **Split:** 70/15/15 by spatial sector (a deterministic hash of each tile's map sector), not
  by random tiles, so validation and test tiles come from ground the model did not train on:

  | Split | Tiles (pilot + block_003) | Crown boxes |
  |---|---|---|
  | Train | 313 (214 + 99) | 2,898 |
  | Validation | 43 (21 + 22) | 427 |
  | Test | 79 (51 + 28) | 716 |

- The plantation is young: canopy height at 1,226 crowns from the first pilot annotation batch
  has a median of 2.33 m (5th–95th percentile 0.92–3.81 m).

## Training procedure

- Started from COCO-pretrained YOLOv8s and refined over successive runs; the released
  checkpoint (run 9) was fine-tuned from run 4.
- Run 9: image size 1280, batch 6, AdamW, initial learning rate 0.0002, up to 150 epochs with
  early-stopping patience 30 (stopped at epoch 92, best at epoch 62).
- Augmentation: rotation up to ±180°, horizontal and vertical flips, mosaic. Perspective and
  shear off (not physical for nadir imagery).
- Hardware: one NVIDIA RTX 4070 Laptop GPU (8 GB).
- Full settings and curves for every run are in `models/oil_palm_crown*/`.

## Evaluation

**Held-out test split** (79 tiles, 716 labelled crowns, never used for training or for picking
the checkpoint), scored with Ultralytics `val`:

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Released model (run 9) | **0.845** | **0.774** | **0.844** | **0.350** |
| Earlier baseline (run 4) | 0.727 | 0.637 | 0.698 | 0.279 |

How to read these:
- About 85% of detections are real palms, and the model finds about 77 of every 100 labelled
  palms. Precision and recall both rose over the baseline, so the gain is not a trade-off.
- Ultralytics reports precision and recall at the confidence threshold that maximises F1, not
  at the pipeline's 0.25 — see counting accuracy below for the pipeline setting.
- The low mAP@0.5:0.95 mostly reflects how tightly boxes fit crown edges, which matters less
  for counting.
- For reference, the validation split at the best epoch gave precision 0.837, recall 0.746,
  mAP@0.5 0.839, mAP@0.5:0.95 0.362.
- Reproduce with `python scripts/evaluate_test_split.py --weights models/oil_palm_crown-current/weights/best.pt`
  (needs the labelled tiles, which are not public).

**Counting accuracy** — model count vs. hand count on the same 79 held-out test tiles, at the
pipeline's settings (confidence 0.25, NMS IoU 0.45):

| Counted region | Hand count | Model count | Difference |
|---|---|---|---|
| Tile interior (box centre more than 128 px from the border) | 518 | 519 | **+0.2%** |
| Whole tile, including palms cut off by the border | 716 | 894 | +24.9% |

- The interior is the fair comparison for a block run: tiles overlap by 256 px, so a palm cut
  off at one tile's border is seen whole in the next tile, and duplicates are merged. The
  interior is what each tile contributes.
- The whole-tile overcount comes from partial crowns at the border, which the annotators mostly
  did not box but the model detects: 206 of the 301 unmatched detections touch the tile edge.
- The interior count is close partly because errors cancel: roughly 100 labelled palms away
  from the edges are missed, and a similar number of detections there have no matching label.
  Totals over many palms are reliable; individual detections are not all correct.
- By block (interior): pilot 321 vs 311 by hand (+3.2%), block_003 198 vs 207 (−4.3%); mean
  absolute error 0.57 palms per tile.
- The baseline (run 4) counted 454 interior palms against 518 (−12.4%), missing about 31% on
  block_003's dense canopy and young palms. Retraining on those cases fixed most of it.
- On the 43 validation tiles the interior count was 304 against 317 by hand (−4.1%).
- These are tile-level checks on held-out ground, not a field count over a whole block.
- Per-tile numbers: [evaluation/](evaluation/).

**Production run:** 26 blocks, 12,166.6 ha net, 718,751 palms after removing cross-block
double counts. Per-block results are in [RESULTS.md](RESULTS.md).

## Limitations

- Trained on one plantation in West Africa, with mostly young palms. It **does not transfer to
  close-up drone photos of mature palms with overlapping crowns**: on an independent dataset
  from Peru (424 photos, 549 palms) it found at best about half the palms, and strict
  mAP@0.5 stayed below 0.05 — see [evaluation/](evaluation/). Fine-tune before using it on
  imagery like that. Other plantations imaged like the training data (orthomosaics at about
  2 cm/px, young palms) remain untested.
- Sensitive to resolution: resample imagery far from ~2 cm/px, or fine-tune.
- Known weak spots during rollout were dense mature canopy and very small, newly planted
  palms. Run 9 was trained to reduce both, but they remain the most likely sources of missed
  palms.
- Count accuracy relies partly on misses and false detections cancelling out (see Evaluation).
  Check a sample visually before relying on counts for a small area.
- Other palm species and similar-looking vegetation were not specifically tested.
- Boxes, not crown outlines: touching crowns can occasionally merge into one detection.

## How to use

For whole orthomosaics, use the repository pipeline — it handles tiling, geo-referencing and
removing duplicates where tiles overlap:

```bash
python scripts/process_block.py --block <block_id> --weights models/oil_palm_crown-current/weights/best.pt
```

For QGIS (Deepness plugin) or ONNX Runtime, export with `python scripts/export_onnx.py`. The
ONNX model takes `1×3×1280×1280` RGB scaled 0–1 and returns `1×5×33600` (box centre x, y,
width, height, palm score) before non-max suppression; it matches the PyTorch model to within
0.001 px.

For a single image tile:

```python
from ultralytics import YOLO

model = YOLO("models/oil_palm_crown-current/weights/best.pt")
result = model.predict("tile.png", imgsz=1280, conf=0.25)[0]
print(len(result.boxes), "palms")
```

## Citation

See [CITATION.cff](CITATION.cff).
