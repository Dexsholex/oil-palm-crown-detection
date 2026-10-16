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
- **Pilot block:** ~85 ha at 2.13 cm/px. 150 tiles (1280 px, 256 px overlap) sampled to cover
  dense/sparse palms, weeds, roads and bare soil; 1,226 crowns hand-annotated as boxes.
- **Rollout additions:** targeted examples of dense mature canopy and newly planted palms from
  a second block (block_003), added after those patterns were under-detected in production.
  The number of added boxes was not recorded in this repository.
- **Split:** train/val/test by spatial sector, not by random tiles, so validation tiles come
  from ground the model did not train on.
- The plantation is young: canopy height at the 1,226 pilot crowns has a median of 2.33 m
  (5th–95th percentile 0.92–3.81 m).

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

Validation split, best epoch of run 9:

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| 0.837 | 0.746 | 0.839 | 0.362 |

How to read these:
- About 84% of detections are real palms, and the model finds about 3 in 4 annotated palms
  in the validation tiles.
- The validation split was also used to pick the best epoch, so these numbers are slightly
  optimistic.
- They are box-matching scores, not counting accuracy. The low mAP@0.5:0.95 mostly reflects
  how tightly boxes fit crown edges, which matters less for counting.

**Counting accuracy** (model count vs. manual count on reference images): being compiled —
this section will be updated.

**Production run:** 26 blocks, 12,166.6 ha net, 718,751 palms after removing cross-block
double counts. Per-block results are in [RESULTS.md](RESULTS.md).

## Limitations

- Trained on one plantation in West Africa, with mostly young palms. Performance on other
  plantations, drones, seasons, or tall mature palms with heavily overlapping crowns has not
  been measured.
- Sensitive to resolution: resample imagery far from ~2 cm/px, or fine-tune.
- Known weak spots during rollout were dense mature canopy and very small, newly planted
  palms. Run 9 was trained to reduce both, but they remain the most likely sources of missed
  palms.
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
