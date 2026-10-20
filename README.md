# Oil Palm Crown Detection & Counting

Detects and counts individual oil palms in drone (UAV) RGB orthomosaics and turns them into a
GIS-ready palm inventory: one point and one crown box per palm, with confidence, crown area and
block ID, as GeoPackage and CSV.

RGB orthomosaic -> tiles -> YOLOv8 crown detector -> geo-referenced, de-duplicated detections -> palm inventory

Built for, and run on, a commercial oil palm plantation in West Africa:

| | |
|---|---|
| Area processed | 12,166.6 ha net (26 blocks of ~300 ha) |
| Palms detected | 718,751 (after removing cross-block double counts) |
| Imagery | UAV RGB orthomosaics (pilot: 2.13 cm/px) |
| Model | YOLOv8s, single class `oil_palm`, 1280 px tiles |
| Training data | 4,041 hand-labelled crowns on 435 tiles from two blocks |

How well it works, on 79 held-out test tiles (716 hand-labelled palms) from ground the model
never trained on:

| | Released model | Earlier baseline |
|---|---|---|
| Precision (share of detections that are real palms) | 0.845 | 0.727 |
| Recall (share of real palms found) | 0.774 | 0.637 |
| mAP@0.5 | 0.844 | 0.698 |
| Palm count vs. hand count, tile interiors | 519 vs 518 (+0.2%) | 454 vs 518 (−12.4%) |

Per-block results and the method for resolving overlapping block deliveries are in
[RESULTS.md](RESULTS.md). Model details, the full evaluation and limitations are in
[MODEL_CARD.md](MODEL_CARD.md) and [evaluation/](evaluation/).

## What's in this repository

- `scripts/` — the full pipeline: data QC, tiling, annotation sampling, dataset building,
  training, inference, de-duplication, canopy-height QC, counting accuracy, inventory merge,
  test-split evaluation and ONNX export.
- `models/oil_palm_crown-current/weights/best.pt` — the released model (training run 9).
- `models/oil_palm_crown-4/weights/best.pt` — the earlier baseline (run 4), kept for before/after comparison.
- `models/oil_palm_crown*/` — training curves, confusion matrices and settings for every training run.
- `evaluation/` — test-split results and per-tile hand vs. model counts.
- `docs/` — methodology, scaling workflow and a step-by-step runbook for new blocks.

The drone imagery, annotations and production outputs belong to the plantation owner and are
not included.

## Setup

Tested with Python 3.14, PyTorch 2.14 and Ultralytics 8.4 (the model was trained with
Ultralytics 8.4.122); Python 3.10 or newer should work.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU makes inference much faster. Without one, set `device: cpu` under `model:` in
`configs/project_config.yaml`.

## Count palms in your own orthomosaic

1. Put your RGB GeoTIFF somewhere on disk (the git-ignored `data/` folder works well).
2. Register it as a block in `configs/blocks.yaml` and set `crs` in
   `configs/project_config.yaml` to your imagery's projected CRS.
3. Run:

```bash
python scripts/qc_check.py --block <your_block_id>
python scripts/process_block.py --block <your_block_id> --weights models/oil_palm_crown-current/weights/best.pt
```

Results land in `outputs/inventory/<block>_palm_inventory.gpkg` (layers `crowns` and
`centroids`) and `.csv`. The palm count is the number of rows in `centroids`.

The model was trained on orthomosaics (about 2 cm/px) of a relatively young plantation.
Imagery at a very different resolution may need resampling, and imagery of tall mature palms
with heavily overlapping crowns needs fine-tuning: on an independent set of close-up photos of
mature palms from Peru it found at best about half the palms (see
[evaluation/](evaluation/#outside-imagery-an-independent-dataset-from-peru)). Check a sample
visually with `scripts/visualize_detections.py` before trusting a count. See [docs/NEW_BLOCK_RUNBOOK.md](docs/NEW_BLOCK_RUNBOOK.md) for the
full per-block workflow.

## Use in QGIS (no code)

Export the model to ONNX — the file carries the settings the
[Deepness](https://plugins.qgis.org/plugins/deepness/) QGIS plugin needs (resolution, tile
size, overlap, thresholds):

```bash
python scripts/export_onnx.py    # writes release/oil_palm_crown.onnx
```

In QGIS, install the Deepness plugin, choose **Detector**, select the `.onnx` file, load its
default parameters and run it over your orthomosaic. If asked for a batch size, use 1.

## Train on your own data

```bash
# 1. Data QC — CRS, resolution, nodata
python scripts/qc_check.py --block pilot

# 2. Tile the orthomosaic (1280 px chips, 256 px overlap by default)
python scripts/tile_orthomosaic.py --block pilot

# 3. Sample a stratified, representative subset of tiles for manual annotation
#    (spans greenness, texture and location so weed-only, palm-only and mixed scenes are covered)
python scripts/sample_tiles_for_annotation.py --block pilot --n-tiles 150

# 4. Annotate crowns in annotations/to_label/pilot/ — at least ~1,000 to start; the released
#    model used 4,041 (bounding boxes, class "oil_palm", YOLO .txt format — see ANNOTATION.md)

# 5. Build the train/val/test dataset (split by spatial sector, not random tiles)
python scripts/build_yolo_dataset.py --block pilot

# 6. Train
python scripts/train_yolo.py --epochs 150

# 6b. Score on the held-out test split: detection metrics + hand vs. model counts per tile
python scripts/evaluate_test_split.py --weights models/oil_palm_crown/weights/best.pt

# 7. Full-block inference -> merge tile detections -> geospatial inventory
python scripts/infer_pipeline.py --block pilot --weights models/oil_palm_crown-current/weights/best.pt

# 8. Counting accuracy against a manually counted area
python scripts/count_accuracy.py --block pilot --actual-count <known_count>

# 9. (Optional) Canopy-height QC — needs dsm/dtm in blocks.yaml
python scripts/chm_qc.py --block pilot
```

To fine-tune the released model instead of starting from COCO weights, pass
`--arch models/oil_palm_crown-current/weights/best.pt` to `train_yolo.py`.

## Outputs

`outputs/inventory/<block>_palm_inventory.gpkg` — two layers:
- `crowns`: crown bounding box + Tree_ID, Crown_ID, Confidence, Crown_Area_m2, Block_ID
- `centroids`: one point per palm (Easting/Northing), same attributes

`outputs/inventory/<block>_palm_inventory.csv` — flat table of the same fields.

## Scaling to many blocks

Each new block is: register in `configs/blocks.yaml` -> `scripts/process_block.py` (tile ->
infer -> geo-dedup -> CHM QC if elevation data is registered). No re-annotation or retraining
is needed unless a block shows a new failure pattern. `scripts/merge_master_inventory.py`
folds all processed blocks into one inventory.

If blocks overlap — common when imagery is delivered incrementally without a shared boundary
file — summing per-block counts double-counts palms. In the production run, naive summing gave
1,021,163 palms; resolving the overlap gave 718,751 (a 29.6% correction). The method is in
[RESULTS.md](RESULTS.md#cross-block-overlap-resolution-methodology-note).

## Licence

AGPL-3.0 — see [LICENSE](LICENSE). The detector is built on
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is itself AGPL-3.0.

## Citation

If you use this code or model, please cite it — see [CITATION.cff](CITATION.cff), or use
GitHub's "Cite this repository" button.

## Author

Oluwasola Oyinloye — [GitHub @Dexsholex](https://github.com/Dexsholex). Model released with the
permission of the plantation owner.
