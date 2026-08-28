# Oil Palm Crown Detection & Counting — Production Pipeline

RGB orthomosaic -> tiles -> YOLO crown detector -> centroids -> geospatial palm inventory.
Baseline production workflow per the project brief: bounding-box detection first,
single class `oil_palm`, everything else is background.

## Setup

```bash
.venv\Scripts\pip install -r requirements.txt
```

GPU: NVIDIA RTX 4070 Laptop (8GB VRAM) detected. PyTorch installed with CUDA 12.4 wheels.

## Blocks

Each 300-ha production block is one entry in `configs/blocks.yaml` (rgb/dsm/dtm paths).
The pilot dev subset is already registered as block `pilot`.

## Workflow

```bash
# 1. Data QC — CRS, resolution, nodata check
.venv\Scripts\python scripts\qc_check.py --block pilot

# 2. Tile the orthomosaic (1280px chips, 256px overlap by default)
.venv\Scripts\python scripts\tile_orthomosaic.py --block pilot

# 3. Sample a stratified, representative subset of tiles for manual annotation
#    (spans greenness / texture / spatial location so weed-only, palm-only and
#    mixed scenarios all get covered — see brief sections 7 & 9)
.venv\Scripts\python scripts\sample_tiles_for_annotation.py --block pilot --n-tiles 150

# 4. Annotate ~1000 crowns manually in annotations/to_label/pilot/
#    (bounding boxes, class "oil_palm", YOLO .txt format — see ANNOTATION.md)

# 5. Build the train/val/test YOLO dataset (split by spatial sector, not random tiles)
.venv\Scripts\python scripts\build_yolo_dataset.py --block pilot

# 6. Train
.venv\Scripts\python scripts\train_yolo.py --epochs 150

# 7. Run full-block inference -> merge tile detections -> geospatial inventory
.venv\Scripts\python scripts\infer_pipeline.py --block pilot --weights models\oil_palm_crown-current\weights\best.pt

# 8. Counting accuracy against a manually-counted validation area
.venv\Scripts\python scripts\count_accuracy.py --block pilot --actual-count <known_count>

# 9. (Optional, only if error analysis shows it helps) CHM-based QC, requires dsm/dtm in blocks.yaml
.venv\Scripts\python scripts\chm_qc.py --block pilot
```

## Outputs

`outputs/inventory/<block>_palm_inventory.gpkg` — two layers:
- `crowns`: crown bounding-box geometry + Tree_ID, Crown_ID, Confidence, Crown_Area_m2, Block_ID
- `centroids`: one point per palm (Easting/Northing) — same attributes

`outputs/inventory/<block>_palm_inventory.csv` — flat table of the same fields.

All geospatial outputs stay in EPSG:32631 (WGS 84 / UTM zone 31N).

## Scaling to production

Once the pilot model/workflow is validated (>=90% counting accuracy target, ideally >=95%):
1. Add each new block's rgb/dsm/dtm path to `configs/blocks.yaml` (`dsm`/`dtm`: `null` if
   not yet delivered for that block).
2. Run `scripts/process_block.py --block <id> --weights models/oil_palm_crown-current/weights/best.pt`
   (tile -> infer -> geo-dedup -> CHM QC if elevation data is registered) — no
   re-annotation or re-training needed per block, unless error analysis reveals a
   systematic new failure mode (in which case: hard-negative mine, add to training data,
   retrain, then update `models/oil_palm_crown-current/` to point at the improved
   checkpoint — see `docs/NEW_BLOCK_RUNBOOK.md`).
3. Run `scripts/merge_master_inventory.py` to fold every processed block's `centroids`
   layer into one plantation-wide inventory GeoPackage/CSV.

As of this writing the pipeline has processed 26 production blocks — 12,166.6 ha net
coverage, 718,751 detected palms after cross-block overlap resolution — on
`oil_palm_crown-current`, the model retrained on dense-canopy and young-palm failure
cases found during production rollout. See [RESULTS.md](RESULTS.md) for the full
per-block breakdown and methodology notes.
