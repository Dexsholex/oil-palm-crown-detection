# Annotating the ~1,000 oil palm crowns

## Tool

**CVAT** (self-hosted via Docker, browser-based) — [http://localhost:8080](http://localhost:8080).

(labelImg was tried first but hit a reproducible hard crash — `Qt5Core.dll` fault on this
machine's PyQt5/Qt 5.15.2 build when drawing after zooming — so we moved to CVAT instead.)

Bring the stack up/down from `~/cvat`:
```bash
docker compose up -d      # start
docker compose down       # stop
```

## Setup already done

- Project `oil_palm_pilot` created, label `oil_palm` (rectangle).
- Task `batch1_annotation` created with all 150 sampled tiles from
  `annotations/to_label/pilot` uploaded (150/150 confirmed via API).

## Annotating

1. Open the task, use the rectangle tool (**Shape** mode — NOT **Track**; Track carries
   boxes forward onto later frames automatically, which we don't want since these tiles
   are independent images, not video).
2. One tight box per visible crown canopy (not trunk/shadow). `Ctrl+S` to save periodically
   (CVAT also autosaves).
3. Frames with no palms: just don't draw anything and move on — an empty frame is a valid
   negative/background example.
4. Deliberately seek out, per the project brief: large/mature palms, young/small palms,
   overlapping/crowded crowns, palms in weeds vs. clean ground, shadowed/partially occluded
   palms, and a good number of pure weed/bush/road/bare-soil frames.
5. No fixed per-tile box count — box whatever's actually visible, honestly, including zero.
   Total target across all 150 tiles is ~1,000 boxes; if well short after finishing all 150,
   more tiles can be sampled from the remaining ~1,683 to top up.

## Getting annotations back into the pipeline

When done (or partway through, to sanity-check):

1. In CVAT: task menu → **Export task dataset** → format **"Ultralytics YOLO Detection 1.0"**
   → download the zip.
2. Run:
   ```bash
   .venv\Scripts\python.exe scripts\import_cvat_export.py --block pilot --zip <path to downloaded zip>
   ```
   This copies the label `.txt` files back into `annotations/to_label/pilot/`, matched by
   filename to the tiles already there, and reports the running total of annotated tiles/boxes.
3. Once happy with coverage, run `scripts/build_yolo_dataset.py` as documented in the main
   [README.md](README.md).
