# Annotating oil palm crowns

The released model was trained on 4,041 hand-drawn crown boxes on 435 tiles: 2,541 boxes on
286 stratified tiles of the pilot block (two annotation batches), plus 1,500 boxes on 149
tiles of dense mature canopy and newly planted palms from block_003, added during rollout.
This is how they were made, so you can annotate your own imagery the same way.

## Tool

**CVAT** (self-hosted via Docker, browser-based). Any tool that exports YOLO-format boxes
works; CVAT was chosen because labelImg crashed repeatedly on the annotation machine.

```bash
docker compose up -d      # start (from your CVAT checkout)
docker compose down       # stop
```

Set up a project with a single rectangle label `oil_palm`, then create a task with the tiles
sampled by `scripts/sample_tiles_for_annotation.py` (in `annotations/to_label/<block>/`).

## Annotating

1. Use the rectangle tool in **Shape** mode — not **Track**. Track carries boxes forward onto
   later frames, which is wrong here: the tiles are independent images, not video.
2. One tight box per visible crown canopy (not trunk or shadow). Save often.
3. Frames with no palms: draw nothing and move on — an empty frame is a valid
   background example.
4. Deliberately include: large/mature palms, young/small palms, overlapping/crowded crowns,
   palms in weeds vs. clean ground, shadowed/partly occluded palms, and plenty of pure
   weed/bush/road/bare-soil frames.
5. No fixed per-tile box count — box whatever is actually visible, including zero. Aim for
   at least ~1,000 boxes to start; if you're well short after all sampled tiles, sample more
   tiles to top up.
6. Keep a label file for every tile you reviewed, even an empty one. `build_yolo_dataset.py`
   treats a tile with an empty `.txt` as a background example, but a tile with no `.txt` at all
   as not yet annotated, and leaves it out.

## Getting annotations back into the pipeline

When done (or partway through, to sanity-check):

1. In CVAT: task menu → **Export task dataset** → format **"Ultralytics YOLO Detection 1.0"**
   → download the zip.
2. Run:
   ```bash
   python scripts/import_cvat_export.py --block pilot --zip <path to downloaded zip>
   ```
   This copies the label `.txt` files back into `annotations/to_label/pilot/`, matched by
   filename to the tiles already there, and reports the running total of annotated tiles/boxes.
3. Once happy with coverage, run `scripts/build_yolo_dataset.py` as documented in the main
   [README.md](README.md).
