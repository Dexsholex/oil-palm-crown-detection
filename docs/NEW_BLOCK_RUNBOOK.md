# Processing a New Block

You need the new block's RGB orthomosaic (`.tif`) and, optionally, its DSM/DTM (`.tif`)
for canopy-height QC. Everything below runs from a terminal at the repository root, with
the Python environment from the README activated.

## Step 1 — Put the files somewhere on disk

Any location works. The git-ignored `data/` folder is a convenient place — one folder per
block, e.g.:

```
data/block_002/rgb.tif
data/block_002/dsm.tif   (optional)
data/block_002/dtm.tif   (optional)
```

## Step 2 — Register the block

Open `configs/blocks.yaml` in any text editor. Add a new entry under `blocks:`, following
the same pattern as the entries already there:

```yaml
blocks:
  - block_id: pilot
    description: "Pilot subset (~85 ha) used to build and validate the workflow"
    rgb: "data/pilot/rgb.tif"
    dsm: "data/pilot/dsm.tif"
    dtm: "data/pilot/dtm.tif"

  - block_id: block_002
    description: "Second production block"
    rgb: "data/block_002/rgb.tif"
    dsm: "data/block_002/dsm.tif"
    dtm: "data/block_002/dtm.tif"
```

Rules:
- `block_id` must be unique — pick something short and consistent (`block_002`, `block_003`, ...).
- Use forward slashes `/` in paths, even on Windows. Relative paths resolve against the repository root.
- If a block has no DSM/DTM, write `dsm: null` and `dtm: null` instead of a path.
- Set `crs` in `configs/project_config.yaml` to your imagery's projected CRS.

## Step 3 — Run the pipeline on that block

One command does everything (tiling, detection, geo-referencing, de-duplication,
inventory, and CHM QC if DSM/DTM are registered):

```bash
python scripts/process_block.py --block block_002 --weights models/oil_palm_crown-current/weights/best.pt --chm-min-height 0.64
```

Swap `block_002` for whatever `block_id` you used in Step 2. `--chm-min-height 0.64` is
optional; drop it if the block has no DSM/DTM registered.

⚠️ **Always use `models/oil_palm_crown-current/weights/best.pt` for `--weights` — never a
numbered folder like `oil_palm_crown-4` or `oil_palm_crown-9` directly.**
`oil_palm_crown-current` is a stable pointer: its *contents* are overwritten with the latest
best checkpoint after each retrain, but the path never changes. During the original
production run, a command pinned to an older numbered folder caused blocks 010–019 to be
processed with an outdated model that missed many palms in dense canopy — don't
reintroduce that.

A full ~300 ha block takes a while (much larger than the ~85 ha pilot) — let it run.
Progress is printed as tiles get processed.

## Step 4 — Find your results

Once it finishes, look in `outputs/inventory/`:

- `block_002_palm_inventory.gpkg` — open in QGIS. Two layers: `crowns` (boxes) and `centroids` (points).
- `block_002_palm_inventory.csv` — flat table, opens in Excel.
- `block_002_palm_inventory_chm_qc.gpkg` — only if you ran with `--chm-min-height`; flags implausibly short detections for review.

The total palm count for that block = the number of rows in `centroids`.

## Step 5 — Merge into the master inventory

After processing one or more blocks, combine them all into one file:

```bash
python scripts/merge_master_inventory.py
```

This picks up every block registered in `blocks.yaml` that has been processed. Output:
`outputs/inventory/master_palm_inventory.gpkg` and `.csv`. If blocks overlap each other,
see "Cross-block overlap resolution" in RESULTS.md before summing counts.

## If something looks wrong

Run the data QC check on the new block *before* the full pipeline, to catch CRS or
resolution problems early:

```bash
python scripts/qc_check.py --block block_002
```

It checks that the CRS matches the project CRS, the resolution is sane, and there's no
unexpected nodata. If it fails, fix the input imagery before going further.

## That's the whole workflow

No annotation, no training, no code changes — just Steps 1–3 for every new block.
Step 5 whenever you want an updated total.

## Keeping `oil_palm_crown-current` up to date

`models/oil_palm_crown-current/weights/best.pt` is a plain copy, not a symlink, updated
by hand whenever a retrain produces a better model:

```bash
cp models/oil_palm_crown-<N>/weights/best.pt models/oil_palm_crown-current/weights/best.pt
```

If you retrain and a new numbered checkpoint folder appears, run that copy command so
`oil_palm_crown-current` actually points at the new best weights — otherwise every block
processed via Step 3 keeps silently using the old one. The released model in
`oil_palm_crown-current` is `oil_palm_crown-9` (trained on pilot data plus dense-canopy and
young-palm examples from block_003).
