# Processing a New Block — 

You need three things before starting: the new block's RGB orthomosaic (`.tif`), and
optionally its DSM/DTM (`.tif`) if the client provided them. Everything below runs from
a terminal at `E:\SHOLA ARPN\compressed\oil_palm_pipeline`.

## Step 1 — Put the files somewhere on disk

Any location works. To stay organized, mirror the pilot's pattern — put each block's
files in their own folder next to the pipeline, e.g.:

```
E:\SHOLA ARPN\compressed\block_002\rgb.tif
E:\SHOLA ARPN\compressed\block_002\dsm.tif   (optional)
E:\SHOLA ARPN\compressed\block_002\dtm.tif   (optional)
```

## Step 2 — Register the block

Open `configs\blocks.yaml` in any text editor (Notepad is fine). Add a new entry under
`blocks:`, following the exact same pattern as the `pilot` entry already there:

```yaml
blocks:
  - block_id: pilot
    description: "Pilot dev subset (~85 ha) used to build and validate the workflow"
    rgb: "E:/SHOLA ARPN/compressed/data/Samplesize.tif"
    dsm: "E:/SHOLA ARPN/compressed/dsm data/dsm.tif"
    dtm: "E:/SHOLA ARPN/compressed/dtm data/dtm.tif"

  - block_id: block_002
    description: "Second production block"
    rgb: "E:/SHOLA ARPN/compressed/block_002/rgb.tif"
    dsm: "E:/SHOLA ARPN/compressed/block_002/dsm.tif"
    dtm: "E:/SHOLA ARPN/compressed/block_002/dtm.tif"
```

Rules:
- `block_id` must be unique — pick something short and consistent (`block_002`, `block_003`, ...).
- Use forward slashes `/` in paths, even on Windows.
- If a block has no DSM/DTM yet, write `dsm: null` and `dtm: null` instead of a path.
- Save the file when done.

## Step 3 — Run the pipeline on that block

One command does everything (tiling, detection, geo-referencing, de-duplication,
inventory, and CHM QC if DSM/DTM are registered):

```bash
.venv/Scripts/python.exe scripts/process_block.py --block block_002 --weights models/oil_palm_crown-current/weights/best.pt --chm-min-height 0.64
```

Swap `block_002` for whatever `block_id` you used in Step 2.

⚠️ **Always use `models/oil_palm_crown-current/weights/best.pt` for `--weights` — never a
numbered folder like `oil_palm_crown-4` or `oil_palm_crown-9` directly.** The model gets
retrained periodically as detection quality improves (dense-canopy misses, young-palm
misses, etc. get fixed by annotating more data and retraining). `oil_palm_crown-current`
is a stable pointer — its *contents* get overwritten with the latest best checkpoint each
time a retrain improves on it, but the path itself never changes, so this command in this
doc never goes stale. Pinning a command to a specific numbered folder (as an earlier
version of this doc did, pointing at `oil_palm_crown-4`) is exactly what caused blocks
010–019 to be processed with an outdated model and miss a large fraction of palms in dense
canopy — don't reintroduce that. `--chm-min-height 0.64` is optional; drop it if the block
has no DSM/DTM registered.

This can take a while for a full 300 ha block (much larger than the ~85 ha pilot) — let
it run. You'll see progress printed as tiles get processed.

## Step 4 — Find your results

Once it finishes, look in `outputs\inventory\`:

- `block_002_palm_inventory.gpkg` — open in QGIS. Two layers: `crowns` (boxes) and `centroids` (points).
- `block_002_palm_inventory.csv` — flat table, opens in Excel.
- `block_002_palm_inventory_chm_qc.gpkg` — only if you ran with `--chm-min-height`, flags implausibly short detections for review.

The total palm count for that block = the number of rows in `centroids`.

## Step 5 — Merge into the master plantation-wide inventory

After processing one or more blocks, combine them all into one file:

```bash
.venv/Scripts/python.exe scripts/merge_master_inventory.py
```

This automatically picks up every block registered in `blocks.yaml` that has been
processed. Output: `outputs\inventory\master_palm_inventory.gpkg` and `.csv`.

## If something looks wrong

Run the data QC check on the new block *before* the full pipeline, to catch CRS or
resolution problems early:

```bash
.venv/Scripts/python.exe scripts/qc_check.py --block block_002
```

It checks CRS matches EPSG:32631, resolution is sane, and there's no unexpected nodata.
If it fails, don't proceed — check the file the client sent you first.

## That's the whole workflow

No annotation, no training, no code changes — just Steps 1–3 for every new block you
receive. Step 5 whenever you want an updated whole-plantation total.

## Keeping `oil_palm_crown-current` up to date

`models/oil_palm_crown-current/weights/best.pt` is a plain copy, not a symlink, updated
by hand whenever a retrain produces a better model:

```bash
cp models/oil_palm_crown-<N>/weights/best.pt models/oil_palm_crown-current/weights/best.pt
```

If you retrain the model yourself (or ask for a retrain) and a new numbered checkpoint
folder appears, run that copy command so `oil_palm_crown-current` actually points at the
new best weights — otherwise every block processed via Step 3 keeps silently using the old
one, same failure mode as the 010–019 incident. As of this note, current best is
`oil_palm_crown-9` (trained on combined pilot + block_003 dense-canopy/young-palm data).
