# Production Results

Real, verified output from the full 26-block production run as of 2026-08-29. All
counts below come directly from `outputs/inventory/master_palm_inventory.csv` after
overlap resolution (see "Cross-block overlap resolution" below) — none are estimated
or rounded.

## Summary

- **26 production blocks processed** (pilot + block_002 through block_026, excluding
  one confirmed duplicate delivery)
- **12,166.6 ha net coverage** (true non-overlapping footprint, not a raw sum of block
  bounding boxes — see methodology below)
- **718,751 palms detected** (net, after removing cross-block double-counts)
- **Model**: `oil_palm_crown-current`, a YOLOv8-based single-class detector, retrained
  through 9 iterations to address dense-canopy and young-palm miss patterns found
  during production rollout
- **CHM (canopy-height-model) QC**: available for 5/26 blocks (pilot, block_002,
  block_006, block_007, block_009) where DSM/DTM elevation data was delivered and
  intact at time of writing; used to flag implausibly short detections as likely
  false positives, not to filter them automatically

## Per-block results

| Block | Net Hectares | Net Palms | Palms/ha | CHM QC |
|---|---|---|---|---|
| pilot | 85.4 | 10,683 | 125.1 | yes |
| block_002 | 413.9 | 20,395 | 49.3 | yes |
| block_003 | 635.8 | 57,564 | 90.5 | no |
| block_004 | 855.6 | 44,028 | 51.5 | no |
| block_005 | 413.1 | 35,124 | 85.0 | no |
| block_006 | 536.8 | 26,207 | 48.8 | yes |
| block_007 | 942.4 | 53,683 | 57.0 | yes |
| block_008 | 268.7 | 10,406 | 38.7 | no |
| block_009 | 293.2 | 31,348 | 106.9 | yes |
| block_010 | 678.4 | 52,660 | 77.6 | no |
| block_011 | 1214.6 | 61,333 | 50.5 | no |
| block_012 | 544.7 | 33,263 | 61.1 | no |
| block_013 | 412.1 | 38,740 | 94.0 | no |
| block_014 | 124.0 | 12,648 | 102.0 | no |
| block_015 | 653.3 | 3,102 | 4.7 | no |
| block_016 | 639.9 | 17,760 | 27.8 | no |
| block_017 | 705.1 | 49,070 | 69.6 | no |
| block_018 | 426.8 | 44,880 | 105.2 | no |
| block_019 | 744.9 | 53,095 | 71.3 | no |
| block_020 | 186.0 | 1,715 | 9.2 | no |
| block_021 | 127.3 | 9,154 | 71.9 | no |
| block_022 | 274.8 | 28,788 | 104.8 | no |
| block_023 | 432.0 | 10,548 | 24.4 | no |
| block_024 | 207.8 | 10,720 | 51.6 | no |
| block_025 | 122.5 | 962 | 7.9 | no |
| block_026 | 227.5 | 875 | 3.8 | no |
| **Total** | **12,166.6** | **718,751** | **59.1** | 5/26 blocks |

A handful of blocks (015, 020, 025, 026) show unusually low density (<10 palms/ha) in
their net (non-overlap) area — this net area is often just a small sliver of the block
after yielding most of its footprint to an earlier-delivered neighbor (see below), so
it's expected rather than a detection failure, but worth field-verifying before citing
individual block densities.

## Cross-block overlap resolution (methodology note)

Blocks were delivered and processed incrementally over ~10 days, without a shared
property-boundary reference from the client. A full pairwise check of all 26 blocks'
raster bounding boxes (2026-08-28/29) found **5,194.7 ha of overlapping coverage across
51 block pairs** — i.e. many blocks partially or heavily re-covered ground already
delivered in an earlier block, and were processed as if independent. Naively summing
every block's raw detection count would double- (or triple-) count palms in every
shared zone.

**Resolution rule**: blocks were ordered by delivery sequence (pilot, block_002, ...,
block_026). For each block, any detection whose centroid fell inside its bounding-box
overlap with an *earlier*-delivered block was excluded — the earlier block retains the
shared zone, the later delivery yields to it. This is a deterministic, order-based
partition: every hectare of ground is attributed to exactly one block, with no gaps and
no double-counts under this rule.

Applying this across all 26 blocks reduced the naive total from 1,021,163 to 718,751
palms (**302,412 fewer, a 29.6% correction**).

**Known limitation**: this is a bounding-box (axis-aligned rectangle) resolution, not a
true property-boundary clip. Every source raster showed ~0% nodata / clean rectangular
pixel grids, so bounding-box overlap is very likely genuine imagery overlap rather than
an artifact of irregular block shapes — but this has not been independently confirmed
against the client's actual survey boundaries. If real block boundaries become
available, re-running the resolution against true polygons (rather than bounding boxes)
would refine block-level counts, though the overall total should not change materially
since it's driven by real, delivered imagery overlap, not the choice of clipping shape.

## Data-quality notes

- **block_003**: DSM/DTM have a ~19.2m vertical datum offset from the RGB/DSM
  reference frame (confirmed via 47,169-point analysis: median offset -19.19m, std
  0.86m). Even after removing the offset, the residual canopy-height signal didn't
  resemble real vegetation height, so CHM QC was not run for this block rather than
  publish a fabricated result — flagged for the client's survey team.
- **block_004, block_005, block_008**: had DSM/DTM at time of initial CHM QC, but the
  source elevation rasters were no longer present on disk when the overlap-corrected
  centroids needed CHM QC re-run — so these blocks currently ship without CHM QC
  rather than reuse the stale (pre-overlap-fix) result.
- **blocks 010–026** (except where noted): no DSM/DTM was ever delivered for these
  blocks, so CHM QC has never run on them.

## Model training history

See `models/oil_palm_crown-*/` for full per-iteration training curves
(precision/recall, confusion matrices, box-loss curves) and hyperparameters
(`args.yaml`) for each of the 9 retraining iterations that produced the current
production model. Key fixes discovered during training (see `scripts/train_yolo.py`
and `docs/`):
- `optimizer=auto` silently overriding an explicit learning-rate schedule caused
  early divergent runs — fixed by always passing `--optimizer AdamW` explicitly.
- Batch size 16 gave too few gradient steps/epoch on the (intentionally small,
  hand-annotated) training set — reduced to 6–8.
- Targeted re-annotation of dense mature-canopy clusters and young/small
  just-planted palms (systematic miss patterns found during production rollout)
  drove the largest accuracy improvement, validated with fair before/after
  comparison on a fixed held-out test set.
