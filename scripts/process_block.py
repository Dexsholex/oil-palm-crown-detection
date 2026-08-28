"""
End-to-end production run for ONE block: tile -> infer -> geo-dedup -> centroids ->
inventory -> CHM QC. This is what gets run for each of the ~20 production blocks once
the model is finalized — no re-annotation or re-training needed, per the brief's
"train once, reuse everywhere" workflow.

Usage:
    python scripts/process_block.py --block block_003 --weights models/oil_palm_crown-current/weights/best.pt
    python scripts/process_block.py --block block_003 --weights models/oil_palm_crown-current/weights/best.pt --skip-tiling
    python scripts/process_block.py --block block_003 --weights models/oil_palm_crown-current/weights/best.pt --chm-min-height 0.64

Always point --weights at models/oil_palm_crown-current/weights/best.pt (a stable pointer
copy, updated whenever a retrain improves on it) rather than a specific numbered folder —
see docs/NEW_BLOCK_RUNBOOK.md for why.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import load_config, get_block, PROJECT_ROOT


def run(cmd, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAILED] {label} (exit {result.returncode}, {elapsed:.0f}s) — stopping.", flush=True)
        sys.exit(result.returncode)
    print(f"[OK] {label} ({elapsed:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, help="block_id from configs/blocks.yaml")
    ap.add_argument("--weights", required=True, help="path to the production model checkpoint")
    ap.add_argument("--skip-tiling", action="store_true",
                     help="skip if tiles/<block>/ already exists from a prior run")
    ap.add_argument("--skip-qc-check", action="store_true")
    ap.add_argument("--chm-min-height", type=float, default=None,
                     help="run CHM QC with this floor if the block has dsm/dtm registered "
                          "(omit to skip CHM QC entirely)")
    args = ap.parse_args()

    cfg = load_config()
    block = get_block(args.block)  # raises clearly if block_id isn't in blocks.yaml
    python = sys.executable

    print(f"Processing block '{args.block}' -> {block['rgb']}", flush=True)
    t_start = time.time()

    if not args.skip_qc_check:
        run([python, "scripts/qc_check.py", "--block", args.block], "1. Data QC")

    # A prior tiling run that crashed early (e.g. process killed, out of memory) can leave
    # behind a tile_index.csv that exists but is empty or header-only — treat that the same
    # as "not tiled" rather than silently skipping tiling and feeding infer_pipeline.py a
    # dataset with 0 tiles (block_017 hit exactly this: 0-byte tile_index.csv from an
    # earlier interrupted run, misread as "already tiled").
    tile_index_path = PROJECT_ROOT / cfg["paths"]["tiles_dir"] / args.block / "tile_index.csv"
    tiles_exist = tile_index_path.exists() and tile_index_path.stat().st_size > 200
    if args.skip_tiling or tiles_exist:
        print(f"\n[skip] Tiling — tiles/{args.block}/ already present", flush=True)
    else:
        run([python, "scripts/tile_orthomosaic.py", "--block", args.block], "2. Tiling")

    run([python, "scripts/infer_pipeline.py", "--block", args.block, "--weights", args.weights],
        "3. Inference + geo-dedup + centroids + inventory")

    if args.chm_min_height is not None:
        if block.get("dsm") and block.get("dtm"):
            run([python, "scripts/chm_qc.py", "--block", args.block,
                 "--min-height-m", str(args.chm_min_height)], "4. CHM QC")
        else:
            print(f"\n[skip] CHM QC — block '{args.block}' has no dsm/dtm registered in blocks.yaml",
                  flush=True)

    total = time.time() - t_start
    print(f"\n{'='*70}\nDone: block '{args.block}' in {total/60:.1f} min\n{'='*70}", flush=True)
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs_dir"] / "inventory"
    print(f"Inventory: {out_dir / f'{args.block}_palm_inventory.gpkg'}", flush=True)
    print(f"CSV:       {out_dir / f'{args.block}_palm_inventory.csv'}", flush=True)


if __name__ == "__main__":
    main()
