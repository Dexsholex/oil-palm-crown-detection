"""
Train the YOLO oil-palm crown detector on the dataset built by build_yolo_dataset.py.

Usage:
    python scripts/train_yolo.py --epochs 150
    python scripts/train_yolo.py --epochs 150 --arch yolov8m.pt --batch 8
"""
import argparse

from ultralytics import YOLO

from common import load_config, PROJECT_ROOT


def main():
    cfg = load_config()
    m = cfg["model"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--arch", default=m["arch"])
    ap.add_argument("--imgsz", type=int, default=m["imgsz"])
    ap.add_argument("--batch", type=int, default=m["batch"])
    ap.add_argument("--device", default=str(m["device"]))
    ap.add_argument("--patience", type=int, default=30, help="early-stop patience")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "False"],
                     help="cache decoded images to avoid re-reading/re-resizing JPEGs every epoch "
                          "(dataset is small enough to fit in RAM easily)")
    ap.add_argument("--degrees", type=float, default=180)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--amp", default="True", choices=["True", "False"])
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--warmup-epochs", type=float, default=3.0, help="lower to ~0 when "
                     "continuing from an already-converged checkpoint, to avoid a fresh "
                     "warmup cycle disrupting it")
    args = ap.parse_args()
    cache = False if args.cache == "False" else args.cache
    amp = args.amp == "True"

    data_yaml = PROJECT_ROOT / cfg["paths"]["yolo_dataset_dir"] / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Missing {data_yaml} — run build_yolo_dataset.py first")

    model = YOLO(args.arch)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        cache=cache,
        lr0=args.lr0,
        optimizer=args.optimizer,
        amp=amp,
        warmup_epochs=args.warmup_epochs,
        project=str(PROJECT_ROOT / cfg["paths"]["models_dir"]),
        name="oil_palm_crown",
        # augmentations sane for top-down UAV crowns: rotation/flip help since there's
        # no canonical "up" in an orthomosaic; keep perspective/shear off (not physical for nadir imagery)
        degrees=args.degrees,
        fliplr=0.5,
        flipud=0.5,
        perspective=0.0,
        shear=0.0,
        mosaic=args.mosaic,
    )

    metrics = model.val(data=str(data_yaml), split="test")
    print("\n=== Held-out test metrics ===")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
