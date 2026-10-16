"""
Export the released detector to ONNX for use outside Python/PyTorch — e.g. the Deepness
QGIS plugin — and embed the metadata Deepness reads (model type, class names, resolution,
tile size/overlap, thresholds).

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights models/oil_palm_crown-current/weights/best.pt --out release/oil_palm_crown.onnx
"""
import argparse
import json
from pathlib import Path

import onnx
from ultralytics import YOLO

from common import load_config, resolve_path

# Ground sampling distance of the pilot orthomosaic the model was mainly trained on.
TRAINING_GSD_CM = 2.13
REPO_URL = "https://github.com/Dexsholex/oil-palm-crown-detection"


def set_metadata(model: onnx.ModelProto, key: str, value: str):
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    prop = model.metadata_props.add()
    prop.key, prop.value = key, value


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/oil_palm_crown-current/weights/best.pt")
    ap.add_argument("--out", default="release/oil_palm_crown.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    imgsz = cfg["model"]["imgsz"]
    exported = Path(YOLO(str(resolve_path(args.weights))).export(
        format="onnx", imgsz=imgsz, opset=args.opset, simplify=True, device="cpu"))

    model = onnx.load(str(exported))
    tiling = cfg["tiling"]
    # Deepness expects every value JSON-encoded; tiles_overlap is a percentage of the tile size.
    deepness = {
        "model_type": "Detector",
        "det_type": "YOLO_Ultralytics",
        "class_names": {i: name for i, name in enumerate(cfg["classes"])},
        "resolution": TRAINING_GSD_CM,
        "tiles_size": tiling["size_px"],
        "tiles_overlap": round(100 * tiling["overlap_px"] / tiling["size_px"]),
        "det_conf": cfg["inference"]["conf_thresh"],
        "det_iou_thresh": cfg["inference"]["iou_thresh"],
        "standardization_mean": [0.0, 0.0, 0.0],
        "standardization_std": [1.0, 1.0, 1.0],
    }
    for key, value in deepness.items():
        set_metadata(model, key, json.dumps(value))
    set_metadata(model, "description", f"Oil palm crown detector (YOLOv8s) - {REPO_URL}")
    set_metadata(model, "author", "Oluwasola Oyinloye")

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    if exported.resolve() != out.resolve():
        exported.unlink()
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB), input 1x3x{imgsz}x{imgsz}", flush=True)


if __name__ == "__main__":
    main()
