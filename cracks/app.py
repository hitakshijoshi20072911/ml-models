"""
DRIFT Crack Detector - Image Inference

Usage examples:
    python app.py --source test.jpg
    python app.py --source test.jpg --conf 0.35
    python app.py --source ./test_images
    python app.py --source test.jpg --weights best.pt --device 0

Outputs:
    - Annotated image(s) in runs/inference/
    - JSON detection metadata alongside each image
    - Detection details printed in the terminal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained Ultralytics YOLO model on images."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("crack_detector_model.pt"),
        help="Path to trained .pt weights (default: crack_detector_model.pt)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input image or directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/inference"),
        help="Output directory (default: runs/inference)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU threshold for NMS (default: 0.70)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: 0, 1, cpu, or auto (default: auto)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detection metadata as JSON files.",
    )
    return parser.parse_args()


def choose_device(requested: str | None) -> str:
    if requested and requested.lower() != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {source.suffix}. "
                f"Supported: {sorted(IMAGE_EXTENSIONS)}"
            )
        yield source
        return

    if source.is_dir():
        images = sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No supported images found in: {source}")
        yield from images
        return

    raise FileNotFoundError(f"Source does not exist: {source}")


def make_detection_record(result) -> list[dict]:
    detections: list[dict] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [float(v) for v in box]
        label = result.names.get(int(cls_id), str(cls_id))

        detections.append(
            {
                "class_id": int(cls_id),
                "label": label,
                "confidence": round(float(conf), 6),
                "bbox_xyxy": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
                "bbox_width": round(x2 - x1, 2),
                "bbox_height": round(y2 - y1, 2),
            }
        )

    return detections


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}\n"
            "Put best.pt in this folder or pass --weights path/to/best.pt"
        )

    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DRIFT - CRACK DETECTION | IMAGE INFERENCE")
    print("=" * 72)
    print(f"Model : {args.weights.resolve()}")
    print(f"Source: {args.source.resolve()}")
    print(f"Device: {device}")
    print(f"Conf  : {args.conf}")
    print(f"Image : {args.imgsz}")
    print()

    model = YOLO(str(args.weights))
    print(f"Classes: {model.names}")
    print()

    for image_path in iter_images(args.source):
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
            save=False,
        )

        result = results[0]
        detections = make_detection_record(result)

        # Ultralytics' plot() returns the annotated BGR image.
        annotated = result.plot(
            conf=True,
            labels=True,
            boxes=True,
        )

        output_path = args.output / image_path.name
        cv2.imwrite(str(output_path), annotated)

        print(f"[IMAGE] {image_path.name}")
        print(f"  Saved -> {output_path}")
        print(f"  Size  -> {result.orig_shape[1]} x {result.orig_shape[0]}")
        print(f"  Speed -> pre={result.speed.get('preprocess', 0):.2f} ms, "
              f"infer={result.speed.get('inference', 0):.2f} ms, "
              f"post={result.speed.get('postprocess', 0):.2f} ms")

        if not detections:
            print("  Detections: NONE")
        else:
            print(f"  Detections: {len(detections)}")
            for i, det in enumerate(detections, start=1):
                print(
                    f"    {i}. {det['label']} | "
                    f"conf={det['confidence']:.3f} | "
                    f"bbox={det['bbox_xyxy']}"
                )

        if args.save_json:
            json_path = output_path.with_suffix(".json")
            payload = {
                "source": str(image_path.resolve()),
                "output": str(output_path.resolve()),
                "model": str(args.weights.resolve()),
                "confidence_threshold": args.conf,
                "iou_threshold": args.iou,
                "image_size": {
                    "width": int(result.orig_shape[1]),
                    "height": int(result.orig_shape[0]),
                },
                "detections": detections,
                "speed_ms": result.speed,
            }
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  JSON  -> {json_path}")

        print("-" * 72)


if __name__ == "__main__":
    main()
