"""
DRIFT Crack Detector - Live Inference

Works with:
    - Webcam:        --source 0
    - Video file:    --source video.mp4
    - RTMP/HTTP URL: --source "rtmp://..."
    - OpenCV camera index: --source 1

Keyboard:
    Q / ESC  -> quit
    S        -> save the current annotated frame

Example:
    python app_live.py
    python app_live.py --source 0
    python app_live.py --source "rtmp://YOUR_STREAM_URL"
    python app_live.py --source road_video.mp4 --conf 0.30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO model on a live/video stream."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("main_crack.pt"),
        help="Path to trained .pt weights (default: main_crack.pt)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help='Camera index, video path, or stream URL (default: "0").',
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
        "--save-dir",
        type=Path,
        default=Path("runs/live"),
        help="Directory for manually saved frames.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Display width.",
    )
    return parser.parse_args()


def choose_device(requested: str | None) -> str:
    if requested and requested.lower() != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def parse_source(source: str):
    # Numeric source -> local camera index.
    try:
        return int(source)
    except ValueError:
        return source


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}\n"
            "Put best.pt in this folder or pass --weights path/to/best.pt"
        )

    device = choose_device(args.device)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DRIFT - CRACK DETECTION | LIVE INFERENCE")
    print("=" * 72)
    print(f"Model : {args.weights.resolve()}")
    print(f"Source: {args.source}")
    print(f"Device: {device}")
    print(f"Conf  : {args.conf}")
    print()
    print("Press Q or ESC to quit. Press S to save a frame.")
    print()

    model = YOLO(str(args.weights))
    print(f"Classes: {model.names}")

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {args.source}\n"
            "For webcam, try --source 0 or --source 1.\n"
            "For RTMP, verify the URL and that OpenCV can open the stream."
        )

    saved_count = 0
    prev_time = time.perf_counter()
    fps_ema = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("\nStream ended or frame could not be read.")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
                save=False,
            )

            result = results[0]
            annotated = result.plot(
                conf=True,
                labels=True,
                boxes=True,
            )

            # Calculate display FPS.
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            fps_ema = instant_fps if fps_ema == 0 else (0.90 * fps_ema + 0.10 * instant_fps)

            detection_count = 0 if result.boxes is None else len(result.boxes)

            # Add a compact DRIFT status panel.
            cv2.rectangle(annotated, (10, 10), (420, 92), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                f"DRIFT | FPS: {fps_ema:.1f}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Detections: {detection_count} | Conf >= {args.conf:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Print detections to terminal when present.
            if result.boxes is not None and len(result.boxes):
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                print(
                    f"\rFrame detections: {len(result.boxes):2d} | "
                    + " | ".join(
                        f"{result.names[int(cls)]}:{float(conf):.2f}"
                        for cls, conf in zip(classes, confs)
                    ),
                    end="",
                    flush=True,
                )

                # Keep the detailed data available in the result object:
                # xyxy = pixel bbox, conf = confidence, cls = class id.

            # Resize only for display, without altering inference resolution.
            h, w = annotated.shape[:2]
            if w > args.window_width:
                scale = args.window_width / w
                annotated = cv2.resize(
                    annotated,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow("DRIFT - Live Crack Detection", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                saved_count += 1
                out = args.save_dir / f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(str(out), annotated)
                print(f"\nSaved frame -> {out}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n\nLive inference stopped.")


if __name__ == "__main__":
    main()
