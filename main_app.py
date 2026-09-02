"""
DRIFT - INTEGRATED MULTI-MODEL DEFECT DETECTION

Models:
    1. CRACKS  -> local Ultralytics YOLO .pt
    2. ROAD    -> local Ultralytics YOLO .pt
    3. RAILWAY -> Roboflow hosted model
    4. RUST    -> Roboflow hosted model

Supported inputs:
    - JPG / JPEG / PNG / BMP / WEBP / TIFF
    - MP4 / AVI / MOV / MKV / WEBM / M4V
    - Webcam index: 0, 1, ...
    - Network stream URL supported by OpenCV

Outputs:

For image:
    outputs/
        image1/
            image1_output.jpg
            image1.json
            image1.csv

For video:
    outputs/
        video1/
            video1_output.mp4
            video1.json
            video1.csv

For live:
    outputs/
        live_YYYYMMDD_HHMMSS/
            live_YYYYMMDD_HHMMSS_output.mp4
            live_YYYYMMDD_HHMMSS.json
            live_YYYYMMDD_HHMMSS.csv

Example:

    python main_app.py --source image.jpg --imgsz 640 --conf 0.30

    python main_app.py --source image.jpg --imgsz 1280 --conf 0.35

    python main_app.py --source video.mp4 --imgsz 640 --conf 0.30

    python main_app.py --source video.mp4 --imgsz 1280 --conf 0.35

    python main_app.py --source 0 --imgsz 640 --conf 0.30 --display

    python main_app.py --source video.mp4 --every-nth-frame 3

Important:
    - Confidence must be >= 0.25.
    - Output coordinates are in ORIGINAL FRAME PIXEL COORDINATES.
    - Each detection contains the originating model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration,
)


# =====================================================================
# MODEL PATHS
# =====================================================================

# IMPORTANT:
# These are the paths you provided.
# Change them here if your actual folder names differ.

CRACK_MODEL_PATH = Path(
    r"C:\ml models\cracks\main_crack.pt"
)

ROAD_MODEL_PATH = Path(
    r"C:\ml models\road-ml\main_road.pt"
)


# =====================================================================
# ROBOFLOW MODELS
# =====================================================================

RAILWAY_MODEL_ID = (
    "railway-track-fault-detection-hrem8/3"
)

RUST_MODEL_ID = (
    "corrosion-yolov8/4"
)

ROBOFLOW_API_URL = (
    "https://serverless.roboflow.com"
)


# =====================================================================
# COLORS
# OpenCV uses BGR
# =====================================================================

MODEL_COLORS = {

    "CRACK": (
        0,
        0,
        255,
    ),      # Red

    "ROAD": (
        255,
        0,
        0,
    ),      # Blue

    "RAILWAY": (
        0,
        255,
        255,
    ),      # Yellow

    "RUST": (
        0,
        255,
        0,
    ),      # Green
}


# =====================================================================
# EXTENSIONS
# =====================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "DRIFT integrated crack + road + railway + rust "
            "multi-model inference"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Image, video, webcam index or stream URL"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        choices=[
            640,
            1280,
        ],
        default=640,
        help=(
            "Local YOLO inference resolution: "
            "640 or 1280"
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help=(
            "Minimum confidence. Must be >= 0.25. "
            "Default: 0.30"
        ),
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help=(
            "IoU threshold for local YOLO NMS. "
            "Default: 0.70"
        ),
    )

    parser.add_argument(
        "--device",
        default="0",
        help=(
            "Local YOLO device. "
            "Example: 0 or cpu"
        ),
    )

    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=1,
        help=(
            "Video/live: process every Nth frame. "
            "Default: 1"
        ),
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Display live/video output window"
        ),
    )

    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help=(
            "Do not save annotated video"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help=(
            "Output root directory. "
            "Default: outputs"
        ),
    )

    parser.add_argument(
        "--disable-crack",
        action="store_true",
        help="Disable crack model",
    )

    parser.add_argument(
        "--disable-road",
        action="store_true",
        help="Disable road model",
    )

    parser.add_argument(
        "--disable-railway",
        action="store_true",
        help="Disable railway model",
    )

    parser.add_argument(
        "--disable-rust",
        action="store_true",
        help="Disable rust model",
    )

    return parser.parse_args()


# =====================================================================
# API KEY
# =====================================================================

def get_roboflow_api_key():

    key = os.getenv(
        "ROBOFLOW_API_KEY"
    )

    if not key:

        raise RuntimeError(
            "\n"
            "ROBOFLOW_API_KEY was not found.\n\n"
            "Set it in PowerShell:\n\n"
            '$env:ROBOFLOW_API_KEY="YOUR_KEY_HERE"\n'
        )

    return key.strip()


# =====================================================================
# ROBOFLOW CLIENT
# =====================================================================

def create_roboflow_client():

    api_key = (
        get_roboflow_api_key()
    )

    return InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    ).configure(
        InferenceConfiguration(
            api_key_transport="header"
        )
    )


# =====================================================================
# LOAD LOCAL MODELS
# =====================================================================

def load_local_models(args):

    models = {}

    if not args.disable_crack:

        if not CRACK_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Crack model not found:\n"
                f"{CRACK_MODEL_PATH}"
            )

        print(
            f"[LOAD] Crack model: "
            f"{CRACK_MODEL_PATH}"
        )

        models["CRACK"] = YOLO(
            str(CRACK_MODEL_PATH)
        )

    if not args.disable_road:

        if not ROAD_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Road model not found:\n"
                f"{ROAD_MODEL_PATH}"
            )

        print(
            f"[LOAD] Road model: "
            f"{ROAD_MODEL_PATH}"
        )

        models["ROAD"] = YOLO(
            str(ROAD_MODEL_PATH)
        )

    return models


# =====================================================================
# SOURCE TYPE
# =====================================================================

def determine_source_type(
    source: str,
):

    path = Path(source)

    if path.exists():

        if path.is_dir():

            return "image_folder"

        if path.suffix.lower() in IMAGE_EXTENSIONS:

            return "image"

        if path.suffix.lower() in VIDEO_EXTENSIONS:

            return "video"

        # Unknown file:
        # let OpenCV attempt to interpret it.
        return "video"

    try:

        int(source)

        return "camera"

    except ValueError:

        return "stream"


# =====================================================================
# OUTPUT DIRECTORIES
# =====================================================================

def create_output_directory(
    root: Path,
    source_type: str,
    source: str,
):

    if source_type in {
        "image",
        "video",
    }:

        source_path = Path(
            source
        )

        timestamp_name = (
            source_path.stem
        )

    else:

        timestamp_name = (
            "live_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

    output_dir = (
        root / timestamp_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# =====================================================================
# NORMALIZE LOCAL YOLO RESULT
# =====================================================================

def infer_local_model(
    model_name,
    model,
    frame,
    args,
):

    start = time.perf_counter()

    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    result = results[0]

    detections = []

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return {
            "model": model_name,
            "detections": [],
            "latency_ms": elapsed,
        }

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    class_ids = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    for (
        bbox,
        confidence,
        class_id,
    ) in zip(
        boxes,
        confidences,
        class_ids,
    ):

        x1, y1, x2, y2 = (
            [
                float(value)
                for value in bbox
            ]
        )

        class_name = (
            result.names.get(
                int(class_id),
                str(class_id),
            )
        )

        detections.append(
            {
                "model": model_name,

                "label": str(
                    class_name
                ),

                "class_id": int(
                    class_id
                ),

                "confidence": round(
                    float(confidence),
                    6,
                ),

                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
            }
        )

    return {
        "model": model_name,
        "detections": detections,
        "latency_ms": elapsed,
    }


# =====================================================================
# NORMALIZE ROBOFLOW
# =====================================================================

def parse_roboflow_result(
    result,
    model_name,
    confidence_threshold,
):

    if isinstance(
        result,
        dict,
    ):

        predictions = result.get(
            "predictions",
            [],
        )

    else:

        predictions = getattr(
            result,
            "predictions",
            [],
        )

    if not predictions:

        return []

    detections = []

    for prediction in predictions:

        if hasattr(
            prediction,
            "dict",
        ):

            prediction = (
                prediction.dict()
            )

        elif hasattr(
            prediction,
            "__dict__",
        ):

            prediction = vars(
                prediction
            )

        if not isinstance(
            prediction,
            dict,
        ):

            continue

        confidence = float(
            prediction.get(
                "confidence",
                0.0,
            )
        )

        if (
            confidence
            < confidence_threshold
        ):

            continue

        label = (
            prediction.get(
                "class"
            )
            or prediction.get(
                "class_name"
            )
            or prediction.get(
                "label"
            )
            or "unknown"
        )

        # -------------------------------------------------------------
        # Standard Roboflow format:
        #
        # x, y = center
        # width, height = box dimensions
        # -------------------------------------------------------------

        if all(
            key in prediction
            for key in (
                "x",
                "y",
                "width",
                "height",
            )
        ):

            x = float(
                prediction["x"]
            )

            y = float(
                prediction["y"]
            )

            width = float(
                prediction["width"]
            )

            height = float(
                prediction["height"]
            )

            x1 = (
                x
                - width / 2
            )

            y1 = (
                y
                - height / 2
            )

            x2 = (
                x
                + width / 2
            )

            y2 = (
                y
                + height / 2
            )

        elif all(
            key in prediction
            for key in (
                "x1",
                "y1",
                "x2",
                "y2",
            )
        ):

            x1 = float(
                prediction["x1"]
            )

            y1 = float(
                prediction["y1"]
            )

            x2 = float(
                prediction["x2"]
            )

            y2 = float(
                prediction["y2"]
            )

        else:

            continue

        detections.append(
            {
                "model": model_name,

                "label": str(
                    label
                ),

                "class_id": int(
                    prediction.get(
                        "class_id",
                        -1,
                    )
                ),

                "confidence": round(
                    confidence,
                    6,
                ),

                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
            }
        )

    return detections


# =====================================================================
# ROBOFLOW INFERENCE
# =====================================================================

def infer_roboflow_model(
    client,
    frame,
    model_id,
    model_name,
    confidence_threshold,
):

    start = time.perf_counter()

    result = client.infer(
        frame,
        model_id=model_id,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    detections = (
        parse_roboflow_result(
            result,
            model_name,
            confidence_threshold,
        )
    )

    return {
        "model": model_name,
        "detections": detections,
        "latency_ms": elapsed,
    }


# =====================================================================
# RUN ALL MODELS CONCURRENTLY
# =====================================================================

def infer_all_models(
    frame,
    local_models,
    roboflow_client,
    args,
):

    tasks = []

    # -------------------------------------------------------------
    # Local models
    # -------------------------------------------------------------

    for model_name, model in (
        local_models.items()
    ):

        tasks.append(
            (
                model_name,
                lambda model=model,
                model_name=model_name:
                    infer_local_model(
                        model_name,
                        model,
                        frame,
                        args,
                    ),
            )
        )

    # -------------------------------------------------------------
    # Roboflow Railway
    # -------------------------------------------------------------

    if not args.disable_railway:

        tasks.append(
            (
                "RAILWAY",
                lambda:
                    infer_roboflow_model(
                        roboflow_client,
                        frame,
                        RAILWAY_MODEL_ID,
                        "RAILWAY",
                        args.conf,
                    ),
            )
        )

    # -------------------------------------------------------------
    # Roboflow Rust
    # -------------------------------------------------------------

    if not args.disable_rust:

        tasks.append(
            (
                "RUST",
                lambda:
                    infer_roboflow_model(
                        roboflow_client,
                        frame,
                        RUST_MODEL_ID,
                        "RUST",
                        args.conf,
                    ),
            )
        )

    if not tasks:

        return []

    results = []

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # All four model calls are dispatched concurrently.
    #
    # This reduces wall-clock latency compared with:
    #
    # crack -> road -> railway -> rust
    #
    # However, local GPU execution may still internally serialize
    # depending on GPU resources.
    # -------------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=len(tasks)
    ) as executor:

        futures = [
            executor.submit(
                function
            )
            for (
                _name,
                function,
            ) in tasks
        ]

        for future in futures:

            try:

                result = future.result()

                results.append(
                    result
                )

            except Exception as error:

                results.append(
                    {
                        "model": "ERROR",

                        "detections": [],

                        "latency_ms": 0,

                        "error": str(
                            error
                        ),
                    }
                )

    # -------------------------------------------------------------
    # Flatten
    # -------------------------------------------------------------

    all_detections = []

    for result in results:

        all_detections.extend(
            result.get(
                "detections",
                [],
            )
        )

    return all_detections


# =====================================================================
# CLAMP BBOX
# =====================================================================

def clamp_bbox(
    bbox,
    frame_width,
    frame_height,
):

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(
            int(round(x1)),
            frame_width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            int(round(y1)),
            frame_height - 1,
        ),
    )

    x2 = max(
        0,
        min(
            int(round(x2)),
            frame_width - 1,
        ),
    )

    y2 = max(
        0,
        min(
            int(round(y2)),
            frame_height - 1,
        ),
    )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# =====================================================================
# DRAW ONE DETECTION
# =====================================================================

def draw_detection(
    frame,
    detection,
):

    height, width = (
        frame.shape[:2]
    )

    bbox = clamp_bbox(
        detection["bbox"],
        width,
        height,
    )

    detection["bbox"] = bbox

    x1, y1, x2, y2 = bbox

    model_name = detection[
        "model"
    ]

    label = detection[
        "label"
    ]

    confidence = detection[
        "confidence"
    ]

    color = MODEL_COLORS.get(
        model_name,
        (
            255,
            255,
            255,
        ),
    )

    # -------------------------------------------------------------
    # Bounding box
    # -------------------------------------------------------------

    cv2.rectangle(
        frame,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        color,
        3,
    )

    # -------------------------------------------------------------
    # Label
    # -------------------------------------------------------------

    text = (
        f"{model_name} | "
        f"{label} | "
        f"{confidence:.2f}"
    )

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.62

    thickness = 2

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    label_top = max(
        0,
        y1 - text_height - baseline - 8,
    )

    label_bottom = (
        y1
        if y1 > text_height
        else text_height + baseline + 8
    )

    cv2.rectangle(
        frame,
        (
            x1,
            label_top,
        ),
        (
            min(
                width - 1,
                x1 + text_width + 10,
            ),
            label_bottom,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x1 + 5,
            label_bottom - baseline - 3,
        ),
        font,
        font_scale,
        (
            0,
            0,
            0,
        ),
        thickness,
        cv2.LINE_AA,
    )


# =====================================================================
# DRAW ALL DETECTIONS
# =====================================================================

def draw_all_detections(
    frame,
    detections,
    confidence_threshold,
):

    output = frame.copy()

    for detection in detections:

        draw_detection(
            output,
            detection,
        )

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    counts = {
        "CRACK": 0,
        "ROAD": 0,
        "RAILWAY": 0,
        "RUST": 0,
    }

    for detection in detections:

        model_name = detection[
            "model"
        ]

        if model_name in counts:

            counts[
                model_name
            ] += 1

    header_height = 125

    cv2.rectangle(
        output,
        (
            10,
            10,
        ),
        (
            610,
            header_height,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    cv2.putText(
        output,
        "DRIFT | MULTI-MODEL DEFECT DETECTION",
        (
            20,
            38,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            f"Cracks: {counts['CRACK']} | "
            f"Road: {counts['ROAD']} | "
            f"Rail: {counts['RAILWAY']} | "
            f"Rust: {counts['RUST']}"
        ),
        (
            20,
            67,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            f"Total: {len(detections)} | "
            f"Confidence >= "
            f"{confidence_threshold:.2f}"
        ),
        (
            20,
            96,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return output


# =====================================================================
# NORMALIZE FOR JSON
# =====================================================================

def prepare_detection_for_output(
    detection,
):

    x1, y1, x2, y2 = (
        detection["bbox"]
    )

    return {
        "model": detection[
            "model"
        ],

        "label": detection[
            "label"
        ],

        "class_id": detection.get(
            "class_id",
            -1,
        ),

        "confidence": round(
            float(
                detection[
                    "confidence"
                ]
            ),
            6,
        ),

        "bounding_box": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        },

        "pixel_coordinates": {
            "top_left": [
                int(x1),
                int(y1),
            ],

            "bottom_right": [
                int(x2),
                int(y2),
            ],
        },

        "center": {
            "x": round(
                (
                    x1 + x2
                ) / 2,
                2,
            ),

            "y": round(
                (
                    y1 + y2
                ) / 2,
                2,
            ),
        },

        "width": int(
            x2 - x1
        ),

        "height": int(
            y2 - y1
        ),
    }


# =====================================================================
# SAVE IMAGE OUTPUT
# =====================================================================

def save_image_outputs(
    output_dir,
    source_path,
    frame,
    detections,
    args,
):

    base_name = (
        Path(source_path).stem
    )

    output_image = (
        output_dir
        / f"{base_name}_output.jpg"
    )

    output_json = (
        output_dir
        / f"{base_name}.json"
    )

    output_csv = (
        output_dir
        / f"{base_name}.csv"
    )

    annotated = (
        draw_all_detections(
            frame,
            detections,
            args.conf,
        )
    )

    # -------------------------------------------------------------
    # Image
    # -------------------------------------------------------------

    cv2.imwrite(
        str(output_image),
        annotated,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            95,
        ],
    )

    # -------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------

    h, w = frame.shape[:2]

    json_data = {
        "source": str(
            Path(
                source_path
            ).resolve()
        ),

        "input_type": "image",

        "image": {
            "width": int(w),
            "height": int(h),
        },

        "configuration": {
            "confidence_threshold":
                args.conf,

            "iou_threshold":
                args.iou,

            "local_imgsz":
                args.imgsz,
        },

        "detection_count": len(
            detections
        ),

        "detections": [
            prepare_detection_for_output(
                detection
            )
            for detection in detections
        ],
    }

    output_json.write_text(
        json.dumps(
            json_data,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "frame",
                "timestamp_sec",
                "model",
                "label",
                "class_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "bbox_width",
                "bbox_height",
                "image_width",
                "image_height",
            ]
        )

        for detection in detections:

            (
                x1,
                y1,
                x2,
                y2,
            ) = detection["bbox"]

            writer.writerow(
                [
                    1,
                    0.0,
                    detection[
                        "model"
                    ],
                    detection[
                        "label"
                    ],
                    detection.get(
                        "class_id",
                        -1,
                    ),
                    detection[
                        "confidence"
                    ],
                    x1,
                    y1,
                    x2,
                    y2,
                    round(
                        (
                            x1 + x2
                        ) / 2,
                        2,
                    ),
                    round(
                        (
                            y1 + y2
                        ) / 2,
                        2,
                    ),
                    x2 - x1,
                    y2 - y1,
                    w,
                    h,
                ]
            )

    print()
    print("=" * 75)
    print("IMAGE COMPLETE")
    print("=" * 75)

    print(
        f"Annotated image : "
        f"{output_image}"
    )

    print(
        f"JSON            : "
        f"{output_json}"
    )

    print(
        f"CSV             : "
        f"{output_csv}"
    )

    print(
        f"Detections      : "
        f"{len(detections)}"
    )


# =====================================================================
# PROCESS ONE IMAGE
# =====================================================================

def process_image(
    source_path,
    output_dir,
    local_models,
    roboflow_client,
    args,
):

    frame = cv2.imread(
        str(source_path)
    )

    if frame is None:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{source_path}"
        )

    print()
    print(
        "=" * 75
    )

    print(
        f"IMAGE: {source_path.name}"
    )

    print(
        f"Resolution: "
        f"{frame.shape[1]}x"
        f"{frame.shape[0]}"
    )

    print(
        "=" * 75
    )

    start = time.perf_counter()

    detections = infer_all_models(
        frame,
        local_models,
        roboflow_client,
        args,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    print(
        f"\nTotal integrated inference: "
        f"{elapsed:.2f} ms"
    )

    for detection in detections:

        print(
            f"  [{detection['model']}] "
            f"{detection['label']} "
            f"| conf="
            f"{detection['confidence']:.3f} "
            f"| bbox="
            f"{detection['bbox']}"
        )

    save_image_outputs(
        output_dir,
        source_path,
        frame,
        detections,
        args,
    )


# =====================================================================
# PROCESS VIDEO
# =====================================================================

def process_video(
    source,
    output_dir,
    local_models,
    roboflow_client,
    args,
):

    capture = None

    try:

        source_index = int(
            source
        )

        capture = cv2.VideoCapture(
            source_index
        )

    except ValueError:

        capture = cv2.VideoCapture(
            source
        )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video/stream:\n"
            f"{source}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 20.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # -------------------------------------------------------------
    # Video output
    # -------------------------------------------------------------

    writer = None

    video_output = (
        output_dir
        / f"{Path(source).stem if Path(source).exists() else 'live'}_output.mp4"
    )

    if not args.no_save_video:

        writer = cv2.VideoWriter(
            str(video_output),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():

            print(
                "[WARNING] "
                "Could not create output video."
            )

            writer.release()
            writer = None

    # -------------------------------------------------------------
    # JSON / CSV
    # -------------------------------------------------------------

    json_output = (
        output_dir
        / f"{Path(source).stem if Path(source).exists() else 'live'}.json"
    )

    csv_output = (
        output_dir
        / f"{Path(source).stem if Path(source).exists() else 'live'}.csv"
    )

    # In-memory frame records.
    #
    # This keeps ONE JSON file for the whole video.
    # For very long videos, consider JSONL instead.

    video_json = {

        "source": str(
            Path(source).resolve()
            if Path(source).exists()
            else source
        ),

        "input_type": (
            "video"
            if Path(source).exists()
            else "live_stream"
        ),

        "video": {
            "width": width,
            "height": height,
            "fps": fps,
        },

        "configuration": {
            "confidence_threshold":
                args.conf,

            "iou_threshold":
                args.iou,

            "local_imgsz":
                args.imgsz,

            "every_nth_frame":
                args.every_nth_frame,
        },

        "frames": [],
    }

    csv_file = csv_output.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow(
        [
            "frame",
            "timestamp_sec",
            "model",
            "label",
            "class_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "bbox_width",
            "bbox_height",
            "image_width",
            "image_height",
        ]
    )

    frame_number = 0

    processed_frames = 0

    total_inference_time = 0.0

    last_detections = []

    print()
    print(
        "=" * 75
    )
    print(
        "VIDEO / LIVE INFERENCE"
    )
    print(
        f"Resolution: {width}x{height}"
    )
    print(
        f"FPS: {fps:.2f}"
    )
    print(
        f"Every Nth frame: "
        f"{args.every_nth_frame}"
    )
    print(
        "=" * 75
    )

    try:

        while True:

            success, frame = (
                capture.read()
            )

            if not success:

                break

            frame_number += 1

            should_process = (
                (
                    frame_number - 1
                )
                % args.every_nth_frame
                == 0
            )

            if should_process:

                inference_start = (
                    time.perf_counter()
                )

                try:

                    detections = (
                        infer_all_models(
                            frame,
                            local_models,
                            roboflow_client,
                            args,
                        )
                    )

                    last_detections = (
                        detections
                    )

                    processed_frames += 1

                    elapsed = (
                        time.perf_counter()
                        - inference_start
                    ) * 1000

                    total_inference_time += (
                        elapsed
                    )

                    timestamp_sec = (
                        frame_number / fps
                    )

                    # ---------------------------------------------
                    # Console
                    # ---------------------------------------------

                    print()
                    print(
                        f"[FRAME "
                        f"{frame_number:06d}] "
                        f"time="
                        f"{timestamp_sec:.2f}s "
                        f"| detections="
                        f"{len(detections)} "
                        f"| inference="
                        f"{elapsed:.0f}ms"
                    )

                    for detection in (
                        detections
                    ):

                        print(
                            f"    "
                            f"[{detection['model']}] "
                            f"{detection['label']} "
                            f"| conf="
                            f"{detection['confidence']:.3f} "
                            f"| bbox="
                            f"{detection['bbox']}"
                        )

                    # ---------------------------------------------
                    # JSON
                    # ---------------------------------------------

                    frame_record = {

                        "frame":
                            frame_number,

                        "timestamp_sec":
                            round(
                                timestamp_sec,
                                3,
                            ),

                        "detection_count":
                            len(detections),

                        "detections": [
                            prepare_detection_for_output(
                                detection
                            )
                            for detection
                            in detections
                        ],
                    }

                    video_json[
                        "frames"
                    ].append(
                        frame_record
                    )

                    # ---------------------------------------------
                    # CSV
                    # ---------------------------------------------

                    for detection in (
                        detections
                    ):

                        (
                            x1,
                            y1,
                            x2,
                            y2,
                        ) = detection[
                            "bbox"
                        ]

                        csv_writer.writerow(
                            [
                                frame_number,

                                round(
                                    timestamp_sec,
                                    3,
                                ),

                                detection[
                                    "model"
                                ],

                                detection[
                                    "label"
                                ],

                                detection.get(
                                    "class_id",
                                    -1,
                                ),

                                detection[
                                    "confidence"
                                ],

                                x1,
                                y1,
                                x2,
                                y2,

                                round(
                                    (
                                        x1 + x2
                                    ) / 2,
                                    2,
                                ),

                                round(
                                    (
                                        y1 + y2
                                    ) / 2,
                                    2,
                                ),

                                x2 - x1,
                                y2 - y1,

                                width,
                                height,
                            ]
                        )

                    csv_file.flush()

                except Exception as error:

                    print(
                        f"\n[FRAME ERROR] "
                        f"{frame_number}"
                    )

                    print(error)

                    last_detections = []

            # -----------------------------------------------------
            # DRAW
            # -----------------------------------------------------

            annotated = (
                draw_all_detections(
                    frame,
                    last_detections,
                    args.conf,
                )
            )

            cv2.putText(
                annotated,
                (
                    f"Frame: "
                    f"{frame_number} | "
                    f"Processed: "
                    f"{processed_frames}"
                ),
                (
                    20,
                    height - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            # -----------------------------------------------------
            # SAVE
            # -----------------------------------------------------

            if writer is not None:

                writer.write(
                    annotated
                )

            # -----------------------------------------------------
            # DISPLAY
            # -----------------------------------------------------

            if args.display:

                cv2.imshow(
                    "DRIFT - Integrated Detection",
                    annotated,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):

                    break

    finally:

        capture.release()

        if writer is not None:

            writer.release()

        csv_file.close()

        cv2.destroyAllWindows()

    # -------------------------------------------------------------
    # Write JSON
    # -------------------------------------------------------------

    video_json[
        "summary"
    ] = {

        "frames_read":
            frame_number,

        "frames_processed":
            processed_frames,

        "total_detection_records":
            sum(
                len(
                    frame[
                        "detections"
                    ]
                )
                for frame
                in video_json[
                    "frames"
                ]
            ),

        "average_inference_ms":
            (
                (
                    total_inference_time
                    / processed_frames
                )
                if processed_frames
                else 0.0
            ),
    }

    json_output.write_text(
        json.dumps(
            video_json,
            indent=4,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 75
    )
    print(
        "VIDEO PROCESSING COMPLETE"
    )
    print(
        "=" * 75
    )

    print(
        f"Frames read        : "
        f"{frame_number}"
    )

    print(
        f"Frames processed   : "
        f"{processed_frames}"
    )

    print(
        f"Avg inference time : "
        f"{video_json['summary']['average_inference_ms']:.2f} ms"
    )

    if writer is not None:

        print(
            f"Video output       : "
            f"{video_output}"
        )

    print(
        f"JSON               : "
        f"{json_output}"
    )

    print(
        f"CSV                : "
        f"{csv_output}"
    )


# =====================================================================
# IMAGE FOLDER
# =====================================================================

def process_image_folder(
    folder,
    output_root,
    local_models,
    roboflow_client,
    args,
):

    images = sorted(
        [
            path
            for path in folder.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    if not images:

        raise RuntimeError(
            f"No images found in:\n"
            f"{folder}"
        )

    print(
        f"\nFound {len(images)} images."
    )

    for image_path in images:

        image_output_dir = (
            output_root
            / image_path.stem
        )

        image_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        process_image(
            image_path,
            image_output_dir,
            local_models,
            roboflow_client,
            args,
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    if args.conf < 0.25:

        raise ValueError(
            "Confidence must be >= 0.25"
        )

    if args.every_nth_frame < 1:

        raise ValueError(
            "--every-nth-frame must be >= 1"
        )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    output_root = Path(
        args.output_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Source
    # -------------------------------------------------------------

    source_type = (
        determine_source_type(
            args.source
        )
    )

    print()
    print(
        "=" * 75
    )
    print(
        "DRIFT - INTEGRATED MULTI-MODEL SYSTEM"
    )
    print(
        "=" * 75
    )

    print(
        f"Source     : "
        f"{args.source}"
    )

    print(
        f"Source type: "
        f"{source_type}"
    )

    print(
        f"YOLO size  : "
        f"{args.imgsz}"
    )

    print(
        f"Confidence : "
        f"{args.conf}"
    )

    print(
        f"IoU        : "
        f"{args.iou}"
    )

    print(
        "=" * 75
    )

    # -------------------------------------------------------------
    # Load local models
    # -------------------------------------------------------------

    local_models = load_local_models(
        args
    )

    # -------------------------------------------------------------
    # Roboflow
    # -------------------------------------------------------------

    needs_roboflow = (
        not args.disable_railway
        or not args.disable_rust
    )

    roboflow_client = None

    if needs_roboflow:

        print(
            "\n[LOAD] Initializing Roboflow..."
        )

        roboflow_client = (
            create_roboflow_client()
        )

        print(
            "[LOAD] Roboflow ready."
        )

    # -------------------------------------------------------------
    # Output location
    # -------------------------------------------------------------

    if source_type == "image_folder":

        source_output_dir = (
            output_root
        )

    else:

        source_output_dir = (
            create_output_directory(
                output_root,
                source_type,
                args.source,
            )
        )

    # -------------------------------------------------------------
    # Process
    # -------------------------------------------------------------

    if source_type == "image":

        process_image(
            Path(args.source),
            source_output_dir,
            local_models,
            roboflow_client,
            args,
        )

    elif source_type == "image_folder":

        process_image_folder(
            Path(args.source),
            source_output_dir,
            local_models,
            roboflow_client,
            args,
        )

    elif source_type in {
        "video",
        "camera",
        "stream",
    }:

        process_video(
            args.source,
            source_output_dir,
            local_models,
            roboflow_client,
            args,
        )

    else:

        raise RuntimeError(
            "Unsupported source type."
        )

    print()
    print(
        "=" * 75
    )
    print(
        "DRIFT INFERENCE FINISHED"
    )
    print(
        "=" * 75
    )

    print(
        f"Output directory:"
        f"\n{source_output_dir.resolve()}"
    )


# =====================================================================
# ENTRY
# =====================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

        sys.exit(0)

    except Exception as error:

        print()
        print(
            "=" * 75
        )
        print(
            "FATAL ERROR"
        )
        print(
            "=" * 75
        )

        print(
            error
        )

        sys.exit(1)