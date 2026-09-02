from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
from inference_sdk import InferenceHTTPClient


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "corrosion-yolov8/4"
API_URL = "https://serverless.roboflow.com"

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


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="DRIFT Rust / Corrosion Detection using Roboflow"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Image, image folder, video, webcam index, or stream URL",
    )

    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key. Prefer ROBOFLOW_API_KEY environment variable.",
    )

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Roboflow model ID. Default: {MODEL_ID}",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output folder. Default: outputs",
    )

    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=1,
        help="Infer every Nth frame for video/live. Default: 1",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live/video preview window.",
    )

    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Do not save annotated video.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for temporary Roboflow failures. Default: 3",
    )

    return parser.parse_args()


# ============================================================
# API KEY
# ============================================================

def get_api_key(cli_key):

    api_key = cli_key or os.getenv("ROBOFLOW_API_KEY")

    if not api_key:

        raise RuntimeError(
            "\nRoboflow API key not found.\n\n"
            "Set it in PowerShell:\n\n"
            '$env:ROBOFLOW_API_KEY="YOUR_REAL_KEY"\n'
        )

    return api_key.strip()


# ============================================================
# CLIENT
# ============================================================

def create_client(api_key):

    return InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key,
    )


# ============================================================
# SOURCE TYPE
# ============================================================

def determine_source_type(source):

    path = Path(source)

    if path.exists():

        if path.is_dir():
            return "image_folder"

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return "image"

        return "video"

    try:

        int(source)
        return "camera"

    except ValueError:

        return "stream"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def create_output_dirs(base):

    base = Path(base)

    directories = {
        "images": base / "images",
        "videos": base / "videos",
        "json": base / "json",
        "logs": base / "logs",
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


# ============================================================
# ROBoflow RESPONSE
# ============================================================

def extract_predictions(
    result,
    confidence_threshold,
):

    if isinstance(result, dict):

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

    detections = []

    for prediction in predictions:

        if hasattr(
            prediction,
            "dict",
        ):

            prediction = prediction.dict()

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
                0,
            )
        )

        if confidence < confidence_threshold:
            continue

        detections.append(
            {
                "label": prediction.get(
                    "class",
                    prediction.get(
                        "label",
                        "unknown",
                    ),
                ),

                "class_id": prediction.get(
                    "class_id",
                    -1,
                ),

                "confidence": round(
                    confidence,
                    6,
                ),

                "x": float(
                    prediction.get(
                        "x",
                        0,
                    )
                ),

                "y": float(
                    prediction.get(
                        "y",
                        0,
                    )
                ),

                "width": float(
                    prediction.get(
                        "width",
                        0,
                    )
                ),

                "height": float(
                    prediction.get(
                        "height",
                        0,
                    )
                ),
            }
        )

    return detections


# ============================================================
# BBOX
# ============================================================

def get_bbox(
    detection,
    frame_width,
    frame_height,
):

    x = detection["x"]
    y = detection["y"]

    width = detection["width"]
    height = detection["height"]

    x1 = int(
        x - width / 2
    )

    y1 = int(
        y - height / 2
    )

    x2 = int(
        x + width / 2
    )

    y2 = int(
        y + height / 2
    )

    x1 = max(
        0,
        x1,
    )

    y1 = max(
        0,
        y1,
    )

    x2 = min(
        frame_width - 1,
        x2,
    )

    y2 = min(
        frame_height - 1,
        y2,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# DRAW RESULTS
# ============================================================

def draw_detections(
    frame,
    detections,
    confidence_threshold,
):

    output = frame.copy()

    height, width = output.shape[:2]

    for detection in detections:

        x1, y1, x2, y2 = get_bbox(
            detection,
            width,
            height,
        )

        label = detection["label"]

        confidence = detection[
            "confidence"
        ]

        text = (
            f"{label} "
            f"{confidence:.2f}"
        )

        # Bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Text dimensions
        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        label_y = max(
            y1,
            text_height + baseline + 5,
        )

        # Label background
        cv2.rectangle(
            output,
            (
                x1,
                label_y
                - text_height
                - baseline
                - 5,
            ),
            (
                x1
                + text_width
                + 8,
                label_y,
            ),
            (0, 255, 0),
            -1,
        )

        # Label text
        cv2.putText(
            output,
            text,
            (
                x1 + 4,
                label_y
                - baseline
                - 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Status box

    cv2.rectangle(
        output,
        (10, 10),
        (460, 80),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        "DRIFT | RUST / CORROSION",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            f"Detections: "
            f"{len(detections)} | "
            f"Conf >= "
            f"{confidence_threshold:.2f}"
        ),
        (20, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# ROBoflow INFERENCE WITH RETRY
# ============================================================

def run_inference(
    client,
    image,
    model_id,
    retries,
):

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            # IMPORTANT:
            # Current Roboflow SDK supports NumPy
            # images directly.

            return client.infer(
                image,
                model_id=model_id,
            )

        except Exception as error:

            last_error = error

            print(
                f"\nRoboflow request failed "
                f"(attempt {attempt}/{retries}):"
            )

            print(error)

            if attempt < retries:

                wait_time = (
                    2 ** (attempt - 1)
                )

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

    raise RuntimeError(
        "\nRoboflow inference failed "
        "after all retries.\n\n"
        f"Last error:\n{last_error}\n\n"
        "If this is HTTP 502/503/504, "
        "the Roboflow hosted inference "
        "service is returning a gateway/server "
        "error rather than rejecting your image."
    )


# ============================================================
# SINGLE IMAGE
# ============================================================

def process_image(
    client,
    image_path,
    output_dirs,
    args,
):

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:

        raise RuntimeError(
            f"Could not read image: "
            f"{image_path}"
        )

    start = time.perf_counter()

    result = run_inference(
        client,
        frame,
        args.model_id,
        args.retries,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    detections = extract_predictions(
        result,
        args.conf,
    )

    annotated = draw_detections(
        frame,
        detections,
        args.conf,
    )

    output_image = (
        output_dirs["images"]
        / f"{image_path.stem}_rust.jpg"
    )

    output_json = (
        output_dirs["json"]
        / f"{image_path.stem}.json"
    )

    cv2.imwrite(
        str(output_image),
        annotated,
    )

    payload = {

        "source": str(
            image_path.resolve()
        ),

        "model_id": args.model_id,

        "confidence_threshold": args.conf,

        "image_width": int(
            frame.shape[1]
        ),

        "image_height": int(
            frame.shape[0]
        ),

        "detection_count": len(
            detections
        ),

        "detections": detections,

        "api_latency_ms": round(
            elapsed,
            2,
        ),
    }

    output_json.write_text(
        json.dumps(
            payload,
            indent=4,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("IMAGE RESULT")
    print("=" * 70)

    print(
        f"Input     : {image_path}"
    )

    print(
        f"Output    : {output_image}"
    )

    print(
        f"JSON      : {output_json}"
    )

    print(
        f"API time  : {elapsed:.0f} ms"
    )

    print(
        f"Detections: {len(detections)}"
    )

    if not detections:

        print(
            "No corrosion detected."
        )

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        x1, y1, x2, y2 = get_bbox(
            detection,
            frame.shape[1],
            frame.shape[0],
        )

        print(
            f"\n{index}. "
            f"{detection['label']}"
        )

        print(
            f"   Confidence : "
            f"{detection['confidence']:.3f}"
        )

        print(
            f"   BoundingBox: "
            f"[{x1}, {y1}, {x2}, {y2}]"
        )


# ============================================================
# IMAGE FOLDER
# ============================================================

def process_image_folder(
    client,
    folder,
    output_dirs,
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
            f"No supported images "
            f"found in {folder}"
        )

    print(
        f"Found {len(images)} images."
    )

    for image in images:

        try:

            process_image(
                client,
                image,
                output_dirs,
                args,
            )

        except Exception as error:

            print(
                f"\n[ERROR] "
                f"{image}"
            )

            print(error)


# ============================================================
# VIDEO / CAMERA / STREAM
# ============================================================

def process_video(
    client,
    source,
    output_dirs,
    args,
):

    # Camera index or URL/file

    try:

        camera_index = int(
            source
        )

        capture = cv2.VideoCapture(
            camera_index
        )

    except ValueError:

        capture = cv2.VideoCapture(
            source
        )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open source: "
            f"{source}"
        )

    source_path = Path(
        source
    )

    if source_path.exists():

        base_name = (
            source_path.stem
        )

    else:

        base_name = "live_stream"

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

    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    writer = None

    video_output = None

    if not args.no_save_video:

        video_output = (
            output_dirs["videos"]
            / f"{base_name}_rust.mp4"
        )

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
                "Warning: video writer "
                "could not be opened."
            )

            writer.release()

            writer = None

    # --------------------------------------------------------
    # LOG FILES
    # --------------------------------------------------------

    jsonl_path = (
        output_dirs["logs"]
        / f"{base_name}_detections.jsonl"
    )

    csv_path = (
        output_dirs["logs"]
        / f"{base_name}_detections.csv"
    )

    csv_file = csv_path.open(
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
            "label",
            "class_id",
            "confidence",
            "x",
            "y",
            "width",
            "height",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ]
    )

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    frame_number = 0

    processed_frames = 0

    last_detections = []

    start_time = time.perf_counter()

    try:

        while True:

            success, frame = (
                capture.read()
            )

            if not success:

                break

            frame_number += 1

            should_infer = (
                (
                    frame_number - 1
                )
                % args.every_nth_frame
                == 0
            )

            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            if should_infer:

                try:

                    result = run_inference(
                        client,
                        frame,
                        args.model_id,
                        args.retries,
                    )

                    detections = extract_predictions(
                        result,
                        args.conf,
                    )

                    last_detections = (
                        detections
                    )

                    processed_frames += 1

                    timestamp = (
                        frame_number
                        / fps
                    )

                    # JSONL
                    frame_record = {

                        "frame":
                            frame_number,

                        "timestamp_sec":
                            round(
                                timestamp,
                                3,
                            ),

                        "detection_count":
                            len(
                                detections
                            ),

                        "detections":
                            detections,
                    }

                    with jsonl_path.open(
                        "a",
                        encoding="utf-8",
                    ) as log:

                        log.write(
                            json.dumps(
                                frame_record
                            )
                            + "\n"
                        )

                    # CSV
                    for detection in detections:

                        x1, y1, x2, y2 = (
                            get_bbox(
                                detection,
                                width,
                                height,
                            )
                        )

                        csv_writer.writerow(
                            [
                                frame_number,

                                round(
                                    timestamp,
                                    3,
                                ),

                                detection[
                                    "label"
                                ],

                                detection[
                                    "class_id"
                                ],

                                detection[
                                    "confidence"
                                ],

                                detection[
                                    "x"
                                ],

                                detection[
                                    "y"
                                ],

                                detection[
                                    "width"
                                ],

                                detection[
                                    "height"
                                ],

                                x1,
                                y1,
                                x2,
                                y2,
                            ]
                        )

                    csv_file.flush()

                    # ------------------------------------------------
                    # TERMINAL OUTPUT
                    # ------------------------------------------------

                    print(
                        f"\n[FRAME "
                        f"{frame_number:06d}] "
                        f"detections="
                        f"{len(detections)}"
                    )

                    if not detections:

                        print(
                            "    No corrosion"
                        )

                    for index, detection in enumerate(
                        detections,
                        start=1,
                    ):

                        x1, y1, x2, y2 = (
                            get_bbox(
                                detection,
                                width,
                                height,
                            )
                        )

                        print(
                            f"    {index}. "
                            f"{detection['label']} | "
                            f"conf="
                            f"{detection['confidence']:.3f} | "
                            f"bbox="
                            f"[{x1}, {y1}, "
                            f"{x2}, {y2}]"
                        )

                except Exception as error:

                    print(
                        f"\n[FRAME ERROR "
                        f"{frame_number}]"
                    )

                    print(error)

            # ------------------------------------------------
            # ANNOTATED FRAME
            # ------------------------------------------------

            annotated = (
                draw_detections(
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
                    max(
                        100,
                        annotated.shape[0] - 20,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # SAVE VIDEO
            # ------------------------------------------------

            if writer is not None:

                writer.write(
                    annotated
                )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            if args.display:

                cv2.imshow(
                    "DRIFT - Corrosion Detection",
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

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Frames read     : "
        f"{frame_number}"
    )

    print(
        f"Frames processed: "
        f"{processed_frames}"
    )

    print(
        f"Time            : "
        f"{elapsed:.1f} sec"
    )

    if video_output:

        print(
            f"Video output    : "
            f"{video_output}"
        )

    print(
        f"JSONL           : "
        f"{jsonl_path}"
    )

    print(
        f"CSV             : "
        f"{csv_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if not 0 <= args.conf <= 1:

        raise ValueError(
            "--conf must be between 0 and 1"
        )

    if args.every_nth_frame < 1:

        raise ValueError(
            "--every-nth-frame must be >= 1"
        )

    api_key = get_api_key(
        args.api_key
    )

    output_dirs = (
        create_output_dirs(
            args.output_dir
        )
    )

    client = create_client(
        api_key
    )

    source_type = (
        determine_source_type(
            args.source
        )
    )

    print()
    print("=" * 70)
    print("DRIFT - RUST / CORROSION")
    print("=" * 70)

    print(
        f"Model      : "
        f"{args.model_id}"
    )

    print(
        f"Source     : "
        f"{args.source}"
    )

    print(
        f"Type       : "
        f"{source_type}"
    )

    print(
        f"Confidence : "
        f"{args.conf}"
    )

    print(
        f"Output     : "
        f"{Path(args.output_dir).resolve()}"
    )

    if source_type == "image":

        process_image(
            client,
            Path(args.source),
            output_dirs,
            args,
        )

    elif source_type == "image_folder":

        process_image_folder(
            client,
            Path(args.source),
            output_dirs,
            args,
        )

    elif source_type in {
        "video",
        "camera",
        "stream",
    }:

        process_video(
            client,
            args.source,
            output_dirs,
            args,
        )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as error:

        print(
            "\n[FATAL ERROR]"
        )

        print(error)

        sys.exit(1)