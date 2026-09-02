from pathlib import Path
import csv
import os
import sys
import time

import cv2
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_ID = "railway-track-fault-detection-hrem8/3"
API_URL = "https://serverless.roboflow.com"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# ENVIRONMENT
# ============================================================

def get_api_key():

    api_key = os.getenv(
        "ROBOFLOW_API_KEY"
    )

    if not api_key:

        print()
        print("=" * 70)
        print("❌ ROBOFLOW API KEY NOT FOUND")
        print("=" * 70)

        print(
            "\nSet your API key in the current PowerShell session:"
        )

        print(
            '\n$env:ROBOFLOW_API_KEY="YOUR_KEY_HERE"'
        )

        print(
            "\nThen run app_new.py again."
        )

        return None

    return api_key


# ============================================================
# ROBoflow CLIENT
# ============================================================

def create_client():

    api_key = get_api_key()

    if api_key is None:
        return None

    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key,
    ).configure(
        InferenceConfiguration(
            api_key_transport="header"
        )
    )

    return client


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_images():

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        [
            path
            for path in IMAGE_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )


# ============================================================
# NORMALIZE ROBoflow RESPONSE
# ============================================================

def extract_predictions(result):

    """
    Roboflow responses may contain predictions under
    different structures depending on the inference API/model.

    This function extracts the common object-detection format.
    """

    if not isinstance(
        result,
        dict,
    ):
        return []

    predictions = result.get(
        "predictions",
        [],
    )

    if predictions is None:
        return []

    if not isinstance(
        predictions,
        list,
    ):
        return []

    normalized = []

    for prediction in predictions:

        if not isinstance(
            prediction,
            dict,
        ):
            continue

        class_name = (
            prediction.get(
                "class"
            )
            or prediction.get(
                "class_name"
            )
            or "unknown"
        )

        confidence = prediction.get(
            "confidence",
            0.0,
        )

        try:
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        # ----------------------------------------------------
        # Roboflow commonly returns center-x, center-y,
        # width, height.
        # ----------------------------------------------------

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

            w = float(
                prediction["width"]
            )

            h = float(
                prediction["height"]
            )

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

        # ----------------------------------------------------
        # Also support explicit coordinates if returned.
        # ----------------------------------------------------

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

        normalized.append(
            {
                "class": class_name,
                "confidence": confidence,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

    return normalized


# ============================================================
# DRAW
# ============================================================

def draw_detection(
    image,
    detection,
    index,
):

    x1, y1, x2, y2 = (
        detection["bbox"]
    )

    height, width = (
        image.shape[:2]
    )

    x1 = max(
        0,
        min(
            int(round(x1)),
            width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            int(round(y1)),
            height - 1,
        ),
    )

    x2 = max(
        0,
        min(
            int(round(x2)),
            width - 1,
        ),
    )

    y2 = max(
        0,
        min(
            int(round(y2)),
            height - 1,
        ),
    )

    detection["bbox"] = [
        x1,
        y1,
        x2,
        y2,
    ]

    label = (
        f"{detection['class']} "
        f"{detection['confidence']:.2f}"
    )

    # Bounding box
    cv2.rectangle(
        image,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (0, 255, 0),
        3,
    )

    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    label_top = max(
        0,
        y1 - text_height - baseline - 8,
    )

    label_bottom = y1

    cv2.rectangle(
        image,
        (
            x1,
            label_top,
        ),
        (
            x1 + text_width + 10,
            label_bottom,
        ),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        image,
        label,
        (
            x1 + 5,
            label_bottom - baseline - 4,
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    csv_path,
    image_path,
    detections,
    image_width,
    image_height,
):

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "image",
                "model",
                "class",
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

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            writer.writerow(
                [
                    image_path.name,
                    MODEL_ID,
                    detection["class"],
                    f"{detection['confidence']:.6f}",
                    x1,
                    y1,
                    x2,
                    y2,
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1,
                    image_width,
                    image_height,
                ]
            )


# ============================================================
# RUN ONE IMAGE
# ============================================================

def run_image(
    client,
    image_path,
    confidence_threshold,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read:\n"
            f"{image_path}"
        )

    image_height, image_width = (
        image.shape[:2]
    )

    print()
    print("-" * 70)
    print(
        f"IMAGE: {image_path.name}"
    )
    print("-" * 70)

    print(
        f"Resolution: "
        f"{image_width}x{image_height}"
    )

    print(
        f"Model: "
        f"{MODEL_ID}"
    )

    print(
        f"Confidence filter: "
        f"{confidence_threshold:.2f}"
    )

    start = time.perf_counter()

    result = client.infer(
        str(image_path),
        model_id=MODEL_ID,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    detections = extract_predictions(
        result
    )

    # Apply local confidence filtering
    detections = [
        detection
        for detection in detections
        if detection["confidence"]
        >= confidence_threshold
    ]

    print(
        f"\nAPI response time: "
        f"{elapsed * 1000:.2f} ms"
    )

    print(
        f"Detections: "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    annotated = image.copy()

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        draw_detection(
            annotated,
            detection,
            index,
        )

    # --------------------------------------------------------
    # Print detections
    # --------------------------------------------------------

    if detections:

        print(
            "\nDetected objects:"
        )

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            print(
                f"  {index}. "
                f"{detection['class']} "
                f"| {detection['confidence']:.3f} "
                f"| bbox=("
                f"{x1}, "
                f"{y1}, "
                f"{x2}, "
                f"{y2})"
            )

    else:

        print(
            "\nNo detections above threshold."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image = (
        OUTPUT_DIR
        / f"{image_path.stem}"
        f"-roboflow-output.jpg"
    )

    output_csv = (
        OUTPUT_DIR
        / f"{image_path.stem}"
        f"-roboflow-detections.csv"
    )

    cv2.imwrite(
        str(output_image),
        annotated,
    )

    save_csv(
        csv_path=output_csv,
        image_path=image_path,
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )

    print(
        f"\nSaved image:"
        f"\n  {output_image}"
    )

    print(
        f"\nSaved CSV:"
        f"\n  {output_csv}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    window_title = (
        f"Roboflow Rail Fault - "
        f"{image_path.name}"
    )

    cv2.namedWindow(
        window_title,
        cv2.WINDOW_NORMAL,
    )

    cv2.imshow(
        window_title,
        annotated,
    )

    print(
        "\nPress Q in the image window "
        "to close it."
    )

    while True:

        key = (
            cv2.waitKey(100)
            & 0xFF
        )

        if key == ord("q"):

            cv2.destroyWindow(
                window_title
            )

            break


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DRIFT — ROBOFLOW RAIL FAULT DETECTOR")
    print("=" * 70)

    print(
        f"\nModel:"
        f"\n  {MODEL_ID}"
    )

    print(
        "\nThis test uses Roboflow's hosted trained model."
    )

    print(
        "No Roboflow dataset is downloaded."
    )

    client = create_client()

    if client is None:
        return

    images = get_images()

    if not images:

        print(
            "\n❌ No test images found."
        )

        print(
            f"\nPut images here:"
            f"\n  {IMAGE_DIR}"
        )

        return

    print(
        "\nAvailable images:"
    )

    for index, image in enumerate(
        images,
        start=1,
    ):

        print(
            f"  {index}. {image.name}"
        )

    print(
        "\nA = test all images"
    )

    choice = input(
        "\nSelect image: "
    ).strip().lower()

    if choice == "a":

        selected_images = images

    elif choice.isdigit():

        index = int(
            choice
        )

        if not (
            1 <= index <= len(images)
        ):

            print(
                "Invalid selection."
            )

            return

        selected_images = [
            images[index - 1]
        ]

    else:

        print(
            "Invalid selection."
        )

        return

    threshold_input = input(
        "\nMinimum confidence "
        "(ENTER = 0.25): "
    ).strip()

    if threshold_input:

        try:

            confidence_threshold = float(
                threshold_input
            )

        except ValueError:

            confidence_threshold = 0.25

    else:

        confidence_threshold = 0.25

    confidence_threshold = max(
        0.0,
        min(
            confidence_threshold,
            1.0,
        ),
    )

    for image_path in selected_images:

        try:

            run_image(
                client=client,
                image_path=image_path,
                confidence_threshold=(
                    confidence_threshold
                ),
            )

        except Exception as error:

            print()
            print(
                "❌ INFERENCE FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    print()
    print(
        "=" * 70
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()


# $env:ROBOFLOW_API_KEY="dz9tSV6EfU9NG10blKMa"