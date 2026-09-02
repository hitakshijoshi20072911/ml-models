import os

# ============================================================
# PyTorch 2.6+ compatibility for older Ultralytics checkpoints
# ============================================================
#
# PyTorch 2.6 changed torch.load() default behavior to
# weights_only=True.
#
# This repository uses an older Ultralytics version and a
# trusted full-model .pt checkpoint, so we explicitly restore
# the legacy loading behavior.
#
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR /  "YOLOv8_Small_RDD.pt"
TEST_IMAGE_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_CONFIDENCE = 0.25

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# MODEL
# ============================================================

def load_model():
    """Load the trained YOLOv8 road-damage model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("\n==============================================")
    print("Loading Road Damage Detection Model")
    print("==============================================")
    print(f"Model:   {MODEL_PATH}")
    print(f"Python:  {torch.__version__}")
    print("Device:  CPU")
    print()

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")
    print("\nModel classes:")

    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")

    return model


# ============================================================
# FILE HELPERS
# ============================================================

def get_test_images():
    """Return all supported images in test_images/."""

    TEST_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        [
            path
            for path in TEST_IMAGE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    return images


# ============================================================
# DRAWING
# ============================================================

def draw_detection(
    image,
    x1,
    y1,
    x2,
    y2,
    class_name,
    confidence,
):
    """Draw bounding box + label + confidence."""

    # Bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    label = f"{class_name} {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    # Position the label above the box where possible
    label_y1 = max(
        0,
        y1 - text_height - baseline - 8,
    )

    label_y2 = y1

    # Green label background
    cv2.rectangle(
        image,
        (x1, label_y1),
        (
            x1 + text_width + 8,
            label_y2,
        ),
        (0, 255, 0),
        -1,
    )

    # Black label text
    cv2.putText(
        image,
        label,
        (
            x1 + 4,
            label_y2 - baseline - 4,
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# IOU / NMS FOR TILED DETECTION
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two boxes.

    Boxes are [x1, y1, x2, y2].
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def apply_classwise_nms(
    detections,
    iou_threshold=0.45,
):
    """
    Remove duplicate detections from overlapping tiles.

    Each detection is:
    {
        "class_id": int,
        "class_name": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2]
    }
    """

    final_detections = []

    class_ids = sorted(
        set(
            detection["class_id"]
            for detection in detections
        )
    )

    for class_id in class_ids:

        class_detections = [
            detection
            for detection in detections
            if detection["class_id"] == class_id
        ]

        class_detections.sort(
            key=lambda x: x["confidence"],
            reverse=True,
        )

        while class_detections:

            best = class_detections.pop(0)

            final_detections.append(best)

            remaining = []

            for candidate in class_detections:

                iou = calculate_iou(
                    best["bbox"],
                    candidate["bbox"],
                )

                if iou < iou_threshold:
                    remaining.append(candidate)

            class_detections = remaining

    final_detections.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return final_detections


# ============================================================
# FULL IMAGE INFERENCE
# ============================================================

def run_full_image_inference(
    model,
    image_path,
    confidence,
    image_size,
):
    """
    Run YOLO directly on the complete image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read:\n{image_path}"
        )

    results = model.predict(
        source=str(image_path),
        device="cpu",
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_id = int(
                box.cls[0].item()
            )

            confidence_score = float(
                box.conf[0].item()
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": confidence_score,
                    "bbox": [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ],
                }
            )

    return image, detections


# ============================================================
# TILED INFERENCE
# ============================================================

def run_tiled_inference(
    model,
    image_path,
    confidence,
    tile_size=640,
    overlap=0.20,
):
    """
    Split the full-resolution image into overlapping tiles.

    Each tile is independently passed through YOLO.

    Coordinates are translated back into the original
    full-image coordinate system.

    Duplicate detections from overlapping tiles are removed
    using class-wise NMS.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read:\n{image_path}"
        )

    image_height, image_width = image.shape[:2]

    stride = int(
        tile_size * (1.0 - overlap)
    )

    if stride <= 0:
        raise ValueError(
            "Overlap is too large."
        )

    all_detections = []

    tile_counter = 0

    y_positions = list(
        range(
            0,
            max(1, image_height - tile_size + 1),
            stride,
        )
    )

    x_positions = list(
        range(
            0,
            max(1, image_width - tile_size + 1),
            stride,
        )
    )

    # Ensure final edge regions are covered
    if image_height > tile_size:
        final_y = image_height - tile_size

        if final_y not in y_positions:
            y_positions.append(final_y)

    if image_width > tile_size:
        final_x = image_width - tile_size

        if final_x not in x_positions:
            x_positions.append(final_x)

    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    total_tiles = (
        len(x_positions)
        * len(y_positions)
    )

    print(
        f"\nTiling image:"
        f" {image_width}x{image_height}"
    )

    print(
        f"Tile size: {tile_size}x{tile_size}"
    )

    print(
        f"Overlap: {overlap:.0%}"
    )

    print(
        f"Total tiles: {total_tiles}"
    )

    for y in y_positions:

        for x in x_positions:

            tile_counter += 1

            x_end = min(
                x + tile_size,
                image_width,
            )

            y_end = min(
                y + tile_size,
                image_height,
            )

            tile = image[
                y:y_end,
                x:x_end,
            ]

            results = model.predict(
                source=tile,
                device="cpu",
                conf=confidence,
                imgsz=tile_size,
                verbose=False,
            )

            result = results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:

                local_x1, local_y1, local_x2, local_y2 = (
                    box.xyxy[0].tolist()
                )

                class_id = int(
                    box.cls[0].item()
                )

                confidence_score = float(
                    box.conf[0].item()
                )

                # Convert tile coordinates back
                # to coordinates in the original image.
                global_x1 = int(
                    round(local_x1 + x)
                )

                global_y1 = int(
                    round(local_y1 + y)
                )

                global_x2 = int(
                    round(local_x2 + x)
                )

                global_y2 = int(
                    round(local_y2 + y)
                )

                # Clamp to image dimensions
                global_x1 = max(
                    0,
                    min(global_x1, image_width - 1),
                )

                global_y1 = max(
                    0,
                    min(global_y1, image_height - 1),
                )

                global_x2 = max(
                    0,
                    min(global_x2, image_width - 1),
                )

                global_y2 = max(
                    0,
                    min(global_y2, image_height - 1),
                )

                all_detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": confidence_score,
                        "bbox": [
                            global_x1,
                            global_y1,
                            global_x2,
                            global_y2,
                        ],
                    }
                )

    print(
        f"Raw tiled detections: "
        f"{len(all_detections)}"
    )

    # Remove duplicate boxes generated by overlapping tiles.
    final_detections = apply_classwise_nms(
        all_detections,
        iou_threshold=0.45,
    )

    print(
        f"After duplicate removal: "
        f"{len(final_detections)}"
    )

    return image, final_detections


# ============================================================
# DRAW RESULTS
# ============================================================

def annotate_image(
    image,
    detections,
):
    """Draw all detections on a copy of the image."""

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        draw_detection(
            image=output,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_name=detection["class_name"],
            confidence=detection["confidence"],
        )

    return output


# ============================================================
# PRINT DETECTION DETAILS
# ============================================================

def print_detections(detections):

    print()
    print("==============================================")
    print("DETECTIONS")
    print("==============================================")

    if not detections:

        print(
            "No detections above the selected "
            "confidence threshold."
        )

        return

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        x1, y1, x2, y2 = detection["bbox"]

        print(
            f"{index}. "
            f"{detection['class_name']:<25} "
            f"confidence="
            f"{detection['confidence']:.3f} "
            f"bbox="
            f"({x1}, {y1}, {x2}, {y2})"
        )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    image,
    image_path,
    mode_name,
):
    """Save annotated image into outputs/."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if mode_name == "normal":
        filename = (
            f"{image_path.stem}-output.jpg"
        )

    elif mode_name == "highres":
        filename = (
            f"{image_path.stem}-highres-output.jpg"
        )

    elif mode_name == "tiled":
        filename = (
            f"{image_path.stem}-tiled-output.jpg"
        )

    else:
        filename = (
            f"{image_path.stem}-{mode_name}-output.jpg"
        )

    output_path = OUTPUT_DIR / filename

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise IOError(
            f"Could not save:\n{output_path}"
        )

    return output_path


# ============================================================
# DISPLAY
# ============================================================

def display_result(
    image,
    window_title,
):
    """
    Display image in an OpenCV window.

    Press Q to close the window.
    """

    display_image = image.copy()

    max_width = 1400
    max_height = 900

    height, width = display_image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale < 1.0:

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        display_image = cv2.resize(
            display_image,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    cv2.namedWindow(
        window_title,
        cv2.WINDOW_NORMAL,
    )

    cv2.imshow(
        window_title,
        display_image,
    )

    print()
    print(
        "Result displayed."
    )

    print(
        "Press Q while the image window "
        "is focused to close it."
    )

    while True:

        key = cv2.waitKey(100) & 0xFF

        if key == ord("q"):

            cv2.destroyWindow(
                window_title
            )

            break


# ============================================================
# GET IMAGE FROM TERMINAL
# ============================================================

def choose_image(images):

    print("\n==============================================")
    print("AVAILABLE TEST IMAGES")
    print("==============================================")

    for index, image in enumerate(
        images,
        start=1,
    ):
        print(
            f"  {index}. {image.name}"
        )

    print()
    print("Enter the image NUMBER or FILENAME.")
    print("Enter Q to quit.")
    print()

    selection = input(
        "Select image: "
    ).strip()

    if selection.lower() == "q":
        return None

    # Select by number
    if selection.isdigit():

        index = int(selection)

        if 1 <= index <= len(images):
            return images[index - 1]

        print(
            "\nInvalid image number."
        )

        return "INVALID"

    # Select by filename
    for image in images:

        if (
            image.name.lower()
            == selection.lower()
        ):
            return image

    print(
        "\nImage not found."
    )

    return "INVALID"


# ============================================================
# SELECT INFERENCE MODE
# ============================================================

def choose_mode():

    print("\n==============================================")
    print("INFERENCE MODE")
    print("==============================================")

    print(
        "  1. Normal"
    )

    print(
        "     Full image, imgsz=640"
    )

    print()

    print(
        "  2. High Resolution"
    )

    print(
        "     Full image, imgsz=1280"
    )

    print()

    print(
        "  3. Tiled"
    )

    print(
        "     640x640 overlapping tiles"
    )

    print()

    print(
        "  4. Run ALL THREE"
    )

    print()

    print(
        "  Q. Back"
    )

    choice = input(
        "Select mode: "
    ).strip().lower()

    if choice == "q":
        return None

    if choice == "1":
        return "normal"

    if choice == "2":
        return "highres"

    if choice == "3":
        return "tiled"

    if choice == "4":
        return "all"

    print(
        "\nInvalid mode."
    )

    return "INVALID"


# ============================================================
# RUN A SINGLE MODE
# ============================================================

def execute_mode(
    model,
    image_path,
    mode,
    confidence,
):

    print("\n")
    print("==============================================")
    print(
        f"RUNNING MODE: {mode.upper()}"
    )
    print("==============================================")
    print(
        f"Image: {image_path.name}"
    )

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    if mode == "normal":

        original, detections = (
            run_full_image_inference(
                model=model,
                image_path=image_path,
                confidence=confidence,
                image_size=640,
            )
        )

    # --------------------------------------------------------
    # High resolution
    # --------------------------------------------------------

    elif mode == "highres":

        original, detections = (
            run_full_image_inference(
                model=model,
                image_path=image_path,
                confidence=confidence,
                image_size=1280,
            )
        )

    # --------------------------------------------------------
    # Tiled
    # --------------------------------------------------------

    elif mode == "tiled":

        original, detections = (
            run_tiled_inference(
                model=model,
                image_path=image_path,
                confidence=confidence,
                tile_size=640,
                overlap=0.20,
            )
        )

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    # --------------------------------------------------------
    # Print detections
    # --------------------------------------------------------

    print_detections(
        detections
    )

    # --------------------------------------------------------
    # Annotate
    # --------------------------------------------------------

    annotated = annotate_image(
        original,
        detections,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_output(
        image=annotated,
        image_path=image_path,
        mode_name=mode,
    )

    print()
    print(
        f"Saved output:\n{output_path}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_result(
        annotated,
        (
            f"Road Damage Detection - "
            f"{image_path.name} - "
            f"{mode}"
        ),
    )

    return detections


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("====================================================")
    print("       ROAD DAMAGE DETECTION - YOLOv8")
    print("====================================================")
    print()
    print("Terminal inference testing application")
    print("CPU mode")
    print()

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            "ERROR: Model file does not exist:"
        )

        print(
            MODEL_PATH
        )

        return

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    TEST_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = load_model()

    except Exception as error:

        print()
        print(
            "=============================================="
        )

        print(
            "MODEL LOADING FAILED"
        )

        print(
            "=============================================="
        )

        print(
            repr(error)
        )

        return

    # --------------------------------------------------------
    # Confidence selection
    # --------------------------------------------------------

    print()
    print("Default confidence threshold:")
    print(
        f"  {DEFAULT_CONFIDENCE:.2f}"
    )

    confidence_input = input(
        "\nEnter confidence threshold "
        "(press ENTER for 0.25): "
    ).strip()

    if confidence_input == "":

        confidence = DEFAULT_CONFIDENCE

    else:

        try:

            confidence = float(
                confidence_input
            )

        except ValueError:

            print(
                "Invalid confidence. "
                "Using 0.25."
            )

            confidence = DEFAULT_CONFIDENCE

    if not 0.01 <= confidence <= 0.99:

        print(
            "Confidence must be between "
            "0.01 and 0.99."
        )

        print(
            "Using 0.25 instead."
        )

        confidence = DEFAULT_CONFIDENCE

    print(
        f"\nUsing confidence threshold: "
        f"{confidence:.2f}"
    )

    # ========================================================
    # MAIN MENU
    # ========================================================

    while True:

        images = get_test_images()

        if not images:

            print()
            print(
                "No test images found."
            )

            print(
                f"Put images in:"
            )

            print(
                f"  {TEST_IMAGE_DIR}"
            )

            return

        image_path = choose_image(
            images
        )

        if image_path is None:

            print(
                "\nExiting..."
            )

            cv2.destroyAllWindows()

            return

        if image_path == "INVALID":

            continue

        # ----------------------------------------------------
        # Choose mode
        # ----------------------------------------------------

        while True:

            mode = choose_mode()

            if mode is None:
                break

            if mode == "INVALID":
                continue

            break

        if mode is None:
            continue

        # ----------------------------------------------------
        # Run all
        # ----------------------------------------------------

        if mode == "all":

            modes = [
                "normal",
                "highres",
                "tiled",
            ]

            for current_mode in modes:

                try:

                    execute_mode(
                        model=model,
                        image_path=image_path,
                        mode=current_mode,
                        confidence=confidence,
                    )

                except Exception as error:

                    print()
                    print(
                        f"ERROR in "
                        f"{current_mode} mode:"
                    )

                    print(
                        repr(error)
                    )

            print()
            print(
                "=============================================="
            )

            print(
                "ALL THREE MODES COMPLETE"
            )

            print(
                "=============================================="
            )

            continue

        # ----------------------------------------------------
        # Run selected mode
        # ----------------------------------------------------

        try:

            execute_mode(
                model=model,
                image_path=image_path,
                mode=mode,
                confidence=confidence,
            )

        except Exception as error:

            print()
            print(
                "=============================================="
            )

            print(
                "INFERENCE FAILED"
            )

            print(
                "=============================================="
            )

            print(
                repr(error)
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()