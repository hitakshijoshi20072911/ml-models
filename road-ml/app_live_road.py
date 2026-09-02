import os

# ============================================================
# PyTorch 2.6+ compatibility for older Ultralytics checkpoints
# ============================================================
#
# Needed for your original YOLOv8 RDD checkpoint if the
# environment contains PyTorch 2.6+.
#
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "YOLOv8_Small_RDD.pt"
)

VIDEO_DIR = (
    BASE_DIR
    / "test_videos"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
}


# ============================================================
# MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}\n"
        )

    print()
    print("=" * 70)
    print("LOADING ROAD DAMAGE MODEL")
    print("=" * 70)

    print(
        f"Model:\n  {MODEL_PATH}"
    )

    print(
        f"\nPyTorch:\n  {torch.__version__}"
    )

    print(
        "\nCUDA available:"
    )

    print(
        f"  {torch.cuda.is_available()}"
    )

    print(
        "\nDevice:"
    )

    print(
        "  CPU"
    )

    print(
        "\nLoading model..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "\n✅ Model loaded successfully."
    )

    print(
        "\nClasses:"
    )

    for class_id, class_name in model.names.items():

        print(
            f"  {class_id}: {class_name}"
        )

    return model


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def get_videos():

    VIDEO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    videos = sorted(
        [
            path
            for path in VIDEO_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_VIDEO_EXTENSIONS
            )
        ]
    )

    return videos


# ============================================================
# DRAWING
# ============================================================

def draw_detection(
    frame,
    x1,
    y1,
    x2,
    y2,
    class_name,
    confidence,
):
    """
    Draw bounding box + class + confidence.
    """

    # Bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    label = (
        f"{class_name} "
        f"{confidence:.2f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.6

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

    # Label background
    cv2.rectangle(
        frame,
        (x1, label_top),
        (
            x1 + text_width + 8,
            label_bottom,
        ),
        (0, 255, 0),
        -1,
    )

    # Label text
    cv2.putText(
        frame,
        label,
        (
            x1 + 4,
            label_bottom - baseline - 4,
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    model,
    frame,
    confidence,
    image_size,
):
    """
    Run YOLO inference on one frame.
    """

    results = model.predict(
        source=frame,
        device="cpu",
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    result = results[0]

    annotated = frame.copy()

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            class_id = int(
                box.cls[0].item()
            )

            confidence_score = float(
                box.conf[0].item()
            )

            class_name = (
                result.names[class_id]
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence_score,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

            draw_detection(
                frame=annotated,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                class_name=class_name,
                confidence=confidence_score,
            )

    return annotated, detections


# ============================================================
# RESIZE DISPLAY ONLY
# ============================================================

def resize_for_display(
    frame,
    max_width=1400,
    max_height=900,
):
    """
    Resize only the preview window.

    The saved video retains the original resolution.
    """

    height, width = frame.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale >= 1.0:
        return frame

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        frame,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# SELECT VIDEO
# ============================================================

def choose_video(videos):

    print()
    print("=" * 70)
    print("AVAILABLE TEST VIDEOS")
    print("=" * 70)

    for i, video in enumerate(
        videos,
        start=1,
    ):

        print(
            f"  {i}. {video.name}"
        )

    print()
    print(
        "Enter video number or filename."
    )

    print(
        "Enter Q to quit."
    )

    print()

    selection = input(
        "Select video: "
    ).strip()

    if selection.lower() == "q":
        return None

    if selection.isdigit():

        index = int(
            selection
        )

        if 1 <= index <= len(videos):

            return videos[index - 1]

        print(
            "\nInvalid video number."
        )

        return "INVALID"

    for video in videos:

        if (
            video.name.lower()
            == selection.lower()
        ):

            return video

    print(
        "\nVideo not found."
    )

    return "INVALID"


# ============================================================
# RUN VIDEO INFERENCE
# ============================================================

def run_video(
    model,
    video_path,
    confidence,
    image_size,
    process_every_n_frames,
):
    """
    Run YOLO inference frame-by-frame on a video.

    Press Q in the OpenCV window to stop.
    """

    print()
    print("=" * 70)
    print("VIDEO INFERENCE")
    print("=" * 70)

    print(
        f"Input:\n  {video_path}"
    )

    print(
        f"\nConfidence:\n  {confidence:.2f}"
    )

    print(
        f"\nYOLO image size:\n  {image_size}"
    )

    print(
        f"\nProcess every N frames:\n"
        f"  {process_every_n_frames}"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    # --------------------------------------------------------
    # Video metadata
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print()
    print("Video information:")
    print(
        f"  Resolution: "
        f"{width}x{height}"
    )

    print(
        f"  FPS: "
        f"{fps:.2f}"
    )

    print(
        f"  Frames: "
        f"{total_frames}"
    )

    print(
        f"  Duration: "
        f"{duration:.2f} seconds"
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{video_path.stem}"
        f"-output.mp4"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video.\n"
            f"Attempted:\n{output_path}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    frame_number = 0

    processed_frames = 0

    total_detections = 0

    max_detections = 0

    class_counts = {}

    start_time = time.time()

    inference_time_total = 0.0

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    window_title = (
        "DRIFT Video Inference"
    )

    cv2.namedWindow(
        window_title,
        cv2.WINDOW_NORMAL,
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Process selected frames
        # ----------------------------------------------------

        should_process = (
            frame_number
            % process_every_n_frames
            == 0
        )

        if should_process:

            inference_start = (
                time.perf_counter()
            )

            try:

                annotated, detections = (
                    process_frame(
                        model=model,
                        frame=frame,
                        confidence=confidence,
                        image_size=image_size,
                    )
                )

            except Exception:

                cap.release()
                writer.release()
                cv2.destroyAllWindows()

                raise

            inference_end = (
                time.perf_counter()
            )

            inference_time = (
                inference_end
                - inference_start
            )

            inference_time_total += (
                inference_time
            )

            processed_frames += 1

            detection_count = len(
                detections
            )

            total_detections += (
                detection_count
            )

            max_detections = max(
                max_detections,
                detection_count,
            )

            # Count classes
            for detection in detections:

                class_name = (
                    detection["class_name"]
                )

                class_counts[class_name] = (
                    class_counts.get(
                        class_name,
                        0,
                    )
                    + 1
                )

        else:

            # If this frame isn't being
            # processed, show it unchanged.
            annotated = frame.copy()

        # ----------------------------------------------------
        # Add status overlay
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        processing_fps = (
            processed_frames
            / elapsed
            if elapsed > 0
            else 0
        )

        overlay_text = (
            f"Frame: {frame_number}"
            f"/{total_frames}"
            f" | Inference FPS: "
            f"{processing_fps:.2f}"
            f" | Q = quit"
        )

        cv2.rectangle(
            annotated,
            (0, 0),
            (
                min(
                    width,
                    650,
                ),
                35,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            overlay_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Save every frame
        # ----------------------------------------------------

        writer.write(
            annotated
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display = resize_for_display(
            annotated
        )

        cv2.imshow(
            window_title,
            display,
        )

        # ----------------------------------------------------
        # Quit with Q
        # ----------------------------------------------------

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            print(
                "\nQ pressed."
            )

            break

        # ----------------------------------------------------
        # Terminal progress
        # ----------------------------------------------------

        if (
            frame_number % 30 == 0
            or frame_number == 1
        ):

            percent = (
                frame_number
                / total_frames
                * 100
                if total_frames > 0
                else 0
            )

            print(
                f"\rProgress: "
                f"{percent:6.2f}% | "
                f"Frame "
                f"{frame_number}/"
                f"{total_frames} | "
                f"Processed: "
                f"{processed_frames}",
                end="",
                flush=True,
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    average_inference_ms = (
        (
            inference_time_total
            / processed_frames
        )
        * 1000
        if processed_frames > 0
        else 0
    )

    actual_processing_fps = (
        processed_frames
        / total_time
        if total_time > 0
        else 0
    )

    print()
    print()
    print("=" * 70)
    print("VIDEO INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"\nFrames read:"
        f"\n  {frame_number}"
    )

    print(
        f"\nFrames processed:"
        f"\n  {processed_frames}"
    )

    print(
        f"\nTotal runtime:"
        f"\n  {total_time:.2f} seconds"
    )

    print(
        f"\nAverage inference time:"
        f"\n  {average_inference_ms:.2f} ms/frame"
    )

    print(
        f"\nProcessing FPS:"
        f"\n  {actual_processing_fps:.2f}"
    )

    print(
        f"\nTotal detections:"
        f"\n  {total_detections}"
    )

    print(
        f"\nMaximum detections in one frame:"
        f"\n  {max_detections}"
    )

    print()

    if class_counts:

        print(
            "Detected classes:"
        )

        for class_name, count in sorted(
            class_counts.items()
        ):

            print(
                f"  {class_name}: "
                f"{count}"
            )

    else:

        print(
            "No detections were recorded."
        )

    print()
    print(
        f"Output video:"
    )

    print(
        f"  {output_path}"
    )

    print(
        "\n" + "=" * 70
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       DRIFT - VIDEO ROAD DAMAGE INFERENCE")
    print("=" * 70)

    print()
    print(
        "CPU-based video inference test"
    )

    print(
        "Press Q in the video window to stop."
    )

    print()

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            "❌ Model does not exist:"
        )

        print(
            MODEL_PATH
        )

        return

    # --------------------------------------------------------
    # Get videos
    # --------------------------------------------------------

    videos = get_videos()

    if not videos:

        print(
            "❌ No test videos found."
        )

        print()
        print(
            "Create this folder:"
        )

        print(
            VIDEO_DIR
        )

        print()
        print(
            "Then put an MP4/AVI/MOV video inside."
        )

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = load_model()

    except Exception as error:

        print()
        print(
            "❌ MODEL LOADING FAILED"
        )

        print()
        print(
            repr(error)
        )

        return

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    print()
    print(
        f"Default confidence: "
        f"{DEFAULT_CONFIDENCE}"
    )

    confidence_input = input(
        "Confidence "
        "(ENTER = 0.25): "
    ).strip()

    if confidence_input:

        try:

            confidence = float(
                confidence_input
            )

        except ValueError:

            print(
                "Invalid value. "
                "Using 0.25."
            )

            confidence = (
                DEFAULT_CONFIDENCE
            )

    else:

        confidence = (
            DEFAULT_CONFIDENCE
        )

    if not (
        0.01
        <= confidence
        <= 0.99
    ):

        print(
            "Confidence must be "
            "between 0.01 and 0.99."
        )

        print(
            "Using 0.25."
        )

        confidence = (
            DEFAULT_CONFIDENCE
        )

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------

    print()
    print(
        "YOLO inference size:"
    )

    print(
        "1. 640"
    )

    print(
        "2. 1280"
    )

    size_choice = input(
        "Choose "
        "(ENTER = 640): "
    ).strip()

    if size_choice == "2":

        image_size = 1280

    else:

        image_size = 640

    # --------------------------------------------------------
    # Frame processing
    # --------------------------------------------------------

    print()
    print(
        "Process every N frames:"
    )

    print(
        "1. Every frame"
    )

    print(
        "2. Every 2nd frame"
    )

    print(
        "3. Every 5th frame"
    )

    print(
        "4. Every 10th frame"
    )

    frame_choice = input(
        "Choose "
        "(ENTER = every frame): "
    ).strip()

    frame_mapping = {
        "1": 1,
        "2": 2,
        "3": 5,
        "4": 10,
    }

    process_every_n_frames = (
        frame_mapping.get(
            frame_choice,
            1,
        )
    )

    # --------------------------------------------------------
    # Choose video
    # --------------------------------------------------------

    video_path = choose_video(
        videos
    )

    if video_path is None:

        print(
            "\nExiting..."
        )

        return

    if video_path == "INVALID":

        return

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    try:

        run_video(
            model=model,
            video_path=video_path,
            confidence=confidence,
            image_size=image_size,
            process_every_n_frames=(
                process_every_n_frames
            ),
        )

    except Exception as error:

        print()
        print(
            "=" * 70
        )

        print(
            "❌ VIDEO INFERENCE FAILED"
        )

        print(
            "=" * 70
        )

        print()
        print(
            repr(error)
        )

    print()
    print(
        "Program finished."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()