# DRIFT — Integrated Multi-Model Defect Detection

DRIFT uses four independent computer-vision models to detect infrastructure defects from a **single image, video, webcam, or live stream**.

The four models are:

| Model       | Purpose                       | Deployment       |
| ----------- | ----------------------------- | ---------------- |
| **CRACK**   | Crack detection               | Local YOLO `.pt` |
| **ROAD**    | Road damage detection         | Local YOLO `.pt` |
| **RAILWAY** | Railway/track fault detection | Roboflow API     |
| **RUST**    | Rust / corrosion detection    | Roboflow API     |

All four models inspect the same input and their detections are combined into a **single annotated output** and corresponding **JSON + CSV logs**.

---

## 1. Project Structure

Recommended structure:

```text
C:\ml model\
│
├── main_app1.py
├── requirements.txt
├── .venv\
│
├── cracks\
│   └── main_crack.pt
│
├── road-ml\
│   └── main_road.pt
│
└── outputs\
```

The integrated application expects these local model paths:

```text
C:\ml model\cracks\main_crack.pt
C:\ml model\road-ml\main_road.pt
```

The Railway and Rust models are hosted through Roboflow.

---

## 2. Models

### CRACK

Local YOLO model:

```text
main_crack.pt
```

Used for detecting cracks in infrastructure surfaces.

Typical detections depend on the classes contained inside the trained checkpoint. The application automatically reads the class names from the model.

---

### ROAD

Local YOLO road-damage model:

```text
main_road.pt
```

This is the RDD/road-damage model.

The Road model can optionally use **overlapping 640×640 tiled inference**, which is useful when defects are small compared with the full image.

---

### RAILWAY

Roboflow model:

```text
railway-track-fault-detection-hrem8/3
```

The model is accessed through the Roboflow inference API.

---

### RUST / CORROSION

Roboflow model:

```text
corrosion-yolov8/4
```

The model currently exposes the class:

```text
Corrosion
```

It detects visible corrosion/rust regions using bounding boxes and confidence scores.

---

## 3. Installation

Open PowerShell:

```powershell
cd "C:\ml model"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Required packages:

```text
ultralytics
inference-sdk
opencv-python
numpy
```

---

## 4. Roboflow API Key

Railway and Rust require a Roboflow API key.

Set it in the current PowerShell session:

```powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
```

Verify that the variable exists:

```powershell
if ($env:ROBOFLOW_API_KEY) {
    Write-Host "Roboflow API key loaded"
} else {
    Write-Host "Roboflow API key missing"
}
```

Do not commit the real API key to GitHub.

---

# 5. Basic Image Inference

Run all four models on one image:

```powershell
python main_app1.py --source "image.jpg" --imgsz 640 --conf 0.30
```

For higher-resolution local YOLO inference:

```powershell
python main_app1.py --source "image.jpg" --imgsz 1280 --conf 0.30
```

The minimum supported confidence threshold is:

```text
0.25
```

Example:

```powershell
python main_app1.py --source "image.jpg" --conf 0.25
```

---

# 6. Recommended Road Detection Mode

For road images, use tiled inference:

```powershell
python main_app1.py --source "road.jpg" --imgsz 640 --conf 0.30 --road-tiling
```

This runs the Road model on overlapping image tiles and converts the detections back into the original image's pixel coordinates.

This can improve detection of small road defects but takes longer than standard full-image inference.

---

# 7. Video Inference

The same application supports videos.

Example:

```powershell
python main_app1.py --source "inspection.mp4" --imgsz 640 --conf 0.30
```

With Road tiling:

```powershell
python main_app1.py --source "drone_test.mp4" --imgsz 640 --conf 0.30 --road-tiling
```

Higher resolution:

```powershell
python main_app1.py --source "inspection.mp4" --imgsz 1280 --conf 0.30
```

---

## Frame Sampling

Running all four models on every video frame can be slow, especially because Railway and Rust use cloud inference.

Use:

```powershell
--every-nth-frame
```

For example:

```powershell
python main_app1.py --source "inspection.mp4" --imgsz 640 --conf 0.30 --every-nth-frame 3
```

This means:

```text
Frame 1 → inference
Frame 2 → skip
Frame 3 → skip
Frame 4 → inference
Frame 5 → skip
Frame 6 → skip
...
```

For a faster initial test:

```powershell
python main_app1.py --source "inspection.mp4" --every-nth-frame 5 --display
```

---

# 8. Live Webcam

Use camera `0`:

```powershell
python main_app1.py --source 0 --imgsz 640 --conf 0.30 --every-nth-frame 5 --display
```

Second camera:

```powershell
python main_app1.py --source 1 --imgsz 640 --conf 0.30 --every-nth-frame 5 --display
```

Press:

```text
Q
```

or:

```text
ESC
```

to stop.

---

# 9. Live / Network Stream

The same application can accept a stream URL when OpenCV supports the source.

Example:

```powershell
python main_app1.py --source "rtsp://YOUR_STREAM_URL" --imgsz 640 --conf 0.30 --every-nth-frame 5 --display
```

HTTP/other supported streams can similarly be passed as the `--source`.

---

# 10. Available Options

### Resolution

```text
640
1280
```

Example:

```powershell
--imgsz 1280
```

`1280` can help with smaller defects but requires more computation.

---

### Confidence

Must be at least:

```text
0.25
```

Examples:

```powershell
--conf 0.25
--conf 0.30
--conf 0.50
```

Higher confidence means fewer but more confident detections.

---

### Road tiling

Enable:

```powershell
--road-tiling
```

Useful especially for high-resolution road imagery and small defects.

---

### Frame sampling

```powershell
--every-nth-frame 1
--every-nth-frame 3
--every-nth-frame 5
```

`1` means every frame.

---

### Display

Show the processed video/live stream:

```powershell
--display
```

---

### Disable individual models

Useful for testing/debugging.

Only Road + Crack:

```powershell
python main_app1.py --source "road.jpg" --disable-railway --disable-rust
```

Only Railway + Rust:

```powershell
python main_app1.py --source "image.jpg" --disable-crack --disable-road
```

Only Road:

```powershell
python main_app1.py --source "road.jpg" --disable-crack --disable-railway --disable-rust --road-tiling
```

Only Crack:

```powershell
python main_app1.py --source "crack.jpg" --disable-road --disable-railway --disable-rust
```

These options are useful when verifying whether a particular model is working correctly.

---

# 11. Output

For an image:

```text
outputs/
└── image/
    ├── image_output.jpg
    ├── image.json
    └── image.csv
```

For a video:

```text
outputs/
└── inspection/
    ├── inspection_output.mp4
    ├── inspection.json
    └── inspection.csv
```

---

# 12. Annotated Output

Every model has a different bounding-box color:

| Model   | Box Color |
| ------- | --------- |
| CRACK   | Red       |
| ROAD    | Blue      |
| RAILWAY | Yellow    |
| RUST    | Green     |

Each bounding box contains:

```text
MODEL | LABEL | CONFIDENCE
```

Example:

```text
ROAD | pothole | 0.91
RUST | Corrosion | 0.87
```

The output image remains at the original input resolution.

---

# 13. JSON Output

The JSON records every detection.

Example:

```json
{
    "model": "ROAD",
    "label": "pothole",
    "class_id": 1,
    "confidence": 0.913,
    "bounding_box": {
        "x1": 421,
        "y1": 318,
        "x2": 764,
        "y2": 591
    },
    "pixel_coordinates": {
        "top_left": [421, 318],
        "bottom_right": [764, 591]
    },
    "center": {
        "x": 592.5,
        "y": 454.5
    },
    "width": 343,
    "height": 273
}
```

This makes the output suitable for later integration with the DRIFT backend, database, map, or alerting system.

---

# 14. Video JSON

For video, the JSON contains detections grouped by frame:

```json
{
    "frame": 120,
    "timestamp_sec": 4.0,
    "detection_count": 2,
    "detections": [
        {
            "model": "ROAD",
            "label": "pothole",
            "confidence": 0.91
        },
        {
            "model": "RUST",
            "label": "Corrosion",
            "confidence": 0.84
        }
    ]
}
```

This allows the system to determine:

```text
Which defect?
Which model?
Which frame?
What confidence?
Where in the frame?
```

---

# 15. CSV Output

The CSV contains one row per detection.

Main fields:

```text
frame
timestamp_sec
model
label
class_id
confidence
x1
y1
x2
y2
center_x
center_y
bbox_width
bbox_height
image_width
image_height
```

Example:

```text
120,4.0,ROAD,pothole,1,0.91,421,318,764,591,...
120,4.0,RUST,Corrosion,0,0.84,1020,242,1321,488,...
```

This is useful for analysis, database storage, and future geolocation mapping.

---

# 16. How the Integrated System Works

Every input is sent through all enabled detectors:

```text
                    INPUT
                      │
          ┌───────────┴───────────┐
          │                       │
       IMAGE                    VIDEO
          │                       │
          └───────────┬───────────┘
                      │
                 main_app1.py
                      │
       ┌──────────────┼──────────────┐
       │              │              │
     CRACK           ROAD         RAILWAY
       │              │              │
       │              │              │
       └──────────────┼──────────────┘
                      │
                    RUST
                      │
                      ↓
             Unified detections
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Image        JSON         CSV
       / Video
```

The system does **not** first classify an image as "road", "railway", "bridge", etc.

Instead, every enabled model examines the same frame independently.

Therefore, a model can produce a false positive on an image belonging to another infrastructure type. This is a property of the individual detector, not the integration layer.

---

# 17. Recommended First Test

For the first local test, use a known road/crack image that already worked with the original models:

```powershell
python main_app1.py --source "test.jpg" --imgsz 640 --conf 0.30 --road-tiling --display
```

Check the terminal output:

```text
CRACK: ...
ROAD: ...
RAILWAY: ...
RUST: ...
```

This makes it easy to verify each model independently.

Once the image pipeline works, move to video:

```powershell
python main_app1.py --source "test.mp4" --imgsz 640 --conf 0.30 --every-nth-frame 5 --display
```

---

# 18. Important Notes

### Roboflow dependency

Railway and Rust currently use Roboflow-hosted inference, so they require:

```text
Internet connection
+
ROBOFLOW_API_KEY
```

The Crack and Road models are local and do not require Roboflow.

### Performance

The full pipeline can be slower than a single YOLO model because four detectors are being executed and two require remote API requests.

For initial video/live testing, use:

```text
imgsz = 640
conf = 0.30
every-nth-frame = 3 or 5
```

Then increase to 1280 once the pipeline is verified.

### Coordinate system

All final bounding boxes are stored as:

```text
[x1, y1, x2, y2]
```

in **pixel coordinates relative to the original input frame**.

This allows the detections to be connected later to other DRIFT metadata such as timestamps, GPS coordinates, asset IDs, and map locations.

---

# Quick Commands

### Image

```powershell
python main_app1.py --source "image.jpg" --imgsz 640 --conf 0.30
```

### High-resolution image

```powershell
python main_app1.py --source "image.jpg" --imgsz 1280 --conf 0.30
```

### Road image with tiling

```powershell
python main_app1.py --source "road.jpg" --imgsz 640 --conf 0.30 --road-tiling
```

### Video

```powershell
python main_app1.py --source "video.mp4" --imgsz 640 --conf 0.30 --every-nth-frame 3
```

### Video with display

```powershell
python main_app1.py --source "video.mp4" --imgsz 640 --conf 0.30 --every-nth-frame 3 --display
```

### Webcam

```powershell
python main_app1.py --source 0 --imgsz 640 --conf 0.30 --every-nth-frame 5 --display
```

### Only local models

```powershell
python main_app1.py --source "image.jpg" --disable-railway --disable-rust
```

### Only Roboflow models

```powershell
python main_app1.py --source "image.jpg" --disable-crack --disable-road
```

---

## Current Defect Categories

The system currently has four detection branches:

```text
CRACK
    → cracks detected by the local crack model

ROAD
    → road defects detected by the local RDD model

RAILWAY
    → railway/track faults detected by the Railway Roboflow model

RUST
    → Corrosion / rust detected by the Rust Roboflow model
```

The exact sub-classes under **CRACK** and **ROAD** come from the class definitions embedded in their respective `.pt` checkpoints. The Rust model currently reports **Corrosion** as its class.

The integrated output preserves the originating model, so downstream DRIFT components can distinguish, for example:

```text
ROAD / pothole
CRACK / crack
RAILWAY / track_fault
RUST / Corrosion
```
