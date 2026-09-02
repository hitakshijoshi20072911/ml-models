# DRIFT Crack Model Inference

Put these files in one folder:

    drift_crack_inference/
    ├── app.py
    ├── app_live.py
    ├── requirements.txt
    ├── best.pt
    ├── test_images/
    └── runs/

The `best.pt` file should be the model produced by the Kaggle run:

`/kaggle/working/runs/detect/Road_Maintenance/crack_detector_v1/weights/best.pt`

## Image inference

Single image:

    python app.py --source test.jpg

Directory of images:

    python app.py --source test_images

Custom threshold:

    python app.py --source test.jpg --conf 0.35

Save JSON detection metadata:

    python app.py --source test.jpg --conf 0.25 --save-json

The output image contains:
- bounding box
- class label
- confidence score

The terminal also prints:
- class ID
- class label
- confidence
- bbox `[x1, y1, x2, y2]`
- bbox width and height
- preprocessing/inference/postprocessing time

## Live inference

Laptop webcam:

    python app_live.py

Second camera:

    python app_live.py --source 1

Video:

    python app_live.py --source road_video.mp4

RTMP/HTTP stream:

    python app_live.py --source "rtmp://YOUR_STREAM_URL"

The live window shows:
- bounding boxes
- labels
- confidence
- detection count
- FPS

Press `Q` or `ESC` to quit.
Press `S` to save the current annotated frame.

## GPU selection

Auto-select CUDA if available:

    python app.py --source test.jpg

Force NVIDIA GPU 0:

    python app.py --source test.jpg --device 0

Force CPU:

    python app.py --source test.jpg --device cpu

Check CUDA:

    python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
