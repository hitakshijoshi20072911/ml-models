# DRIFT Railway Fault Detection — Roboflow Module

Local inference client for the Roboflow model:

`railway-track-fault-detection-hrem8/3`

The model is hosted by Roboflow, so you do **not** need to download the dataset or model weights for this API-based setup. An internet connection is required.

## 1. Create the virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Add the Roboflow API key

Create a local file named `.env` in this folder:

```text
ROBOFLOW_API_KEY=YOUR_ROBOFLOW_API_KEY
```

Keep this file private. Do **not** upload it to GitHub.

You can also set the key only for the current PowerShell session:

```powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
```

## 4. Expected folder structure

```text
DRIFT_Roboflow_Rail_Module/
├── .venv/
├── models/              # optional; not required for hosted inference
├── test_images/
├── test_videos/
├── outputs/
├── app_new.py
├── app_live.py
├── requirements.txt
├── README.md
└── .env                 # local secret; never commit
```

Put images in `test_images/` and videos in `test_videos/`.

## 5. Image inference

```powershell
python app_new.py
```

The script calls the Roboflow model and saves annotated images and detection metadata in `outputs/`.

## 6. Video inference

```powershell
python app_live.py
```

The video script sends selected frames to Roboflow and records frame/timestamp/detection information for backend integration.

## Production note

Keep the Roboflow API key on your backend/server. Do **not** expose it in browser-side JavaScript or frontend code.
