# VigilEye - real-time driver fatigue detection

OpenCV + MediaPipe Face Mesh. Calibrates to each driver, tracks eye closure (EAR / PERCLOS),
yawns (MAR) and head nods, builds a 0-100 fatigue score, escalates alerts, logs events to
`events.csv`, and optionally sends an AWS SNS alert.

## Run
    pip install -r requirements.txt
    python main.py

Keys: `q` quit, `c` recalibrate, `l` low-light enhancement (CLAHE).

## Optional AWS alert
See the header of `cloud_alert.py`.