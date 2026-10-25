"""VigilEye web front end.  Run:  python server.py   then open http://127.0.0.1:5000

Reuses the detection code in Main.py (keep both files in the same folder).
pip install flask
"""
import csv
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory

import Main as vg

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))


class Engine:
    def __init__(self):
        self.running = False
        self.thread = None
        self.frame_jpg = None
        self.enhance = False
        self.recal = False
        self.alerter = None
        self.status = {"running": False, "error": None}

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.status = {"running": True, "error": None, "calibrated": False, "calib_left": vg.CALIB_SECONDS}
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        backend = cap = stats = None
        try:
            backend = vg.create_backend()
            cap = vg.open_camera()
            stats = vg.SessionStats()
            det = vg.DrowsinessDetector(stats)
            self.alerter = alerter = vg.Alerter(stats)
            fps, last_t = 0.0, time.time()
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.time()
                if self.recal:
                    det.reset()
                    self.recal = False
                src = vg.enhance_low_light(frame) if self.enhance else frame
                lm = backend.process(src, int(now * 1000))
                if lm is not None:
                    P = lm * np.array([w, h])
                    st = det.update(P, now)
                    for eye in (P[vg.RIGHT_EYE], P[vg.LEFT_EYE]):
                        cv2.polylines(frame, [eye.astype(np.int32)], True, (0, 255, 0), 1)
                else:
                    st = det.no_face(now)
                stats.max_score = max(stats.max_score, st.score)
                alerter.handle(st, now)
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
                last_t = now
                r = lambda v, n=2: None if v is None else round(v, n)
                self.status = dict(
                    running=True, error=None, calibrated=det.calibrated,
                    calib_left=round(det.calib_left(now), 1), level=st.level,
                    label=vg.LABELS[st.level], reason=st.reason, score=round(st.score),
                    ear=r(st.ear), mar=r(st.mar), closure=r(st.closure, 1), away=r(st.away_sec, 1),
                    fps=round(fps), yawns=stats.yawns, nods=stats.nods, warnings=stats.warnings,
                    dangers=stats.dangers, away_events=stats.away_events,
                    lowlight=self.enhance, voice=alerter.voice.enabled, elapsed=int(now - stats.start))
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.frame_jpg = buf.tobytes()
        except (Exception, SystemExit) as exc:
            self.status = {"running": False, "error": str(exc) or exc.__class__.__name__}
        finally:
            if cap:
                cap.release()
            if backend:
                backend.close()
            if stats:
                vg.save_and_print_summary(stats)
            self.running = False
            self.frame_jpg = None
            self.status = {**self.status, "running": False}


engine = Engine()


@app.get("/")
def index():
    return send_from_directory(BASE, "live.html")


@app.get("/video")
def video():
    def gen():
        while True:
            jpg = engine.frame_jpg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def status():
    return jsonify(engine.status)


@app.get("/api/events")
def events():
    rows = []
    if os.path.exists(vg.LOG_FILE):
        with open(vg.LOG_FILE, newline="") as f:
            rows = list(csv.DictReader(f))
    return jsonify(rows[-15:][::-1])


@app.post("/api/<action>")
def action(action):
    if action == "start":
        engine.start()
    elif action == "stop":
        engine.stop()
    elif action == "recalibrate":
        engine.recal = True
    elif action == "lowlight":
        engine.enhance = not engine.enhance
    elif action == "voice" and engine.alerter and engine.alerter.voice.available:
        engine.alerter.voice.enabled = not engine.alerter.voice.enabled
    return jsonify(ok=True)


if __name__ == "__main__":
    print("VigilEye front end: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)