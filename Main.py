"""VigilEye v3 - real-time driver drowsiness + distraction detection (OpenCV + MediaPipe).

Keys:  q = quit   c = recalibrate   l = toggle low-light enhancement   v = toggle voice

Works with any mediapipe version:
  * old versions  -> uses mp.solutions.face_mesh (models are bundled, no download)
  * new versions  -> uses the Tasks FaceLandmarker (needs face_landmarker.task,
                     which is validated and auto-downloaded if missing/corrupt)

Voice: built in on Windows (no install); other systems need  pip install pyttsx3
"""
import csv
import os
import platform
import queue
import subprocess
import threading
import time
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from cloud_alert import send_cloud_alert
except Exception as exc:  # cloud alerts are optional
    print(f"[info] cloud_alert not available ({exc}); continuing without it.")

    def send_cloud_alert(message):
        print("[cloud alert skipped]", message)

# ======================= SETTINGS =======================
CAMERA_INDEX = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
LOG_FILE = os.path.join(BASE_DIR, "events.csv")
SUMMARY_FILE = os.path.join(BASE_DIR, "session_summary.csv")
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
]

CALIB_SECONDS = 5         # eyes open, mouth closed, look straight at camera
EAR_RATIO = 0.72          # eyes "closed" below 72% of your own open-eye EAR
YAWN_MAR = 0.6            # mouth aspect ratio that counts as a yawn
YAWN_MIN_SEC = 1.2
NOD_DELTA = 0.12          # head-down shift vs calibrated baseline
NOD_MIN_SEC = 0.8
YAW_DELTA = 0.22          # head-turn shift vs baseline that counts as "looking away"
LOOK_AWAY_WARN_SEC = 2.0
LOOK_AWAY_DANGER_SEC = 4.0
WINDOW_SEC = 30           # rolling window for fatigue score
WARN_CLOSURE_SEC = 1.0
DANGER_CLOSURE_SEC = 2.0
WARN_SCORE = 40
DANGER_SCORE = 70
NO_FACE_SEC = 3.0
CLOUD_COOLDOWN_SEC = 60   # min gap between cloud alerts

VOICE_ENABLED = True
VOICE_REPEAT_SEC = 0.5  # repeat the same spoken warning at most this often
VOICE_LINES = {
    "Eyes closed too long": "Danger! Wake up! Please pull over.",
    "High fatigue score": "You seem very tired. Please pull over and rest.",
    "Eyes closing": "Warning. Keep your eyes open.",
    "Fatigue building": "You are getting tired. Consider taking a break.",
    "Looking away from road": "Eyes on the road!",
    "Distracted too long": "Danger! Look at the road now!",
    "Face not visible": "I cannot see your face. Please face the camera.",
}

# Landmark indices (Face Mesh numbering)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_V = [(81, 178), (13, 14), (311, 402)]
MOUTH_H = (78, 308)
NOSE, CHIN, R_EYE_OUT, L_EYE_OUT = 1, 152, 33, 263

COLORS = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 0, 255)}
LABELS = {0: "ALERT DRIVER", 1: "WARNING", 2: "DANGER - WAKE UP!"}
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ======================= FACE BACKENDS =======================
def model_ok(path):
    return (os.path.exists(path) and os.path.getsize(path) > 1_000_000
            and zipfile.is_zipfile(path))


def ensure_model():
    """Return a valid model path, downloading it if needed."""
    if model_ok(MODEL_PATH):
        return MODEL_PATH
    if os.path.exists(MODEL_PATH):
        print("[info] face_landmarker.task is invalid/corrupt - replacing it.")
        os.remove(MODEL_PATH)
    for url in MODEL_URLS:
        try:
            print(f"[info] Downloading model from {url}")
            urllib.request.urlretrieve(url, MODEL_PATH)
            if model_ok(MODEL_PATH):
                print("[info] Model downloaded OK.")
                return MODEL_PATH
            print("[warn] Downloaded file is not a valid model.")
        except Exception as exc:
            print(f"[warn] Download failed: {exc}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
    raise SystemExit(
        "\nCould not get face_landmarker.task automatically.\n"
        "Download it manually in a browser from Google's MediaPipe Face Landmarker\n"
        "page (Models section) and place it here:\n  " + MODEL_PATH +
        "\nIt should be a few MB in size.")


class LegacyBackend:
    """mp.solutions.face_mesh (older mediapipe versions)."""
    name = "legacy solutions"

    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, bgr, ts_ms):
        res = self.fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        return np.array([[p.x, p.y] for p in res.multi_face_landmarks[0].landmark])

    def close(self):
        self.fm.close()


class TasksBackend:
    """Tasks FaceLandmarker (newer mediapipe versions)."""
    name = "tasks FaceLandmarker"

    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        path = ensure_model()
        self.lm = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        self.last_ts = 0

    def process(self, bgr, ts_ms):
        ts_ms = max(ts_ms, self.last_ts + 1)  # timestamps must strictly increase
        self.last_ts = ts_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, ts_ms)
        if not res.face_landmarks:
            return None
        return np.array([[p.x, p.y] for p in res.face_landmarks[0]])

    def close(self):
        self.lm.close()


def create_backend():
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return LegacyBackend()
    except Exception:
        pass
    return TasksBackend()


# ======================= METRICS =======================
def dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_ratio(p):
    """Eye Aspect Ratio from 6 eye points."""
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-6)


def mouth_ratio(P):
    """Mouth Aspect Ratio (large when yawning)."""
    vert = np.mean([dist(P[a], P[b]) for a, b in MOUTH_V])
    return vert / (dist(P[MOUTH_H[0]], P[MOUTH_H[1]]) + 1e-6)


def head_pitch(P):
    """2D pitch proxy: nose position between eye line and chin (rises as head drops)."""
    eye_y = (P[R_EYE_OUT][1] + P[L_EYE_OUT][1]) / 2
    return (P[NOSE][1] - eye_y) / (P[CHIN][1] - eye_y + 1e-6)


def head_yaw(P):
    """2D yaw proxy: nose offset from the eye midpoint, relative to eye distance.
    ~0 when facing the camera; grows in magnitude when the head turns."""
    mid_x = (P[R_EYE_OUT][0] + P[L_EYE_OUT][0]) / 2
    width = abs(P[L_EYE_OUT][0] - P[R_EYE_OUT][0]) + 1e-6
    return (P[NOSE][0] - mid_x) / width


def enhance_low_light(frame):
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((CLAHE.apply(l), a, b)), cv2.COLOR_LAB2BGR)


# ======================= SESSION STATS =======================
@dataclass
class SessionStats:
    start: float = field(default_factory=time.time)
    warnings: int = 0
    dangers: int = 0
    yawns: int = 0
    nods: int = 0
    away_events: int = 0
    max_score: float = 0.0


def save_and_print_summary(stats):
    duration = time.time() - stats.start
    mins, secs = divmod(int(duration), 60)
    print("\n========== VigilEye session summary ==========")
    print(f"Duration          : {mins} min {secs} s")
    print(f"Warnings          : {stats.warnings}")
    print(f"Danger alerts     : {stats.dangers}")
    print(f"Yawns             : {stats.yawns}")
    print(f"Head nods         : {stats.nods}")
    print(f"Looked-away events: {stats.away_events}")
    print(f"Peak fatigue score: {stats.max_score:.0f}/100")
    print("==============================================")
    try:
        new = not os.path.exists(SUMMARY_FILE)
        with open(SUMMARY_FILE, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["ended_at", "duration_sec", "warnings", "dangers",
                             "yawns", "nods", "look_away_events", "peak_score"])
            wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), int(duration), stats.warnings,
                         stats.dangers, stats.yawns, stats.nods, stats.away_events,
                         round(stats.max_score, 1)])
        print(f"Saved to {SUMMARY_FILE}")
    except Exception as exc:
        print(f"[warn] Could not save summary: {exc}")


# ======================= DETECTOR =======================
@dataclass
class Status:
    level: int = 0
    reason: str = "OK"
    score: float = 0.0
    closure: float = 0.0
    away_sec: float = 0.0
    ear: float = None
    mar: float = None


class DrowsinessDetector:
    def __init__(self, stats):
        self.stats = stats
        self.reset()

    def reset(self):
        self.calib_start = time.time()
        self.calib_ear, self.calib_pitch, self.calib_yaw = [], [], []
        self.ear_thr = self.pitch_base = self.yaw_base = None
        self.ear_smooth = None
        self.closed_start = self.mouth_start = self.head_start = self.look_start = None
        self.no_face_start = None
        self.yawn_counted = self.nod_counted = self.look_counted = False
        self.closed_hist, self.yawns, self.nods = deque(), deque(), deque()

    @property
    def calibrated(self):
        return self.ear_thr is not None

    def calib_left(self, now):
        return max(0.0, CALIB_SECONDS - (now - self.calib_start))

    @staticmethod
    def _prune(dq, now):
        while dq and now - dq[0] > WINDOW_SEC:
            dq.popleft()

    def no_face(self, now):
        self.closed_start = self.mouth_start = self.head_start = self.look_start = None
        self.look_counted = False
        self.no_face_start = self.no_face_start or now
        st = Status()
        if self.calibrated and now - self.no_face_start >= NO_FACE_SEC:
            st.level, st.reason = 1, "Face not visible"
        return st

    def update(self, P, now):
        """P = landmark pixel coordinates (N x 2)."""
        self.no_face_start = None
        ear_raw = (eye_ratio(P[RIGHT_EYE]) + eye_ratio(P[LEFT_EYE])) / 2
        self.ear_smooth = ear_raw if self.ear_smooth is None else 0.6 * ear_raw + 0.4 * self.ear_smooth
        st = Status(ear=self.ear_smooth, mar=mouth_ratio(P))
        pitch, yaw = head_pitch(P), head_yaw(P)

        # ---- calibration ----
        if not self.calibrated:
            self.calib_ear.append(ear_raw)
            self.calib_pitch.append(pitch)
            self.calib_yaw.append(yaw)
            if now - self.calib_start >= CALIB_SECONDS and len(self.calib_ear) >= 30:
                self.ear_thr = float(np.median(self.calib_ear)) * EAR_RATIO
                self.pitch_base = float(np.median(self.calib_pitch))
                self.yaw_base = float(np.median(self.calib_yaw))
            return st

        # ---- distraction (head turned away) ----
        away = abs(yaw - self.yaw_base) > YAW_DELTA
        if away:
            self.look_start = self.look_start or now
            st.away_sec = now - self.look_start
            if not self.look_counted and st.away_sec >= LOOK_AWAY_WARN_SEC:
                self.stats.away_events += 1
                self.look_counted = True
        else:
            self.look_start, self.look_counted = None, False

        # ---- eye closure + PERCLOS (ignored while head is turned: eyes unreliable) ----
        closed = (self.ear_smooth < self.ear_thr) and not away
        self.closed_start = (self.closed_start or now) if closed else None
        st.closure = now - self.closed_start if self.closed_start else 0.0
        self.closed_hist.append((now, closed))
        while self.closed_hist and now - self.closed_hist[0][0] > WINDOW_SEC:
            self.closed_hist.popleft()
        perclos = sum(c for _, c in self.closed_hist) / len(self.closed_hist)

        # ---- yawns ----
        if st.mar > YAWN_MAR and not away:
            self.mouth_start = self.mouth_start or now
            if not self.yawn_counted and now - self.mouth_start >= YAWN_MIN_SEC:
                self.yawns.append(now)
                self.stats.yawns += 1
                self.yawn_counted = True
        else:
            self.mouth_start, self.yawn_counted = None, False

        # ---- head nods ----
        if pitch - self.pitch_base > NOD_DELTA:
            self.head_start = self.head_start or now
            if not self.nod_counted and now - self.head_start >= NOD_MIN_SEC:
                self.nods.append(now)
                self.stats.nods += 1
                self.nod_counted = True
        else:
            self.head_start, self.nod_counted = None, False

        self._prune(self.yawns, now)
        self._prune(self.nods, now)

        # ---- fatigue score & level ----
        st.score = (min(perclos / 0.30, 1) * 60
                    + min(len(self.yawns) / 2, 1) * 20
                    + min(len(self.nods) / 3, 1) * 20)
        if st.closure >= DANGER_CLOSURE_SEC:
            st.level, st.reason = 2, "Eyes closed too long"
        elif st.score >= DANGER_SCORE:
            st.level, st.reason = 2, "High fatigue score"
        elif st.away_sec >= LOOK_AWAY_DANGER_SEC:
            st.level, st.reason = 2, "Distracted too long"
        elif st.closure >= WARN_CLOSURE_SEC:
            st.level, st.reason = 1, "Eyes closing"
        elif st.score >= WARN_SCORE:
            st.level, st.reason = 1, "Fatigue building"
        elif st.away_sec >= LOOK_AWAY_WARN_SEC:
            st.level, st.reason = 1, "Looking away from road"
        return st


# ======================= VOICE =======================
class Voice:
    """Speaks warnings on a background thread.
    Windows : PowerShell + System.Speech (no extra install, works for repeated use).
    Others  : pyttsx3 with a fresh engine per message.
    A newer warning replaces one that is still waiting in the queue."""

    def __init__(self):
        self.q = queue.Queue(maxsize=1)
        self.windows = platform.system() == "Windows"
        self.available = VOICE_ENABLED and (self.windows or pyttsx3 is not None)
        self.enabled = self.available
        if VOICE_ENABLED and not self.available:
            print("[info] Voice off. Install it with:  pip install pyttsx3")
        if self.available:
            threading.Thread(target=self._worker, daemon=True).start()

    def _speak(self, text):
        if self.windows:
            safe = text.replace("'", "''")
            cmd = ("Add-Type -AssemblyName System.Speech; "
                   "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                   f"$s.Speak('{safe}')")
            subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           creationflags=0x08000000, timeout=20, check=False)
        else:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    def _worker(self):
        while True:
            text = self.q.get()
            try:
                self._speak(text)
            except Exception as exc:
                print(f"[warn] Voice error: {exc}")

    def say(self, text):
        if not self.enabled:
            return False
        try:
            self.q.put_nowait(text)
        except queue.Full:
            try:
                self.q.get_nowait()          # drop the stale message
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(text)
            except queue.Full:
                return False
        return True


# ======================= ALERTS =======================
class Alerter:
    def __init__(self, stats):
        self.stats = stats
        self.voice = Voice()
        self.prev_level = 0
        self.last_beep = self.last_cloud = self.last_voice = 0.0
        self.last_reason = None

    @staticmethod
    def _beep(level):
        def run():
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000 if level == 1 else 2200, 200 if level == 1 else 600)
            else:
                print("\a", end="", flush=True)
        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def _log(level, score, reason):
        new = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["timestamp", "level", "score", "reason"])
            wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), level, round(score, 1), reason])

    def toggle_voice(self):
        if not self.voice.available:
            print("[info] Voice unavailable on this system (pip install pyttsx3).")
            return
        self.voice.enabled = not self.voice.enabled
        print(f"[info] Voice {'ON' if self.voice.enabled else 'OFF'}")

    def handle(self, st, now):
        # beep
        if st.level > 0 and now - self.last_beep >= (3.0 if st.level == 1 else 1.0):
            self._beep(st.level)
            self.last_beep = now

        # spoken warning
        if st.level > 0:
            changed = st.reason != self.last_reason
            if now - self.last_voice >= (4.0 if changed else VOICE_REPEAT_SEC):
                if self.voice.say(VOICE_LINES.get(st.reason, "Warning. Stay alert.")):
                    self.last_voice = now
        self.last_reason = st.reason if st.level > 0 else None

        # log + cloud + stats on escalation
        if st.level > self.prev_level:
            self._log(st.level, st.score, st.reason)
            if st.level == 1:
                self.stats.warnings += 1
            if st.level == 2:
                self.stats.dangers += 1
                if now - self.last_cloud >= CLOUD_COOLDOWN_SEC:
                    self.last_cloud = now
                    msg = f"VigilEye DANGER: {st.reason} (fatigue score {st.score:.0f})"
                    threading.Thread(target=send_cloud_alert, args=(msg,), daemon=True).start()
        self.prev_level = st.level


# ======================= HUD =======================
def draw_hud(frame, det, st, now, fps, enhance, voice_on):
    h, w = frame.shape[:2]
    color = COLORS[st.level]
    if not det.calibrated:
        cv2.putText(frame, f"Calibrating {det.calib_left(now):.0f}s - eyes open, look at camera",
                    (20, 40), FONT, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, LABELS[st.level], (20, 40), FONT, 1.0, color, 3)
        cv2.putText(frame, st.reason, (20, 70), FONT, 0.6, color, 2)
        cv2.rectangle(frame, (20, 85), (220, 105), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 85), (20 + int(2 * st.score), 105), color, -1)
        cv2.putText(frame, f"Fatigue {st.score:.0f}/100", (230, 102), FONT, 0.55, (255, 255, 255), 1)
    if st.ear is not None:
        cv2.putText(frame,
                    f"EAR {st.ear:.2f}  MAR {st.mar:.2f}  closed {st.closure:.1f}s  away {st.away_sec:.1f}s",
                    (20, h - 40), FONT, 0.55, (255, 255, 255), 1)
    cv2.putText(frame,
                f"FPS {fps:.0f}  yawns {len(det.yawns)}  nods {len(det.nods)}  "
                f"lowlight {'ON' if enhance else 'off'}  voice {'ON' if voice_on else 'off'}",
                (20, h - 15), FONT, 0.55, (255, 255, 255), 1)
    if st.level == 2:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)


# ======================= MAIN =======================
def open_camera():
    api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, api)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try CAMERA_INDEX = 1.")
    return cap


def main():
    backend = create_backend()
    print(f"[info] Face backend: {backend.name}")
    cap = open_camera()
    stats = SessionStats()
    det, alerter = DrowsinessDetector(stats), Alerter(stats)
    enhance, fps, last_t = False, 0.0, time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            src = enhance_low_light(frame) if enhance else frame

            lm = backend.process(src, int(now * 1000))
            if lm is not None:
                P = lm * np.array([w, h])
                st = det.update(P, now)
                for eye in (P[RIGHT_EYE], P[LEFT_EYE]):
                    cv2.polylines(frame, [eye.astype(np.int32)], True, (0, 255, 0), 1)
            else:
                st = det.no_face(now)

            stats.max_score = max(stats.max_score, st.score)
            alerter.handle(st, now)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now
            draw_hud(frame, det, st, now, fps, enhance, alerter.voice.enabled)

            cv2.imshow("VigilEye", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                enhance = not enhance
            if key == ord("v"):
                alerter.toggle_voice()
            if key == ord("c"):
                det.reset()
    finally:
        backend.close()
        cap.release()
        cv2.destroyAllWindows()
        save_and_print_summary(stats)


if __name__ == "__main__":
    main()