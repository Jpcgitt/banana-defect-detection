"""
Banana Sorting System — Vision-Only Trigger + Single Sorter Servo
=================================================================
LATENCY FIXES APPLIED (v2):
  1. IMG_SIZE reduced 64 -> 48  (HOG extraction ~44% faster on Pi CPU)
     !! Must retrain banana_model.pkl after this change !!
  2. VIS_COOLDOWN reduced 2.0s -> 1.0s  (faster re-arm between bananas)
  3. SCAN_TIMEOUT reduced 8.0s -> 4.0s  (partial results finalised sooner)
  4. Classification timing print added to _worker() for profiling
  5. Servo delay tunable comment added — measure actual conveyor travel
     time and lower SERVO_DELAY_S if it is safe to do so.

SERVO DIRECTION FIX (v4):
  Servo now mirrors the working test script exactly:
    HOME  =   0 deg (500 us)  — retracted / ramp-clear position
    PUSH  = 180 deg (2500 us) — extended / eject position
  fire_servo_bad() sequence:
    wait SERVO_DELAY_S → extend to 180° → hold SERVO_HOLD_S → retract to 0°

Process flow
------------
  Banana hangs on hook → travels overhead conveyor at 0.15 m/s →

  Zone A : Camera 1 watches live feed continuously.
           find_banana_roi() locates the banana each frame.
           When ROI centre-X lands within CENTER_TOL_PX of frame
           centre-X  →  snapshot captured  →  classify pipeline.

  Zone B : Camera 2 does the same independently.

  Re-arm logic (per camera):
    Armed by default.
    Capture fires  → camera disarmed.
    ROI disappears (banana left FOV) → camera re-armed for next fruit.
    This prevents double-triggering while the banana lingers in frame.

  AND gate combines both results:
    GOOD + GOOD  →  size check → sz < 13 cm → force BAD (too small)
                               → sz >= 13 cm → FINAL GOOD
                                   servo stays home
                                   banana reaches fixed GOOD-zone ramp naturally
    either BAD   →  FINAL BAD   →  2 s delay → servo extends to 180°
                                    (pushes ramp into hook path → banana slides off)
                                    5 s hold → servo retracts to home (0°)

  All scan data uploaded to Firestore via persistent retry queue.
  Images saved locally.

  On startup: any records left in upload_failures.json (from a previous
  session's dead-letter queue) are automatically re-queued and uploaded.
  No separate recovery script needed.

Hardware pins
-------------
  Sorter Servo : GPIO 25   (BAD ejector only — GOOD does nothing)
  Camera 1     : /dev/video0   (Zone A)
  Camera 2     : /dev/video2   (Zone B)

  Shutdown btn : GPIO 5  (physical pin 29) — pull-up, active LOW
                 Hold >= BUTTON_HOLD_S → clean shutdown
  Reboot   btn : GPIO 6  (physical pin 31) — pull-up, active LOW
                 Hold >= BUTTON_HOLD_S → clean reboot

  No ultrasonic sensors required.

Lighting setup
--------------
  LED strip light co-axial with camera (lights along with the camera at
  the banana).  White fabric background.  Lighting physically adjusted
  to eliminate shadows.

  Consequences addressed in code:
    - White background pixels inside ROI  → masked out before any feature
      extraction so they cannot corrupt the HSV histogram.
    - Strip light brightness falloff along conveyor width → CLAHE applied
      per-frame to equalise luminance before classification.
    - Camera auto-exposure meters off bright white background → exposure
      and white-balance locked manually at startup.
    - Motion blur at 0.15 m/s → sharpness gate rejects blurry captures.
    - SVM uncertainty → confidence gate falls back to HSV when < threshold.

Visual trigger parameters
-------------------------
  CENTER_TOL_PX  = 40 px
  VIS_COOLDOWN   = 1.0 s   [LATENCY FIX: was 2.0s]

Servo parameters
----------------
  HOME_DEG             =   0 deg (500 us)  — retracted / GOOD path (matches test script)
  PUSH_DEG             = 180 deg (2500 us) — extended eject for BAD banana
  SERVO_SIGNAL_SUSTAIN = 0.5  s  (hold PWM until rotor settles, then cut)
  SERVO_DELAY_S        = 2.0  s  (wait after BAD result before rotating)
  SERVO_HOLD_S         = 5.0  s  (hold ejected position before retracting)

Size rejection
--------------
  MIN_SIZE_CM = 13.0

Classification
--------------
  1. banana_model.pkl present  ->  SVM (HOG + HSV histogram, fully offline)
  2. pkl missing               ->  HSV colour-threshold fallback + warning

  NOTE: IMG_SIZE changed from 64 to 48.  Re-run train_model.py before use.

Firestore upload
----------------
  Uses a persistent single-worker queue with exponential back-off retries
  (up to 5 attempts: 2 → 4 → 8 → 16 → 32 s). Failed records after all
  retries are written to upload_failures.json so nothing is silently lost.

  On next startup, recover_dlq_on_startup() reads upload_failures.json,
  checks Firestore for IDs already present, and re-queues only the missing
  ones — safe to run every time, never creates duplicates.
"""

import lgpio
import cv2
import numpy as np
import time
import threading
import pickle
import json
import subprocess
import collections
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# =========================================
# FIREBASE
# =========================================
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db         = firestore.client()
STATUS_DOC = db.collection("device_status").document("current")


def push_device_status(status: str):
    try:
        STATUS_DOC.set({
            "status":     status,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[Firestore] status -> {status}")
    except Exception as e:
        print(f"[Firestore] error: {e}")


# =========================================
# Sanitiser for Firestore & json.dump
# =========================================
def _sanitize(obj):
    """Recursively convert numpy types / tuples to plain Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================================
# FIRESTORE UPLOAD QUEUE
# =========================================
_upload_queue  = collections.deque()
_upload_lock   = threading.Lock()
UPLOAD_RETRIES = 5
UPLOAD_BACKOFF = [2, 4, 8, 16, 32]
DLQ_FILE       = Path(__file__).parent / "upload_failures.json"


def _upload_worker():
    """Persistent daemon thread — one upload at a time, retries on failure."""
    while True:
        item = None
        with _upload_lock:
            if _upload_queue:
                item = _upload_queue.popleft()

        if item is None:
            time.sleep(0.5)
            continue

        doc_id   = item["banana_id"]
        attempts = item.get("_attempts", 0)

        try:
            db.collection("banana_records").document(doc_id).set(item)
            print(f"[Firestore] ✓ Uploaded: {doc_id}  (attempt {attempts + 1})")

        except Exception as e:
            attempts += 1
            print(f"[Firestore] ✗ Failed ({attempts}/{UPLOAD_RETRIES}): "
                  f"{doc_id}  -- {e}")

            if attempts < UPLOAD_RETRIES:
                item["_attempts"] = attempts
                delay = UPLOAD_BACKOFF[min(attempts - 1, len(UPLOAD_BACKOFF) - 1)]
                print(f"[Firestore]   Retrying in {delay}s ...")

                def _requeue(it=item, d=delay):
                    time.sleep(d)
                    with _upload_lock:
                        _upload_queue.appendleft(it)

                threading.Thread(target=_requeue, daemon=True).start()
            else:
                print(f"[Firestore] ✗✗ Giving up on {doc_id} after "
                      f"{UPLOAD_RETRIES} attempts — saving to {DLQ_FILE.name}")
                try:
                    with open(DLQ_FILE, "a") as f:
                        json.dump(_sanitize(item), f)
                        f.write("\n")
                except Exception as fe:
                    print(f"[Firestore] DLQ write error: {fe}")


threading.Thread(target=_upload_worker, daemon=True,
                 name="FirestoreWorker").start()


def upload_to_firestore(data: dict):
    """Non-blocking enqueue. Returns immediately."""
    clean = _sanitize(data)
    clean.setdefault("_attempts", 0)
    with _upload_lock:
        _upload_queue.append(clean)
    print(f"[Firestore] Queued: {clean['banana_id']}  "
          f"(queue depth: {len(_upload_queue)})")


# =========================================
# STARTUP DLQ RECOVERY
# =========================================
def recover_dlq_on_startup():
    if not DLQ_FILE.exists():
        print("[DLQ Recovery] No upload_failures.json found — nothing to recover.")
        return

    dlq_records = []
    with open(DLQ_FILE) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                dlq_records.append(json.loads(raw))
            except json.JSONDecodeError as e:
                print(f"[DLQ Recovery] Line {lineno} parse error: {e} — skipped")

    if not dlq_records:
        print("[DLQ Recovery] upload_failures.json is empty — nothing to recover.")
        return

    print(f"[DLQ Recovery] {len(dlq_records)} record(s) found in DLQ.")

    by_id = {}
    for r in dlq_records:
        bid = r.get("banana_id")
        if bid:
            by_id[bid] = r

    all_ids = list(by_id.keys())

    already_uploaded = set()
    BATCH = 30
    try:
        for i in range(0, len(all_ids), BATCH):
            chunk = all_ids[i: i + BATCH]
            snap  = (
                db.collection("banana_records")
                  .where("banana_id", "in", chunk)
                  .get()
            )
            for doc in snap:
                already_uploaded.add(doc.get("banana_id"))
    except Exception as e:
        print(f"[DLQ Recovery] Firestore existence check failed: {e}")
        print("[DLQ Recovery] Will re-queue all DLQ records to be safe.")

    to_requeue = [r for bid, r in by_id.items() if bid not in already_uploaded]

    print(f"[DLQ Recovery] Already in Firestore : {len(already_uploaded)}")
    print(f"[DLQ Recovery] Re-queuing            : {len(to_requeue)}")

    for r in to_requeue:
        r["_attempts"] = 0
        clean = _sanitize(r)
        with _upload_lock:
            _upload_queue.appendleft(clean)

    requeued_ids  = {r["banana_id"] for r in to_requeue}
    remaining_dlq = [r for r in dlq_records
                     if r.get("banana_id") not in requeued_ids
                     and r.get("banana_id") not in already_uploaded]
    try:
        with open(DLQ_FILE, "w") as f:
            for r in remaining_dlq:
                json.dump(r, f)
                f.write("\n")
        if remaining_dlq:
            print(f"[DLQ Recovery] {len(remaining_dlq)} record(s) kept in DLQ.")
        else:
            print(f"[DLQ Recovery] DLQ cleared — all records re-queued.")
    except Exception as e:
        print(f"[DLQ Recovery] Could not rewrite {DLQ_FILE.name}: {e}")

    if to_requeue:
        print(f"[DLQ Recovery] ✓ {len(to_requeue)} record(s) handed to upload "
              f"worker (queue depth: {len(_upload_queue)})")


# =========================================
# PATHS
# =========================================
BASE_DIR  = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "banana_pictures"
LOG_FILE  = BASE_DIR / "banana_log.json"
IMAGE_DIR.mkdir(exist_ok=True)
print(f"[Images] -> {IMAGE_DIR}")

MODEL_PATH   = BASE_DIR / "banana_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
INFO_PATH    = BASE_DIR / "model_info.json"


def save_image(frame, banana_id: str, cam_label: str):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = IMAGE_DIR / f"{banana_id}_{cam_label}_{ts}.jpg"
    try:
        cv2.imwrite(str(path), frame)
        print(f"  [Image] Saved {path.name}")
    except Exception as e:
        print(f"  [Image] Save error: {e}")


# =========================================
# CLAHE  (luminance equalisation)
# =========================================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Equalise luminance in LAB space. Hue and saturation are unchanged."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# =========================================
# SHARPNESS GATE
# =========================================
SHARPNESS_THRESHOLD = 60.0


def _is_sharp(frame: np.ndarray, roi: tuple) -> bool:
    """Return True only when the banana ROI is sharp enough to classify."""
    if roi is None:
        return False
    rx, ry, rw, rh = roi
    crop = frame[ry:ry + rh, rx:rx + rw]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= SHARPNESS_THRESHOLD


# =========================================
# LOCAL ML MODEL
# =========================================
_ml_model   = None
_ml_encoder = None
_ml_ready   = False
_ml_info    = {}

# -----------------------------------------------------------------
# LATENCY FIX 1: IMG_SIZE reduced from 64 to 48.
# HOG computation scales roughly with (IMG_SIZE^2), so 48x48 is
# ~44% fewer pixels than 64x64 — meaningfully faster on Pi ARM CPU.
# HOG_BLOCK and HOG_STRIDE also scaled down proportionally so the
# descriptor structure remains consistent.
#
# !! IMPORTANT: you MUST retrain banana_model.pkl after this change.
#    Run train_model.py before deploying this file.  If you run this
#    with the old 64-px model the feature vector length will mismatch
#    and predict() will raise a ValueError. !!
# -----------------------------------------------------------------
IMG_SIZE   = 48        # was 64
HOG_CELL   = (8, 8)
HOG_BLOCK  = (16, 16)  # kept the same — still valid at 48px
HOG_STRIDE = (8, 8)
HOG_BINS   = 9
HIST_BINS  = 32

_HOG = cv2.HOGDescriptor(
    (IMG_SIZE, IMG_SIZE),
    HOG_BLOCK, HOG_STRIDE, HOG_CELL, HOG_BINS,
)


def load_local_model():
    global _ml_model, _ml_encoder, _ml_ready, _ml_info

    if not MODEL_PATH.exists():
        print(f"[ML] WARNING: {MODEL_PATH.name} not found -- HSV fallback mode.")
        print("[ML]    Run train_model.py to generate the model.")
        _refresh_title()
        return

    try:
        with open(MODEL_PATH, "rb") as f:
            _ml_model = pickle.load(f)
        print(f"[ML] Loaded {MODEL_PATH.name}")
    except Exception as e:
        print(f"[ML] Failed to load model: {e}")
        _refresh_title()
        return

    if not ENCODER_PATH.exists():
        print(f"[ML] {ENCODER_PATH.name} not found -- run train_model.py again.")
        _refresh_title()
        return

    try:
        with open(ENCODER_PATH, "rb") as f:
            _ml_encoder = pickle.load(f)
        print(f"[ML] Loaded {ENCODER_PATH.name}  "
              f"classes={list(_ml_encoder.classes_)}")
    except Exception as e:
        print(f"[ML] Failed to load encoder: {e}")
        _refresh_title()
        return

    if INFO_PATH.exists():
        try:
            with open(INFO_PATH) as f:
                _ml_info = json.load(f)
            print(f"[ML]    Accuracy={_ml_info.get('accuracy_pct','?')}%  "
                  f"Trained={_ml_info.get('trained_at','?')}  "
                  f"good={_ml_info.get('good_images','?')}  "
                  f"bad={_ml_info.get('bad_images','?')}")
        except Exception:
            pass

    _ml_ready = True
    print("[ML] Local model READY -- fully offline classification active.")
    _refresh_title()


def _refresh_title():
    if _ml_ready:
        txt = (f"BANANA SORTING SYSTEM  --  "
               f"Local ML ({_ml_info.get('accuracy_pct','?')}% acc)")
        col = "#f0c040"
    else:
        txt = "BANANA SORTING SYSTEM  --  HSV Fallback  (run train_model.py)"
        col = "#ff9800"
    try:
        root.after(0, lambda: title_label.config(text=txt, fg=col))
    except Exception:
        pass


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    HOG shape + HSV colour histogram feature vector.
    Must stay in sync with train_model.py.
    IMG_SIZE is now 48 — retrain model before use.
    """
    img_bgr = _apply_clahe(img_bgr)

    img     = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_vec = _HOG.compute(gray).flatten()

    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [HIST_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [HIST_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [HIST_BINS], [0, 256]).flatten()
    colour_vec = np.concatenate([h_hist, s_hist, v_hist])
    total = colour_vec.sum()
    if total > 0:
        colour_vec /= total

    return np.concatenate([hog_vec, colour_vec]).astype(np.float32).reshape(1, -1)


# =========================================
# BANANA ID GENERATOR
# =========================================
COUNTER_FILE = BASE_DIR / "banana_counter.json"


def load_counter():
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(COUNTER_FILE) as f:
            d = json.load(f)
        if d.get("date") == today:
            return today, int(d.get("counter", 1))
    except Exception:
        pass
    return today, 1


def save_counter(today: str, n: int):
    try:
        with open(COUNTER_FILE, "w") as f:
            json.dump({"date": today, "counter": n}, f)
    except Exception as e:
        print(f"[Counter] {e}")


_id_lock = threading.Lock()


def make_banana_id() -> str:
    global _today_str, _banana_counter
    with _id_lock:
        now   = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today != _today_str:
            _today_str, _banana_counter = today, 1
        bid             = f"BN{now.strftime('%m%d%y')}{_banana_counter:06d}"
        _banana_counter += 1
        save_counter(_today_str, _banana_counter)
    return bid


_today_str, _banana_counter = load_counter()
print(f"[Counter] Start at {_banana_counter} for {_today_str}")


# =========================================
# GPIO SETUP
# =========================================
SERVO_PIN = 25
PWM_FREQ  = 50        # standard servo PWM frequency (Hz)

# -----------------------------------------------------------------
# SERVO DIRECTION FIX (v4) — matches working test script exactly:
#
#   HOME  =   0 deg (500 us)  — retracted / ramp-clear position
#   PUSH  = 180 deg (2500 us) — extended / eject position
#
#   fire_servo_bad() sequence:
#     wait SERVO_DELAY_S
#     → extend to 180° (set_angle(180) in test script)
#     → hold SERVO_HOLD_S
#     → retract to 0°  (set_angle(0)   in test script)
#
#   _set_servo_angle() mirrors test script set_angle() exactly:
#     tx_pwm → time.sleep(0.5) → tx_pwm(0,0)
# -----------------------------------------------------------------
SERVO_HOME_DEG = 0     # retracted / GOOD path  (500 us)
SERVO_PUSH_DEG = 180   # extended  / BAD eject  (2500 us)

# How long to hold PWM on after commanding a new angle so the rotor
# physically completes its travel before the signal is cut.
SERVO_SIGNAL_SUSTAIN_S = 0.5

# -----------------------------------------------------------------
# Shutdown Event — used to wake threads that are sleeping with
# time.sleep() so they exit cleanly when the program closes.
# -----------------------------------------------------------------
_stop_event = threading.Event()

SHUTDOWN_PIN   = 5
REBOOT_PIN     = 6
BUTTON_HOLD_S  = 5.0
BUTTON_POLL_MS = 50

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, SERVO_PIN, 0)


# ---- helpers ----
def _angle_to_duty(pulse_us: float) -> float:
    """Convert a pulse-width in µs to a duty-cycle % for a 50 Hz signal."""
    return (pulse_us / 20000.0) * 100.0


def _set_servo_angle(deg: float):
    """
    Drive the servo to `deg` degrees.
    Mirrors test script set_angle() exactly:
      1. Send PWM pulse
      2. Plain time.sleep(0.5) — rotor physically reaches the target
      3. Cut PWM — servo holds position mechanically without power
    """
    pulse_us = 500.0 + (deg / 180.0) * 2000.0
    lgpio.tx_pwm(h, SERVO_PIN, PWM_FREQ, _angle_to_duty(pulse_us))
    time.sleep(SERVO_SIGNAL_SUSTAIN_S)   # plain sleep — matches test script
    lgpio.tx_pwm(h, SERVO_PIN, 0, 0)    # cut PWM — hold position mechanically


def _servo_home():
    """Retract servo back to home position (0° / 500 us)."""
    _set_servo_angle(SERVO_HOME_DEG)


# Home the servo at startup
_set_servo_angle(SERVO_HOME_DEG)
print(f"[GPIO] Servo homed to {SERVO_HOME_DEG}° (500 us) and PWM cut at startup.")

lgpio.gpio_claim_input(h, SHUTDOWN_PIN, lgpio.SET_PULL_UP)
lgpio.gpio_claim_input(h, REBOOT_PIN,   lgpio.SET_PULL_UP)

print(f"[GPIO] Servo      GPIO{SERVO_PIN}  "
      f"HOME={SERVO_HOME_DEG}° (500us)  PUSH={SERVO_PUSH_DEG}° (2500us)  "
      f"sustain={SERVO_SIGNAL_SUSTAIN_S}s  PWM={PWM_FREQ}Hz")
print(f"[GPIO] Shutdown   GPIO{SHUTDOWN_PIN}  (hold {BUTTON_HOLD_S}s -> poweroff)")
print(f"[GPIO] Reboot     GPIO{REBOOT_PIN}  (hold {BUTTON_HOLD_S}s -> reboot)")


# =========================================
# TIMING CONSTANTS
# =========================================
CONVEYOR_SPEED = 0.15

# -----------------------------------------------------------------
# LATENCY FIX 2: SERVO_DELAY_S
# Intentional travel-time compensation — banana must physically
# travel from the camera scan point to the servo ejector.
# Measure: delay = distance_camera_to_servo_m / CONVEYOR_SPEED
# Default kept at 2.0 s. Lower only after measuring.
# -----------------------------------------------------------------
SERVO_DELAY_S = 2.0

SERVO_HOLD_S  = 5.0   # hold ejected position for 5 s before retracting

# -----------------------------------------------------------------
# LATENCY FIX 3: SCAN_TIMEOUT reduced from 8.0 s to 4.0 s.
# -----------------------------------------------------------------
SCAN_TIMEOUT       = 4.0    # was 8.0

FULL_CYCLE_TIME    = SERVO_DELAY_S + SERVO_HOLD_S + 5.0
SAME_BANANA_WINDOW = 8.0

# =========================================
# VISUAL TRIGGER PARAMETERS
# =========================================
CENTER_TOL_PX   = 40
CENTER_TOL_Y_PX = 80
MIN_TRIGGER_PX  = 8000

# -----------------------------------------------------------------
# LATENCY FIX 4: VIS_COOLDOWN reduced from 2.0 s to 1.0 s.
# -----------------------------------------------------------------
VIS_COOLDOWN = 1.0   # was 2.0

# =========================================
# SIZE REJECTION THRESHOLD
# =========================================
MIN_SIZE_CM = 13.0


def is_size_acceptable(cm: float) -> bool:
    return cm >= MIN_SIZE_CM


# =========================================
# CAMERAS
# =========================================
SCAN_W = 640
SCAN_H = 480
DISP_W = 480
DISP_H = 360

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(2)

for cam, idx in ((cam1, 0), (cam2, 2)):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  SCAN_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, SCAN_H)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE,   0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE,        -5)
    cam.set(cv2.CAP_PROP_AUTO_WB,          0)
    cam.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

    st = "OK" if cam.isOpened() else "NOT FOUND"
    print(f"[Camera {idx}] {st}  ({int(cam.get(3))}x{int(cam.get(4))})")


# =========================================
# HSV THRESHOLDS
# =========================================
SEG_GREEN_LOW   = np.array([ 22,  55,  60])
SEG_GREEN_HIGH  = np.array([ 88, 255, 255])
SEG_YELLOW_LOW  = np.array([ 18,  70,  90])
SEG_YELLOW_HIGH = np.array([ 32, 255, 255])

MIN_BANANA_PX         = 3000
DARK_V_MAX            = 80
BROWN_HUE_LOW         = np.array([  5,  30,  20])
BROWN_HUE_HIGH        = np.array([ 22, 255, 200])
SCRATCH_S_MAX         = 80
SCRATCH_V_MIN         = 20
SCRATCH_V_MAX         = 150
OVERRIPE_HUE_LOW      = np.array([ 10,  80, 100])
OVERRIPE_HUE_HIGH     = np.array([ 18, 255, 255])
DEFECT_RATIO_MAX      = 0.04
MIN_DEFECT_CONTOUR_PX = 15
CM_PER_PIXEL          = 0.05
MIN_BBOX_AREA         = 300

CONFIDENCE_THRESHOLD  = 0.70


# =========================================
# SHARED STATE
# =========================================
_state_lock = threading.Lock()
_shared = {
    "frame1":         None,
    "frame2":         None,
    "overlay1":       None,
    "overlay2":       None,
    "last_grade1":    "",
    "last_grade2":    "",
    "last_final":     "",
    "last_banana_id": "",
    "pipeline_count": 0,
    "results":        [],
    "cam1_busy":      False,
    "cam2_busy":      False,
    "ml_status1":     "",
    "ml_status2":     "",
    "vis1_offset":    0,
    "vis2_offset":    0,
    "vis1_state":     "ARMED",
    "vis2_state":     "ARMED",
    "system_action":  "",
    "btn_hold_pct":   0,
}


def state_write(**kw):
    with _state_lock:
        _shared.update(kw)


def state_read():
    with _state_lock:
        return dict(_shared)


# =========================================
# ROI DETECTION
# =========================================
def find_banana_roi(frame: np.ndarray, min_area: int = MIN_BBOX_AREA):
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask  = cv2.inRange(hsv, SEG_GREEN_LOW,  SEG_GREEN_HIGH)
    yellow_mask = cv2.inRange(hsv, SEG_YELLOW_LOW, SEG_YELLOW_HIGH)
    banana_mask = cv2.bitwise_or(green_mask, yellow_mask)

    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_CLOSE,  k11, iterations=3)
    banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_DILATE, k11, iterations=1)

    cnts, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None

    x, y, w, hh = cv2.boundingRect(best)
    pad = 10
    x  = max(0, x - pad)
    y  = max(0, y - pad)
    w  = min(frame.shape[1] - x, w + 2 * pad)
    hh = min(frame.shape[0] - y, hh + 2 * pad)
    return x, y, w, hh


# =========================================
# HSV HELPERS
# =========================================
def _segment_banana(hsv: np.ndarray) -> np.ndarray:
    green_mask  = cv2.inRange(hsv, SEG_GREEN_LOW,  SEG_GREEN_HIGH)
    yellow_mask = cv2.inRange(hsv, SEG_YELLOW_LOW, SEG_YELLOW_HIGH)
    mask = cv2.bitwise_or(green_mask, yellow_mask)
    k11  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  k11, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k11, iterations=1)
    return mask


def _detect_defects(hsv: np.ndarray, banana_mask: np.ndarray) -> np.ndarray:
    dark_mask     = cv2.inRange(hsv,
                                np.array([  0,   0,   0]),
                                np.array([180, 255, DARK_V_MAX]))
    brown_mask    = cv2.inRange(hsv, BROWN_HUE_LOW,    BROWN_HUE_HIGH)
    overripe_mask = cv2.inRange(hsv, OVERRIPE_HUE_LOW, OVERRIPE_HUE_HIGH)
    scratch_mask  = cv2.inRange(hsv,
                                np.array([  0,           0, SCRATCH_V_MIN]),
                                np.array([180, SCRATCH_S_MAX, SCRATCH_V_MAX]))
    combined  = cv2.bitwise_or(dark_mask,    brown_mask)
    combined  = cv2.bitwise_or(combined,     overripe_mask)
    combined  = cv2.bitwise_or(combined,     scratch_mask)
    on_banana = cv2.bitwise_and(combined,    banana_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    on_banana = cv2.morphologyEx(on_banana, cv2.MORPH_OPEN, k3)

    cleaned = np.zeros_like(on_banana)
    cnts, _ = cv2.findContours(on_banana, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_DEFECT_CONTOUR_PX:
            cv2.drawContours(cleaned, [c], -1, 255, -1)
    return cleaned


# =========================================
# BACKGROUND MASKING
# =========================================
def _masked_crop(img_bgr: np.ndarray, roi: tuple):
    rx, ry, rw, rh = roi
    crop = img_bgr[ry:ry + rh, rx:rx + rw].copy()
    if crop.size == 0:
        return crop, np.zeros((rh, rw), dtype=np.uint8)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = _segment_banana(hsv)
    crop[mask == 0] = 0
    return crop, mask


# =========================================
# CLASSIFICATION -- HSV FALLBACK
# =========================================
def classify_banana_hsv(frame: np.ndarray, cam_label: str = "") -> tuple:
    frame = _apply_clahe(frame)

    roi = find_banana_roi(frame)
    if roi is None:
        return "EMPTY", 0.0, {"method": "HSV", "roi": None}

    crop, banana_mask = _masked_crop(frame, roi)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    banana_px = int(np.sum(banana_mask > 0))
    if banana_px < MIN_BANANA_PX:
        return "EMPTY", 0.0, {"method": "HSV", "banana_px": banana_px,
                               "roi": list(roi)}

    defect_mask  = _detect_defects(hsv, banana_mask)
    defect_px    = int(np.sum(defect_mask > 0))
    defect_ratio = defect_px / banana_px
    defect_pct   = round(defect_ratio * 100, 2)
    grade        = "BAD" if defect_ratio >= DEFECT_RATIO_MAX else "GOOD"

    print(f"  [{cam_label}][HSV] {grade}  defect={defect_pct}%")
    return grade, defect_pct, {
        "method":       "HSV",
        "banana_px":    banana_px,
        "defect_px":    defect_px,
        "defect_ratio": round(defect_ratio, 4),
        "defect_pct":   defect_pct,
        "roi":          list(roi),
    }


# =========================================
# CLASSIFICATION -- LOCAL ML MODEL
# =========================================
def classify_banana_local(frame: np.ndarray, cam_label: str = "") -> tuple:
    if not _ml_ready:
        return classify_banana_hsv(frame, cam_label)

    frame = _apply_clahe(frame)

    roi = find_banana_roi(frame)
    if roi is None:
        print(f"  [{cam_label}][ML] EMPTY (no ROI)")
        return "EMPTY", 0.0, {"method": "ML", "roi": None}

    crop, banana_mask = _masked_crop(frame, roi)

    banana_px = int(np.sum(banana_mask > 0))
    if banana_px < MIN_BANANA_PX:
        print(f"  [{cam_label}][ML] EMPTY ({banana_px}px)")
        return "EMPTY", 0.0, {"method": "ML", "banana_px": banana_px,
                               "roi": list(roi)}

    feat = extract_features(crop)

    try:
        proba      = _ml_model.predict_proba(feat)[0]
        pred_idx   = int(np.argmax(proba))
        pred_label = _ml_encoder.classes_[pred_idx].upper()
        confidence = float(proba[pred_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"  [{cam_label}][ML] Low confidence ({confidence:.2f}) "
                  f"-> HSV fallback")
            return classify_banana_hsv(frame, cam_label)

        classes_list = list(_ml_encoder.classes_)
        if "bad" in classes_list:
            bad_idx  = classes_list.index("bad")
            bad_prob = float(proba[bad_idx])
        else:
            bad_prob = 1.0 - confidence if pred_label == "GOOD" else confidence

        defect_pct = round(bad_prob * 100, 2)
    except Exception as e:
        print(f"  [{cam_label}][ML] Predict error: {e} -> HSV fallback")
        return classify_banana_hsv(frame, cam_label)

    grade = pred_label
    print(f"  [{cam_label}][ML] {grade}  bad_prob={bad_prob:.3f}  "
          f"conf={confidence:.3f}")
    return grade, defect_pct, {
        "method":     "ML",
        "banana_px":  banana_px,
        "defect_pct": defect_pct,
        "bad_prob":   round(bad_prob, 4),
        "confidence": round(confidence, 4),
        "roi":        list(roi),
    }


def classify_banana(frame: np.ndarray, cam_label: str = "") -> tuple:
    if _ml_ready:
        return classify_banana_local(frame, cam_label)
    return classify_banana_hsv(frame, cam_label)


# =========================================
# SIZE ESTIMATION
# =========================================
def estimate_size(frame) -> float:
    roi = find_banana_roi(frame)
    if roi is None:
        return 0.0
    _, _, w, hh = roi
    return round(max(w, hh) * CM_PER_PIXEL, 2)


# =========================================
# save_banana_data
# =========================================
def save_banana_data(data: dict):
    try:
        clean = _sanitize(data)
        with open(LOG_FILE, "a") as f:
            json.dump(clean, f)
            f.write("\n")
    except Exception as e:
        print(f"[Log] {e}")


# =========================================
# SORTER SERVO
# =========================================
# Matches test script cycle() exactly:
#   HOME  =   0° (500 us)  — retracted, ramp clear
#   PUSH  = 180° (2500 us) — extended, BAD eject
#
# fire_servo_bad() sequence:
#   wait SERVO_DELAY_S → extend to 180° → hold SERVO_HOLD_S → retract to 0°
# =========================================
_servo_lock = threading.Lock()


def fire_servo_bad():
    """
    BAD ejection sequence — mirrors test script cycle() exactly:
      wait SERVO_DELAY_S  →  extend to 180° (set_angle(180))
      →  hold SERVO_HOLD_S  →  retract to 0° (set_angle(0))

    Wrapped in a threading.Lock so two BAD results in quick
    succession cannot overlap on the servo.
    """
    def _fire():
        with _servo_lock:
            # --- wait for banana to travel to ejector ---
            print(f"  [Servo] BAD — waiting {SERVO_DELAY_S}s for banana "
                  f"to reach ejector ...")
            time.sleep(SERVO_DELAY_S)

            # --- extend to 180° (eject) — mirrors set_angle(180) ---
            print(f"  [Servo] EXTEND -> {SERVO_PUSH_DEG}°  (2500 us)")
            _set_servo_angle(SERVO_PUSH_DEG)

            # --- hold ejected position ---
            print(f"  [Servo] Holding ejected position for {SERVO_HOLD_S}s ...")
            time.sleep(SERVO_HOLD_S)

            # --- retract to home (0°) — mirrors set_angle(0) ---
            print(f"  [Servo] RETRACT HOME -> {SERVO_HOME_DEG}°  (500 us)")
            _servo_home()

    threading.Thread(target=_fire, daemon=True).start()


# =========================================
# PIPELINE
# =========================================
class BananaInFlight:
    _seq      = 0
    _seq_lock = threading.Lock()

    def __init__(self, t: float):
        with BananaInFlight._seq_lock:
            BananaInFlight._seq += 1
            seq = BananaInFlight._seq
        self.label            = f"Banana#{seq}"
        self.scan_start_time  = t
        self.scan1_done       = False;  self.scan1_grade = None
        self.scan1_defect     = 0.0;    self.scan1_frame = None
        self.scan1_debug      = {}
        self.scan2_done       = False;  self.scan2_grade = None
        self.scan2_defect     = 0.0;    self.scan2_frame = None
        self.scan2_debug      = {}
        self.grade_determined = False
        self.banana_data      = None


pipeline       = []
_pipeline_lock = threading.Lock()


# =========================================
# GRADE FINALISATION
# =========================================
def finalise_grade(b: BananaInFlight):
    def effective(g):
        return "GOOD" if g in ("GOOD", "EMPTY", None) else "BAD"

    g1 = effective(b.scan1_grade if b.scan1_done else b.scan2_grade)
    g2 = effective(b.scan2_grade if b.scan2_done else b.scan1_grade)
    d1 = b.scan1_defect if b.scan1_done else b.scan2_defect
    d2 = b.scan2_defect if b.scan2_done else b.scan1_defect

    final_grade = "GOOD" if (g1 == "GOOD" and g2 == "GOOD") else "BAD"
    defect_pct  = round((d1 + d2) / 2, 2)

    ref = b.scan1_frame or b.scan2_frame
    sz  = estimate_size(ref) if ref is not None else 0.0
    bid = make_banana_id()
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    classifier = "ML" if _ml_ready else "HSV-fallback"

    if not is_size_acceptable(sz):
        final_grade   = "BAD"
        size_rejected = True
    else:
        size_rejected = False

    b.banana_data = {
        "timestamp":     ts,
        "banana_id":     bid,
        "size_cm":       sz,
        "size_rejected": size_rejected,
        "defect_pct":    defect_pct,
        "grade":         final_grade,
        "defect":        final_grade == "BAD",
        "cam1_grade":    b.scan1_grade,
        "cam2_grade":    b.scan2_grade,
        "cam1_debug":    b.scan1_debug,
        "cam2_debug":    b.scan2_debug,
        "classifier":    classifier,
    }

    bar       = "=" * 55
    size_note = f"REJECTED (< {MIN_SIZE_CM}cm)" if size_rejected else "OK"
    print(f"\n{bar}")
    print(f"  {b.label}  ->  {bid}  [{classifier}]")
    print(f"  Cam1 : {b.scan1_grade}  ({b.scan1_defect}%)")
    print(f"  Cam2 : {b.scan2_grade}  ({b.scan2_defect}%)")
    print(f"  -- AND gate + size check ---------------------------")
    print(f"  FINAL: {final_grade}  |  {defect_pct}%  |  {sz}cm  size={size_note}")
    print(bar)

    save_banana_data(b.banana_data)

    for frame, lbl in ((b.scan1_frame, "cam1"), (b.scan2_frame, "cam2")):
        if frame is not None:
            threading.Thread(target=save_image,
                             args=(frame, bid, lbl), daemon=True).start()

    upload_to_firestore(b.banana_data)

    if final_grade == "BAD":
        reason = "too small" if size_rejected else "defect/vision"
        print(f"  [BAD]  -> Servo EXTEND sequence starting  ({reason})")
        fire_servo_bad()
    else:
        print("  [GOOD] -> Servo stays home  (banana reaches GOOD-zone ramp)")

    if b.scan1_done:
        c1_display = b.scan1_grade or "N/A"
    else:
        c1_display = f"~{b.scan2_grade}" if b.scan2_grade else "N/A"

    if b.scan2_done:
        c2_display = b.scan2_grade or "N/A"
    else:
        c2_display = f"~{b.scan1_grade}" if b.scan1_grade else "N/A"

    with _state_lock:
        _shared["last_grade1"]    = str(b.scan1_grade or "")
        _shared["last_grade2"]    = str(b.scan2_grade or "")
        _shared["last_final"]     = final_grade
        _shared["last_banana_id"] = bid
        _shared["cam1_busy"]      = False
        _shared["cam2_busy"]      = False
        _shared["results"].insert(0, {
            "time":    ts,
            "id":      bid,
            "size":    sz,
            "size_ok": "NO" if size_rejected else "YES",
            "defect":  f"{defect_pct}%",
            "grade":   final_grade,
            "c1":      c1_display,
            "c2":      c2_display,
        })
        if len(_shared["results"]) > 50:
            _shared["results"] = _shared["results"][:50]


# =========================================
# CLASSIFY DISPATCHER
# =========================================
def _dispatch_classify(cam_id: int, frame: np.ndarray, cam_label: str):
    now = time.time()
    state_write(**{f"ml_status{cam_id}": "Classifying..."})

    with _pipeline_lock:
        target = None
        for b in pipeline:
            if b.grade_determined:
                continue
            if (now - b.scan_start_time) > SAME_BANANA_WINDOW:
                continue
            if cam_id == 1 and b.scan1_frame is None:
                target = b; break
            if cam_id == 2 and b.scan2_frame is None:
                target = b; break

        if target is None:
            target = BananaInFlight(now)
            pipeline.append(target)
            state_write(pipeline_count=len(pipeline))
            print(f"  -> new {target.label}  [{len(pipeline)} in pipeline]")
        else:
            print(f"  -> attached to {target.label}")

        if cam_id == 1:
            target.scan1_frame = frame
        else:
            target.scan2_frame = frame

    def _worker():
        # -----------------------------------------------------------------
        # LATENCY FIX 5: classification timing print.
        # -----------------------------------------------------------------
        t0 = time.time()
        grade, defect, dbg = classify_banana(frame, cam_label)
        elapsed = time.time() - t0
        print(f"  [TIMING] {cam_label} classify_banana took {elapsed:.3f}s")

        with _pipeline_lock:
            if cam_id == 1:
                target.scan1_grade  = grade
                target.scan1_defect = defect
                target.scan1_debug  = dbg
                target.scan1_done   = True
            else:
                target.scan2_grade  = grade
                target.scan2_defect = defect
                target.scan2_debug  = dbg
                target.scan2_done   = True

        method = dbg.get("method", "HSV")
        icon   = "OK" if grade in ("GOOD", "EMPTY") else "!!"
        msg    = f"[{icon}] {grade}  ({defect:.1f}%)  [{method}]"
        if cam_id == 1:
            state_write(cam1_busy=False, ml_status1=msg)
        else:
            state_write(cam2_busy=False, ml_status2=msg)

        _check_finalise(target)

    threading.Thread(target=_worker, daemon=True).start()


# =========================================
# CHECK FINALISE
# =========================================
def _check_finalise(b: BananaInFlight):
    with _pipeline_lock:
        if b.grade_determined:
            return
        if b.scan1_done and b.scan2_done:
            b.grade_determined = True
        else:
            return
    threading.Thread(target=finalise_grade, args=(b,), daemon=True).start()


# =========================================
# RUNNING FLAG
# =========================================
_running = True


# =========================================
# TIMEOUT WATCHER
# =========================================
def timeout_watcher():
    while _running:
        now = time.time()
        with _pipeline_lock:
            for b in list(pipeline):
                if b.grade_determined:
                    continue
                elapsed      = now - b.scan_start_time
                slot1_active = b.scan1_frame is not None
                slot2_active = b.scan2_frame is not None

                if (slot1_active or slot2_active) and elapsed >= SCAN_TIMEOUT:
                    b.grade_determined = True
                    if slot1_active and not slot2_active:
                        missed = "Cam2"
                    elif slot2_active and not slot1_active:
                        missed = "Cam1"
                    else:
                        missed = "none (both fired)"
                    print(f"  [Timeout] {missed} never completed -- "
                          f"grading partial  ({elapsed:.1f}s elapsed)")
                    threading.Thread(target=finalise_grade,
                                     args=(b,), daemon=True).start()

        with _pipeline_lock:
            before = len(pipeline)
            alive  = [
                b for b in pipeline
                if not b.grade_determined
                or (now - b.scan_start_time) < FULL_CYCLE_TIME
            ]
            pipeline[:] = alive
            if len(pipeline) != before:
                state_write(pipeline_count=len(pipeline))

        time.sleep(0.25)


# =========================================
# VISUAL CENTER WATCHER
# =========================================
def visual_center_watcher():
    armed        = {1: True,  2: True}
    roi_present  = {1: False, 2: False}
    last_trigger = {1: 0.0,   2: 0.0}

    frame_cx = SCAN_W // 2
    frame_cy = SCAN_H // 2

    cam_configs = [
        (1, "frame1", "cam1_busy", "vis1_offset", "vis1_state", "Cam1"),
        (2, "frame2", "cam2_busy", "vis2_offset", "vis2_state", "Cam2"),
    ]

    while _running:
        st = state_read()

        for cam_id, frame_key, busy_key, offset_key, vstate_key, label \
                in cam_configs:

            frame = st.get(frame_key)
            if frame is None:
                continue

            roi = find_banana_roi(frame, min_area=MIN_TRIGGER_PX)

            if roi is None:
                if roi_present[cam_id]:
                    armed[cam_id]       = True
                    roi_present[cam_id] = False
                    print(f"  [VIS{cam_id}] Banana left FOV -- camera re-armed")
                state_write(**{offset_key: 0,
                               vstate_key: ("ARMED" if armed[cam_id]
                                            else "DISARMED")})
                continue

            roi_present[cam_id] = True
            rx, ry, rw, rh = roi
            roi_cx   = rx + rw // 2
            roi_cy   = ry + rh // 2
            offset_x = roi_cx - frame_cx
            offset_y = roi_cy - frame_cy

            now         = time.time()
            centred_x   = abs(offset_x) <= CENTER_TOL_PX
            centred_y   = abs(offset_y) <= CENTER_TOL_Y_PX
            centred     = centred_x and centred_y
            cooldown_ok = (now - last_trigger[cam_id]) >= VIS_COOLDOWN
            busy        = st[busy_key]

            if centred and armed[cam_id] and cooldown_ok and not busy:
                if not _is_sharp(frame, roi):
                    print(f"  [VIS{cam_id}] Frame blurry -- staying armed")
                    state_write(**{offset_key: offset_x,
                                   vstate_key: "BLURRY"})
                    continue

                armed[cam_id]        = False
                last_trigger[cam_id] = now
                snapshot             = frame.copy()
                state_write(**{busy_key:   True,
                               offset_key: offset_x,
                               vstate_key: "TRIGGERED"})
                print(f"\n[VIS{cam_id}] {label} banana centred  "
                      f"X={offset_x:+d}px  Y={offset_y:+d}px  -> capture")
                _dispatch_classify(cam_id, snapshot, label)
            else:
                if not armed[cam_id]:
                    vstate = "DISARMED"
                elif centred:
                    vstate = "CENTRED"
                elif not centred_x and not centred_y:
                    vstate = f"ARMED X{offset_x:+d} Y{offset_y:+d}"
                elif not centred_x:
                    vstate = f"ARMED X{offset_x:+d}"
                else:
                    vstate = f"ARMED Y{offset_y:+d}"
                state_write(**{offset_key: offset_x, vstate_key: vstate})

        time.sleep(0.033)


# =========================================
# CAMERA LOOP
# =========================================
def camera_loop():
    while _running:
        ret1, f1 = cam1.read()
        ret2, f2 = cam2.read()
        st = state_read()

        if ret1:
            f1  = cv2.resize(f1, (SCAN_W, SCAN_H))
            ov1 = draw_fov_overlay(f1, st["last_grade1"],
                                   st.get("vis1_offset", 0))
            state_write(frame1=f1.copy(), overlay1=ov1)

        if ret2:
            f2  = cv2.resize(f2, (SCAN_W, SCAN_H))
            ov2 = draw_fov_overlay(f2, st["last_grade2"],
                                   st.get("vis2_offset", 0))
            state_write(frame2=f2.copy(), overlay2=ov2)

        time.sleep(0.033)


# =========================================
# GUI OVERLAY
# =========================================
GRID_COLS  = 8
GRID_ROWS  = 6
GRID_COL   = (50,  50,  50)
CROSS_COL  = (0,  180, 255)
BBOX_COL   = (0,  255, 120)
DEF_COL    = (0,   60, 255)
CENTRE_COL = (255, 200,   0)


def draw_fov_overlay(frame: np.ndarray, grade_label: str = "",
                     offset_px: int = 0) -> np.ndarray:
    disp       = cv2.resize(frame, (DISP_W, DISP_H))
    h_px, w_px = disp.shape[:2]
    sx = DISP_W / SCAN_W
    sy = DISP_H / SCAN_H

    cw, ch = w_px // GRID_COLS, h_px // GRID_ROWS
    for c in range(1, GRID_COLS):
        cv2.line(disp, (c*cw, 0), (c*cw, h_px), GRID_COL, 1)
    for r in range(1, GRID_ROWS):
        cv2.line(disp, (0, r*ch), (w_px, r*ch), GRID_COL, 1)
    cv2.rectangle(disp, (0, 0), (w_px-1, h_px-1), GRID_COL, 2)

    cx = w_px // 2
    cy = h_px // 2

    tol_x_disp = int(CENTER_TOL_PX   * sx)
    tol_y_disp = int(CENTER_TOL_Y_PX * sy)
    cv2.rectangle(disp,
                  (cx - tol_x_disp, 0),
                  (cx + tol_x_disp, h_px),
                  CENTRE_COL, 1)
    cv2.rectangle(disp,
                  (0, cy - tol_y_disp),
                  (w_px - 1, cy + tol_y_disp),
                  CENTRE_COL, 1)
    cv2.line(disp, (cx-20, cy), (cx+20, cy), CROSS_COL, 2)
    cv2.line(disp, (cx, cy-20), (cx, cy+20), CROSS_COL, 2)

    roi = find_banana_roi(frame, min_area=MIN_BBOX_AREA)
    if roi is not None:
        rx, ry, rw, rh = roi
        drx = int(rx * sx);  dry = int(ry * sy)
        drw = int(rw * sx);  drh = int(rh * sy)

        roi_cx_disp = drx + drw // 2
        roi_cy_disp = dry + drh // 2
        offset_x    = (rx + rw // 2) - (SCAN_W // 2)
        offset_y    = (ry + rh // 2) - (SCAN_H // 2)
        in_zone     = (abs(offset_x) <= CENTER_TOL_PX and
                       abs(offset_y) <= CENTER_TOL_Y_PX)

        hsv_chk    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gm         = cv2.inRange(hsv_chk, SEG_GREEN_LOW,  SEG_GREEN_HIGH)
        ym         = cv2.inRange(hsv_chk, SEG_YELLOW_LOW, SEG_YELLOW_HIGH)
        bm         = cv2.bitwise_or(gm, ym)
        trigger_ok = int(np.sum(bm > 0)) >= MIN_TRIGGER_PX
        sharp      = _is_sharp(frame, roi)

        if not trigger_ok:
            bbox_col = (0, 140, 255)
        elif in_zone and sharp:
            bbox_col = (0, 255, 60)
        else:
            bbox_col = BBOX_COL

        cv2.rectangle(disp, (drx, dry), (drx+drw, dry+drh), bbox_col, 2)
        cv2.putText(disp, f"{rw}x{rh}px",
                    (drx, max(dry-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_col, 1)
        cv2.circle(disp, (roi_cx_disp, roi_cy_disp), 5, bbox_col, -1)

        sx_sign = "+" if offset_x >= 0 else ""
        sy_sign = "+" if offset_y >= 0 else ""
        cv2.putText(disp,
                    f"X{sx_sign}{offset_x} Y{sy_sign}{offset_y}px",
                    (drx, dry + drh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 60) if in_zone else (180, 180, 180), 1)

        if not trigger_ok:
            cv2.putText(disp, "too small",
                        (drx, dry + drh + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 140, 255), 1)
        elif not sharp:
            cv2.putText(disp, "blurry",
                        (drx, dry + drh + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 140, 255), 1)

        if drw > 0 and drh > 0:
            crop_disp = disp[dry:dry+drh, drx:drx+drw]
            if crop_disp.size > 0:
                hsv_c       = cv2.cvtColor(crop_disp, cv2.COLOR_BGR2HSV)
                banana_mask = _segment_banana(hsv_c)
                defect_mask = _detect_defects(hsv_c, banana_mask)
                crop_disp[defect_mask > 0] = DEF_COL
                disp[dry:dry+drh, drx:drx+drw] = crop_disp
                def_px = int(np.sum(defect_mask > 0))
                ban_px = int(np.sum(banana_mask > 0))
                ratio  = def_px / ban_px * 100 if ban_px > 0 else 0.0
                cv2.putText(disp, f"HSV {ratio:.1f}%",
                            (drx+2, dry+drh-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)
    else:
        cv2.putText(disp, "no banana",
                    (w_px-80, h_px-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    if _ml_ready:
        badge_txt = f"ML {_ml_info.get('accuracy_pct','?')}%"
        badge_col = (0, 220, 100)
    else:
        badge_txt = "HSV-fallback"
        badge_col = (0, 160, 255)
    cv2.putText(disp, badge_txt, (w_px - 115, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_col, 1)

    if grade_label in ("GOOD", "BAD"):
        col = (0, 210, 0) if grade_label == "GOOD" else (0, 0, 220)
        cv2.putText(disp, grade_label,
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)

    return disp


# =========================================
# MANUAL OVERRIDE BUTTONS
# =========================================
def _flash_button(btn: tk.Button, normal_text: str):
    btn.config(bg="#00c853", text="CAPTURED!", fg="#000000")
    btn.after(700, lambda: btn.config(text=normal_text,
                                       bg="#333333", fg="#888888"))


def on_manual_cam1():
    st = state_read()
    if st["cam1_busy"] or st["frame1"] is None:
        return
    state_write(cam1_busy=True)
    snapshot = st["frame1"].copy()
    print(f"\n[MANUAL] Cam1 override capture")
    _flash_button(btn_cam1, "MANUAL CAM 1")
    _dispatch_classify(1, snapshot, "Cam1")


def on_manual_cam2():
    st = state_read()
    if st["cam2_busy"] or st["frame2"] is None:
        return
    state_write(cam2_busy=True)
    snapshot = st["frame2"].copy()
    print(f"\n[MANUAL] Cam2 override capture")
    _flash_button(btn_cam2, "MANUAL CAM 2")
    _dispatch_classify(2, snapshot, "Cam2")


# =========================================
# HARDWARE BUTTON WATCHER  (GPIO 5 / GPIO 6)
# =========================================
_btn_hold_start: dict = {SHUTDOWN_PIN: None, REBOOT_PIN: None}


def _hardware_teardown():
    global _running
    _running = False
    _stop_event.set()
    time.sleep(0.3)
    _servo_home()
    lgpio.tx_pwm(h, SERVO_PIN, 0, 0)
    lgpio.gpiochip_close(h)
    cam1.release()
    cam2.release()
    try:
        root.after(0, root.destroy)
    except Exception:
        pass


def _do_shutdown():
    print("\n[BUTTON] SHUTDOWN triggered -- cleaning up ...")
    push_device_status("SHUTTING DOWN")
    _hardware_teardown()
    print("[BUTTON] Issuing: sudo shutdown -h now")
    subprocess.run(["sudo", "shutdown", "-h", "now"])


def _do_reboot():
    print("\n[BUTTON] REBOOT triggered -- cleaning up ...")
    push_device_status("REBOOTING")
    _hardware_teardown()
    print("[BUTTON] Issuing: sudo reboot")
    subprocess.run(["sudo", "reboot"])


def hardware_button_watcher():
    poll_s     = BUTTON_POLL_MS / 1000.0
    action_map = {
        SHUTDOWN_PIN: ("SHUTTING DOWN", _do_shutdown),
        REBOOT_PIN:   ("REBOOTING",     _do_reboot),
    }
    fired = {SHUTDOWN_PIN: False, REBOOT_PIN: False}

    while _running:
        for pin, (label, action_fn) in action_map.items():
            try:
                level = lgpio.gpio_read(h, pin)
            except Exception:
                return

            if level == 0:
                if _btn_hold_start[pin] is None:
                    _btn_hold_start[pin] = time.time()
                    print(f"[BTN] GPIO{pin} pressed -- "
                          f"hold {BUTTON_HOLD_S}s to {label}")

                held = time.time() - _btn_hold_start[pin]
                pct  = min(100, int(held / BUTTON_HOLD_S * 100))
                state_write(btn_hold_pct=pct,
                            system_action=label if pct < 100 else "")

                if held >= BUTTON_HOLD_S and not fired[pin]:
                    fired[pin] = True
                    state_write(system_action=label, btn_hold_pct=100)
                    threading.Thread(target=action_fn, daemon=False).start()
                    return
            else:
                if _btn_hold_start[pin] is not None:
                    held = time.time() - _btn_hold_start[pin]
                    print(f"[BTN] GPIO{pin} released after {held:.1f}s "
                          f"(no action)")
                _btn_hold_start[pin] = None
                fired[pin]           = False
                st = state_read()
                if st.get("system_action") in (action_map[pin][0], ""):
                    state_write(btn_hold_pct=0, system_action="")

        time.sleep(poll_s)


# =========================================
# GUI LAYOUT
# =========================================
root = tk.Tk()
root.title("Banana Sorting -- Vision Trigger + Local ML")
root.configure(bg="#1a1a1a")
root.resizable(False, False)

title_label = tk.Label(
    root,
    text="BANANA SORTING SYSTEM  --  Loading model...",
    font=("Courier", 13, "bold"),
    bg="#1a1a1a", fg="#f0c040",
)
title_label.pack(pady=(10, 2))

tk.Label(
    root,
    text=(f"Cam {SCAN_W}x{SCAN_H}  |  Disp {DISP_W}x{DISP_H}  |  "
          f"{CONVEYOR_SPEED} m/s  |  AND gate  |  "
          f"tol={CENTER_TOL_PX}px  cd={VIS_COOLDOWN}s  |  "
          f"GPIO{SERVO_PIN} servo  |  min={MIN_SIZE_CM}cm  |  "
          f"CLAHE + bg-mask + sharp({SHARPNESS_THRESHOLD}) + "
          f"conf({CONFIDENCE_THRESHOLD})  |  "
          f"sustain={SERVO_SIGNAL_SUSTAIN_S}s"),
    font=("Courier", 8),
    bg="#1a1a1a", fg="#555555",
).pack(pady=(0, 6))

feed_row = tk.Frame(root, bg="#1a1a1a")
feed_row.pack(padx=10, pady=4)


def make_cam_panel(parent, header: str, col: int):
    outer = tk.Frame(parent, bg="#2a2a2a", bd=2, relief="solid")
    outer.grid(row=0, column=col, padx=8)

    tk.Label(outer, text=header,
             font=("Courier", 9, "bold"),
             bg="#2a2a2a", fg="#00e676").pack(pady=(6, 2))

    img_lbl = tk.Label(outer, bg="#000000", width=DISP_W, height=DISP_H)
    img_lbl.pack()

    vis_lbl = tk.Label(outer, text="offset: -- px  ARMED",
                       font=("Courier", 8, "bold"),
                       bg="#2a2a2a", fg="#00bcd4")
    vis_lbl.pack(pady=(4, 1))

    status_lbl = tk.Label(outer, text="Waiting for banana...",
                           font=("Courier", 8), bg="#2a2a2a", fg="#888888")
    status_lbl.pack(pady=(1, 1))

    ml_lbl = tk.Label(outer, text="",
                       font=("Courier", 8, "bold"),
                       bg="#2a2a2a", fg="#00e676")
    ml_lbl.pack(pady=(0, 4))

    return img_lbl, vis_lbl, status_lbl, ml_lbl


(cam1_img, cam1_vis, cam1_status, cam1_ml) = make_cam_panel(
    feed_row, "CAM 1  --  Zone A", 0)
(cam2_img, cam2_vis, cam2_status, cam2_ml) = make_cam_panel(
    feed_row, "CAM 2  --  Zone B", 1)

btn_row = tk.Frame(root, bg="#1a1a1a")
btn_row.pack(pady=(2, 6))

MANUAL_BTN = dict(
    font=("Courier", 9),
    bg="#333333", fg="#888888",
    activebackground="#444444", activeforeground="#cccccc",
    relief="flat", cursor="hand2",
    padx=18, pady=7, bd=0,
)

tk.Label(btn_row,
         text="Manual override (force capture on demand):",
         font=("Courier", 8), bg="#1a1a1a", fg="#555555").grid(
    row=0, column=0, columnspan=2, pady=(0, 3))

btn_cam1 = tk.Button(btn_row, text="MANUAL CAM 1",
                      command=on_manual_cam1, **MANUAL_BTN)
btn_cam1.grid(row=1, column=0, padx=(8, 30))

btn_cam2 = tk.Button(btn_row, text="MANUAL CAM 2",
                      command=on_manual_cam2, **MANUAL_BTN)
btn_cam2.grid(row=1, column=1, padx=(30, 8))

# =========================================
# STATUS BAR
# =========================================
sbar = tk.Frame(root, bg="#222222", pady=6)
sbar.pack(fill="x", padx=10, pady=(4, 2))

pipeline_lbl = tk.Label(sbar, text="Pipeline: 0",
                          font=("Courier", 10), bg="#222222", fg="#aaaaaa")
pipeline_lbl.pack(side="left", padx=12)

final_lbl = tk.Label(sbar, text="Last: --",
                      font=("Courier", 12, "bold"),
                      bg="#222222", fg="#ffffff")
final_lbl.pack(side="left", padx=20)

id_lbl = tk.Label(sbar, text="ID: --",
                   font=("Courier", 9), bg="#222222", fg="#666666")
id_lbl.pack(side="right", padx=12)

_hold_bar_frame = tk.Frame(root, bg="#1a1a1a")
_hold_bar_frame.pack(fill="x", padx=10, pady=(0, 2))

_hold_action_lbl = tk.Label(
    _hold_bar_frame, text="",
    font=("Courier", 9, "bold"),
    bg="#1a1a1a", fg="#ff9800",
    width=20, anchor="w",
)
_hold_action_lbl.pack(side="left", padx=(4, 6))

_hold_canvas = tk.Canvas(
    _hold_bar_frame,
    height=14, bg="#2a2a2a",
    bd=0, highlightthickness=0,
)
_hold_canvas.pack(side="left", fill="x", expand=True, padx=(0, 8))

_hold_hint_lbl = tk.Label(
    _hold_bar_frame,
    text=(f"GPIO{SHUTDOWN_PIN}=shutdown  GPIO{REBOOT_PIN}=reboot  "
          f"(hold {BUTTON_HOLD_S:.0f}s)"),
    font=("Courier", 8),
    bg="#1a1a1a", fg="#444444",
)
_hold_hint_lbl.pack(side="right", padx=4)

# =========================================
# RESULTS TABLE
# =========================================
tbl_wrap = tk.Frame(root, bg="#1a1a1a")
tbl_wrap.pack(padx=10, pady=(4, 10), fill="x")

cols   = ("Time", "Banana ID", "cm", "Size OK", "Defect%", "Cam1", "Cam2", "FINAL")
cw_map = {"Time": 145, "Banana ID": 145, "cm": 55, "Size OK": 65,
           "Defect%": 65, "Cam1": 55, "Cam2": 55, "FINAL": 65}

tbl = ttk.Treeview(tbl_wrap, columns=cols, show="headings", height=8)
for c in cols:
    tbl.heading(c, text=c)
    tbl.column(c, width=cw_map[c], anchor="center")
tbl.tag_configure("good", foreground="#00e676")
tbl.tag_configure("bad",  foreground="#ff5252")

sty = ttk.Style()
sty.theme_use("default")
sty.configure("Treeview",
    background="#1e1e1e", foreground="#dddddd",
    fieldbackground="#1e1e1e", rowheight=22, font=("Courier", 9))
sty.configure("Treeview.Heading",
    background="#2a2a2a", foreground="#f0c040", font=("Courier", 9, "bold"))

sb = ttk.Scrollbar(tbl_wrap, orient="vertical", command=tbl.yview)
tbl.configure(yscrollcommand=sb.set)
tbl.pack(side="left", fill="x", expand=True)
sb.pack(side="right", fill="y")


# =========================================
# GUI REFRESH  (~12 fps)
# =========================================
_seen_result_ids: set = set()
_last_hold_pct: int   = -1


def _redraw_hold_bar(pct: int, action: str):
    global _last_hold_pct
    if pct == _last_hold_pct:
        return
    _last_hold_pct = pct

    _hold_canvas.update_idletasks()
    total_w = _hold_canvas.winfo_width()
    if total_w < 2:
        return

    _hold_canvas.delete("all")
    if pct <= 0 or not action:
        _hold_action_lbl.config(text="")
        return

    fill_w  = max(1, int(total_w * pct / 100))
    bar_col = "#ff5252" if pct >= 90 else "#ff9800"
    _hold_canvas.create_rectangle(0, 0, fill_w, 14, fill=bar_col, outline="")
    _hold_canvas.create_rectangle(fill_w, 0, total_w, 14,
                                   fill="#2a2a2a", outline="")
    _hold_canvas.create_text(
        total_w // 2, 7,
        text=f"  {action} ({pct}%)",
        fill="#ffffff", font=("Courier", 8, "bold"), anchor="center",
    )
    _hold_action_lbl.config(text=action)


def gui_refresh():
    st = state_read()

    vis1_off   = st.get("vis1_offset", 0)
    vis2_off   = st.get("vis2_offset", 0)
    vis1_state = st.get("vis1_state", "ARMED")
    vis2_state = st.get("vis2_state", "ARMED")

    ov1 = st.get("overlay1")
    if ov1 is not None:
        i1 = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(ov1, cv2.COLOR_BGR2RGB)))
        cam1_img.imgtk = i1; cam1_img.configure(image=i1)
        cam1_status.config(text=f"Last scan: {st['last_grade1'] or '--'}")
        cam1_ml.config(text=st.get("ml_status1", ""))

    ov2 = st.get("overlay2")
    if ov2 is not None:
        i2 = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB)))
        cam2_img.imgtk = i2; cam2_img.configure(image=i2)
        cam2_status.config(text=f"Last scan: {st['last_grade2'] or '--'}")
        cam2_ml.config(text=st.get("ml_status2", ""))

    def _vis_color(vstate):
        return {"TRIGGERED": "#ff9800",
                "CENTRED":   "#00e676",
                "BLURRY":    "#ff5252"}.get(vstate, "#00bcd4")

    sign1 = "+" if vis1_off >= 0 else ""
    sign2 = "+" if vis2_off >= 0 else ""
    cam1_vis.config(text=f"offset: {sign1}{vis1_off}px  {vis1_state}",
                    fg=_vis_color(vis1_state))
    cam2_vis.config(text=f"offset: {sign2}{vis2_off}px  {vis2_state}",
                    fg=_vis_color(vis2_state))

    for btn, busy_key, normal in (
            (btn_cam1, "cam1_busy", "MANUAL CAM 1"),
            (btn_cam2, "cam2_busy", "MANUAL CAM 2")):
        busy = st[busy_key]
        btn.config(
            state="disabled" if busy else "normal",
            bg="#252525"     if busy else "#333333",
            fg="#555555"     if busy else "#888888",
            text="CLASSIFYING..." if busy else normal,
        )

    pipeline_lbl.config(
        text=f"Pipeline: {st['pipeline_count']} hook(s) in flight")

    final = st["last_final"]
    if final == "GOOD":
        final_lbl.config(text="Last:  GOOD", fg="#00e676")
    elif final == "BAD":
        final_lbl.config(text="Last:  BAD",  fg="#ff5252")
    else:
        final_lbl.config(text="Last: --", fg="#ffffff")

    id_lbl.config(text=f"ID: {st['last_banana_id'] or '--'}")

    _redraw_hold_bar(st.get("btn_hold_pct", 0), st.get("system_action", ""))

    for r in st["results"]:
        if r["id"] not in _seen_result_ids:
            _seen_result_ids.add(r["id"])
            tag = "good" if r["grade"] == "GOOD" else "bad"
            tbl.insert("", 0, values=(
                r["time"], r["id"], r["size"], r["size_ok"],
                r["defect"], r["c1"], r["c2"], r["grade"],
            ), tags=(tag,))
            _bid = r["id"]
            root.after(2000, lambda bid=_bid: _clear_grade_if_stale(bid))

    root.after(80, gui_refresh)


def _clear_grade_if_stale(banana_id: str):
    st = state_read()
    if st.get("last_banana_id") == banana_id:
        state_write(last_grade1="", last_grade2="")


# =========================================
# STARTUP / SHUTDOWN
# =========================================
def start_background_threads():
    load_local_model()
    recover_dlq_on_startup()

    threading.Thread(target=camera_loop,             daemon=True).start()
    threading.Thread(target=visual_center_watcher,   daemon=True).start()
    threading.Thread(target=timeout_watcher,         daemon=True).start()
    threading.Thread(target=hardware_button_watcher, daemon=True).start()

    push_device_status("ACTIVE")
    mode = "ML (local SVM)" if _ml_ready else "HSV fallback"
    print(f"[System] Ready -- classifier: {mode}")
    print(f"[System] Lighting: co-axial LED strip, white fabric background, "
          f"no shadows")
    print(f"[System] Active preprocessing: CLAHE, background masking, "
          f"sharpness gate (thr={SHARPNESS_THRESHOLD}), "
          f"confidence gate (thr={CONFIDENCE_THRESHOLD})")
    print(f"[System] Camera: exposure locked, WB locked @ 4500K")
    print(f"[System] Vision trigger: centre_tol={CENTER_TOL_PX}px  "
          f"cooldown={VIS_COOLDOWN}s")
    print(f"[System] Sorter servo=GPIO{SERVO_PIN}  "
          f"HOME={SERVO_HOME_DEG}deg(500us)  PUSH={SERVO_PUSH_DEG}deg(2500us)  "
          f"delay={SERVO_DELAY_S}s  hold={SERVO_HOLD_S}s  "
          f"sustain={SERVO_SIGNAL_SUSTAIN_S}s")
    print(f"[System] Servo direction: HOME=0deg(retracted) -> "
          f"PUSH=180deg(extended/eject) -> HOME=0deg(retract)")
    print(f"[System] Size rejection threshold={MIN_SIZE_CM}cm")
    print(f"[System] HW buttons: GPIO{SHUTDOWN_PIN}=shutdown  "
          f"GPIO{REBOOT_PIN}=reboot  (hold {BUTTON_HOLD_S}s)")
    print(f"[System] Firestore: persistent upload queue active  "
          f"(retries={UPLOAD_RETRIES}  back-off={UPLOAD_BACKOFF}s  "
          f"DLQ={DLQ_FILE.name})")
    print(f"[System] DLQ auto-recovery: ON  (runs every startup)")
    print(f"[System] ---- LATENCY FIXES ACTIVE ----")
    print(f"[System] IMG_SIZE={IMG_SIZE}px  VIS_COOLDOWN={VIS_COOLDOWN}s  "
          f"SCAN_TIMEOUT={SCAN_TIMEOUT}s")
    print(f"[System] Classification timing prints ON  "
          f"(watch console for [TIMING] lines)")


def on_close():
    global _running
    _running = False
    _stop_event.set()

    def _cleanup():
        push_device_status("OFFLINE")
        _servo_home()
        lgpio.tx_pwm(h, SERVO_PIN, 0, 0)
        lgpio.gpiochip_close(h)
        cam1.release()
        cam2.release()
        print("[System] Shutdown complete.")

    try:
        root.withdraw()
    except Exception:
        pass

    t = threading.Thread(target=_cleanup, daemon=True)
    t.start()
    t.join(timeout=3.0)

    try:
        root.destroy()
    except Exception:
        pass


root.protocol("WM_DELETE_WINDOW", on_close)
root.after(500, start_background_threads)
root.after(600, gui_refresh)
root.mainloop()
