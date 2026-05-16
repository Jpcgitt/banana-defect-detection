"""
Banana Sorting System — Flask Live Feed Edition
================================================
Tkinter GUI removed. Flask serves a local live-feed web UI.

Architecture
------------
  Firebase (cloud)  : Dashboard, Report, About — accessible anywhere
  Flask   (local)   : Live Feed at http://<rpi-ip>:5000 — same-WiFi only

Flask endpoints
---------------
  GET  /                  → live feed HTML page
  GET  /video_feed/1      → MJPEG stream, Camera 1
  GET  /video_feed/2      → MJPEG stream, Camera 2
  GET  /events            → Server-Sent Events (real-time state)
  POST /capture/<1|2>     → manual capture trigger
  POST /system/shutdown   → clean shutdown
  POST /system/reboot     → clean reboot
  GET  /health            → JSON health check

All banana-sorting logic (GPIO, cameras, classification, servo,
Firestore upload) is unchanged from the tkinter version.
"""

import lgpio
import cv2
import numpy as np
import time
import threading
import pickle
import json
import signal
import sys
import subprocess
import collections
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, render_template, jsonify, request, stream_with_context

import firebase_admin
from firebase_admin import credentials, firestore

# ── Firebase-hosted site URL (update if your hosting URL differs) ──────────
FIREBASE_URL = "https://banana-sorting-5f22d.web.app"

# =========================================
# FIREBASE
# =========================================
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
# SANITISER
# =========================================
def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
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
            print(f"[Firestore] ✗ Failed ({attempts}/{UPLOAD_RETRIES}): {doc_id} -- {e}")
            if attempts < UPLOAD_RETRIES:
                item["_attempts"] = attempts
                delay = UPLOAD_BACKOFF[min(attempts - 1, len(UPLOAD_BACKOFF) - 1)]
                def _requeue(it=item, d=delay):
                    time.sleep(d)
                    with _upload_lock:
                        _upload_queue.appendleft(it)
                threading.Thread(target=_requeue, daemon=True).start()
            else:
                print(f"[Firestore] ✗✗ Giving up on {doc_id} — saving to DLQ")
                try:
                    with open(DLQ_FILE, "a") as f:
                        json.dump(_sanitize(item), f); f.write("\n")
                except Exception as fe:
                    print(f"[Firestore] DLQ write error: {fe}")


threading.Thread(target=_upload_worker, daemon=True, name="FirestoreWorker").start()


def upload_to_firestore(data: dict):
    clean = _sanitize(data)
    clean.setdefault("_attempts", 0)
    with _upload_lock:
        _upload_queue.append(clean)
    print(f"[Firestore] Queued: {clean['banana_id']}  (depth: {len(_upload_queue)})")


# =========================================
# STARTUP DLQ RECOVERY
# =========================================
def recover_dlq_on_startup():
    if not DLQ_FILE.exists():
        print("[DLQ Recovery] No upload_failures.json — nothing to recover.")
        return

    dlq_records = []
    with open(DLQ_FILE) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw: continue
            try:
                dlq_records.append(json.loads(raw))
            except json.JSONDecodeError as e:
                print(f"[DLQ Recovery] Line {lineno} parse error: {e} — skipped")

    if not dlq_records:
        print("[DLQ Recovery] DLQ is empty.")
        return

    print(f"[DLQ Recovery] {len(dlq_records)} record(s) found.")
    by_id = {r.get("banana_id"): r for r in dlq_records if r.get("banana_id")}
    all_ids = list(by_id.keys())
    already_uploaded = set()

    try:
        for i in range(0, len(all_ids), 30):
            chunk = all_ids[i: i + 30]
            snap  = db.collection("banana_records").where("banana_id", "in", chunk).get()
            for doc in snap:
                already_uploaded.add(doc.get("banana_id"))
    except Exception as e:
        print(f"[DLQ Recovery] Firestore check failed: {e} — re-queuing all")

    to_requeue = [r for bid, r in by_id.items() if bid not in already_uploaded]
    for r in to_requeue:
        r["_attempts"] = 0
        with _upload_lock:
            _upload_queue.appendleft(_sanitize(r))

    requeued_ids  = {r["banana_id"] for r in to_requeue}
    remaining_dlq = [r for r in dlq_records
                     if r.get("banana_id") not in requeued_ids
                     and r.get("banana_id") not in already_uploaded]
    try:
        with open(DLQ_FILE, "w") as f:
            for r in remaining_dlq:
                json.dump(r, f); f.write("\n")
    except Exception as e:
        print(f"[DLQ Recovery] Could not rewrite DLQ: {e}")

    print(f"[DLQ Recovery] ✓ {len(to_requeue)} re-queued, "
          f"{len(already_uploaded)} already uploaded.")


# =========================================
# PATHS
# =========================================
BASE_DIR  = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "banana_pictures"
LOG_FILE  = BASE_DIR / "banana_log.json"
IMAGE_DIR.mkdir(exist_ok=True)

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
# CLAHE
# =========================================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# =========================================
# SHARPNESS GATE
# =========================================
SHARPNESS_THRESHOLD = 60.0


def _is_sharp(frame: np.ndarray, roi: tuple) -> bool:
    if roi is None: return False
    rx, ry, rw, rh = roi
    crop = frame[ry:ry + rh, rx:rx + rw]
    if crop.size == 0: return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= SHARPNESS_THRESHOLD


# =========================================
# LOCAL ML MODEL
# =========================================
_ml_model   = None
_ml_encoder = None
_ml_ready   = False
_ml_info    = {}

IMG_SIZE   = 48
HOG_CELL   = (8, 8)
HOG_BLOCK  = (16, 16)
HOG_STRIDE = (8, 8)
HOG_BINS   = 9
HIST_BINS  = 32

_HOG = cv2.HOGDescriptor(
    (IMG_SIZE, IMG_SIZE), HOG_BLOCK, HOG_STRIDE, HOG_CELL, HOG_BINS,
)


def load_local_model():
    global _ml_model, _ml_encoder, _ml_ready, _ml_info
    if not MODEL_PATH.exists():
        print(f"[ML] WARNING: {MODEL_PATH.name} not found — HSV fallback mode.")
        return
    try:
        with open(MODEL_PATH, "rb") as f:
            _ml_model = pickle.load(f)
        print(f"[ML] Loaded {MODEL_PATH.name}")
    except Exception as e:
        print(f"[ML] Failed to load model: {e}"); return

    if not ENCODER_PATH.exists():
        print(f"[ML] {ENCODER_PATH.name} not found"); return
    try:
        with open(ENCODER_PATH, "rb") as f:
            _ml_encoder = pickle.load(f)
        print(f"[ML] Loaded {ENCODER_PATH.name}  classes={list(_ml_encoder.classes_)}")
    except Exception as e:
        print(f"[ML] Failed to load encoder: {e}"); return

    if INFO_PATH.exists():
        try:
            with open(INFO_PATH) as f:
                _ml_info = json.load(f)
            print(f"[ML]    Accuracy={_ml_info.get('accuracy_pct','?')}%  "
                  f"Trained={_ml_info.get('trained_at','?')}")
        except Exception:
            pass

    _ml_ready = True
    mode = "ML (local SVM)" if _ml_ready else "HSV fallback"
    print(f"[ML] Classifier ready: {mode}")


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = _apply_clahe(img_bgr)
    img     = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_vec = _HOG.compute(gray).flatten()
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist  = cv2.calcHist([hsv], [0], None, [HIST_BINS], [0, 180]).flatten()
    s_hist  = cv2.calcHist([hsv], [1], None, [HIST_BINS], [0, 256]).flatten()
    v_hist  = cv2.calcHist([hsv], [2], None, [HIST_BINS], [0, 256]).flatten()
    colour_vec = np.concatenate([h_hist, s_hist, v_hist])
    total = colour_vec.sum()
    if total > 0: colour_vec /= total
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
_today_str, _banana_counter = load_counter()
print(f"[Counter] Start at {_banana_counter} for {_today_str}")


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


# =========================================
# GPIO SETUP
# =========================================
SERVO_PIN            = 25
PWM_FREQ             = 50
SERVO_HOME_DEG       = 0
SERVO_PUSH_DEG       = 180
SERVO_SIGNAL_SUSTAIN_S = 0.5

SHUTDOWN_PIN  = 5
REBOOT_PIN    = 6
BUTTON_HOLD_S = 5.0
BUTTON_POLL_MS = 50

_stop_event = threading.Event()

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, SERVO_PIN, 0)


def _angle_to_duty(pulse_us: float) -> float:
    return (pulse_us / 20000.0) * 100.0


def _set_servo_angle(deg: float):
    pulse_us = 500.0 + (deg / 180.0) * 2000.0
    lgpio.tx_pwm(h, SERVO_PIN, PWM_FREQ, _angle_to_duty(pulse_us))
    time.sleep(SERVO_SIGNAL_SUSTAIN_S)
    lgpio.tx_pwm(h, SERVO_PIN, 0, 0)


def _servo_home():
    _set_servo_angle(SERVO_HOME_DEG)


_set_servo_angle(SERVO_HOME_DEG)
print(f"[GPIO] Servo homed to {SERVO_HOME_DEG}° at startup.")

lgpio.gpio_claim_input(h, SHUTDOWN_PIN, lgpio.SET_PULL_UP)
lgpio.gpio_claim_input(h, REBOOT_PIN,   lgpio.SET_PULL_UP)


# =========================================
# TIMING CONSTANTS
# =========================================
CONVEYOR_SPEED  = 0.15
SERVO_DELAY_S   = 2.0
SERVO_HOLD_S    = 5.0
SCAN_TIMEOUT    = 4.0
FULL_CYCLE_TIME = SERVO_DELAY_S + SERVO_HOLD_S + 5.0
SAME_BANANA_WINDOW = 8.0

# =========================================
# VISUAL TRIGGER PARAMETERS
# =========================================
CENTER_TOL_PX   = 40
CENTER_TOL_Y_PX = 80
MIN_TRIGGER_PX  = 8000
VIS_COOLDOWN    = 1.0

# =========================================
# SIZE REJECTION
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
SEG_GREEN_LOW    = np.array([ 22,  55,  60])
SEG_GREEN_HIGH   = np.array([ 88, 255, 255])
SEG_YELLOW_LOW   = np.array([ 18,  70,  90])
SEG_YELLOW_HIGH  = np.array([ 32, 255, 255])

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
    "frame1": None, "frame2": None,
    "overlay1": None, "overlay2": None,
    "last_grade1": "", "last_grade2": "",
    "last_final": "", "last_banana_id": "",
    "pipeline_count": 0,
    "results": [],
    "cam1_busy": False, "cam2_busy": False,
    "ml_status1": "", "ml_status2": "",
    "vis1_offset": 0, "vis2_offset": 0,
    "vis1_state": "ARMED", "vis2_state": "ARMED",
    "system_action": "", "btn_hold_pct": 0,
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
    cnts, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area: return None
    x, y, w, hh = cv2.boundingRect(best)
    pad = 10
    x  = max(0, x - pad);  y  = max(0, y - pad)
    w  = min(frame.shape[1] - x, w + 2 * pad)
    hh = min(frame.shape[0] - y, hh + 2 * pad)
    return x, y, w, hh


# =========================================
# HSV HELPERS
# =========================================
def _segment_banana(hsv: np.ndarray) -> np.ndarray:
    gm   = cv2.inRange(hsv, SEG_GREEN_LOW,  SEG_GREEN_HIGH)
    ym   = cv2.inRange(hsv, SEG_YELLOW_LOW, SEG_YELLOW_HIGH)
    mask = cv2.bitwise_or(gm, ym)
    k11  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  k11, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k11, iterations=1)
    return mask


def _detect_defects(hsv: np.ndarray, banana_mask: np.ndarray) -> np.ndarray:
    dark_mask     = cv2.inRange(hsv, np.array([0,0,0]),                  np.array([180,255,DARK_V_MAX]))
    brown_mask    = cv2.inRange(hsv, BROWN_HUE_LOW,    BROWN_HUE_HIGH)
    overripe_mask = cv2.inRange(hsv, OVERRIPE_HUE_LOW, OVERRIPE_HUE_HIGH)
    scratch_mask  = cv2.inRange(hsv, np.array([0,0,SCRATCH_V_MIN]),       np.array([180,SCRATCH_S_MAX,SCRATCH_V_MAX]))
    combined  = cv2.bitwise_or(dark_mask, brown_mask)
    combined  = cv2.bitwise_or(combined,  overripe_mask)
    combined  = cv2.bitwise_or(combined,  scratch_mask)
    on_banana = cv2.bitwise_and(combined, banana_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    on_banana = cv2.morphologyEx(on_banana, cv2.MORPH_OPEN, k3)
    cleaned = np.zeros_like(on_banana)
    cnts, _ = cv2.findContours(on_banana, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_DEFECT_CONTOUR_PX:
            cv2.drawContours(cleaned, [c], -1, 255, -1)
    return cleaned


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
# CLASSIFICATION
# =========================================
def classify_banana_hsv(frame: np.ndarray, cam_label: str = "") -> tuple:
    frame = _apply_clahe(frame)
    roi   = find_banana_roi(frame)
    if roi is None:
        return "EMPTY", 0.0, {"method": "HSV", "roi": None}
    crop, banana_mask = _masked_crop(frame, roi)
    hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    banana_px  = int(np.sum(banana_mask > 0))
    if banana_px < MIN_BANANA_PX:
        return "EMPTY", 0.0, {"method": "HSV", "banana_px": banana_px, "roi": list(roi)}
    defect_mask  = _detect_defects(hsv, banana_mask)
    defect_px    = int(np.sum(defect_mask > 0))
    defect_ratio = defect_px / banana_px
    defect_pct   = round(defect_ratio * 100, 2)
    grade        = "BAD" if defect_ratio >= DEFECT_RATIO_MAX else "GOOD"
    print(f"  [{cam_label}][HSV] {grade}  defect={defect_pct}%")
    return grade, defect_pct, {
        "method": "HSV", "banana_px": banana_px,
        "defect_px": defect_px, "defect_ratio": round(defect_ratio, 4),
        "defect_pct": defect_pct, "roi": list(roi),
    }


def classify_banana_local(frame: np.ndarray, cam_label: str = "") -> tuple:
    if not _ml_ready:
        return classify_banana_hsv(frame, cam_label)
    frame = _apply_clahe(frame)
    roi   = find_banana_roi(frame)
    if roi is None:
        return "EMPTY", 0.0, {"method": "ML", "roi": None}
    crop, banana_mask = _masked_crop(frame, roi)
    banana_px = int(np.sum(banana_mask > 0))
    if banana_px < MIN_BANANA_PX:
        return "EMPTY", 0.0, {"method": "ML", "banana_px": banana_px, "roi": list(roi)}
    feat = extract_features(crop)
    try:
        proba      = _ml_model.predict_proba(feat)[0]
        pred_idx   = int(np.argmax(proba))
        pred_label = _ml_encoder.classes_[pred_idx].upper()
        confidence = float(proba[pred_idx])
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"  [{cam_label}][ML] Low conf ({confidence:.2f}) → HSV fallback")
            return classify_banana_hsv(frame, cam_label)
        classes_list = list(_ml_encoder.classes_)
        if "bad" in classes_list:
            bad_idx  = classes_list.index("bad")
            bad_prob = float(proba[bad_idx])
        else:
            bad_prob = 1.0 - confidence if pred_label == "GOOD" else confidence
        defect_pct = round(bad_prob * 100, 2)
    except Exception as e:
        print(f"  [{cam_label}][ML] Predict error: {e} → HSV fallback")
        return classify_banana_hsv(frame, cam_label)
    grade = pred_label
    print(f"  [{cam_label}][ML] {grade}  bad_prob={bad_prob:.3f}  conf={confidence:.3f}")
    return grade, defect_pct, {
        "method": "ML", "banana_px": banana_px,
        "defect_pct": defect_pct, "bad_prob": round(bad_prob, 4),
        "confidence": round(confidence, 4), "roi": list(roi),
    }


def classify_banana(frame: np.ndarray, cam_label: str = "") -> tuple:
    return classify_banana_local(frame, cam_label) if _ml_ready else classify_banana_hsv(frame, cam_label)


def estimate_size(frame) -> float:
    roi = find_banana_roi(frame)
    if roi is None: return 0.0
    _, _, w, hh = roi
    return round(max(w, hh) * CM_PER_PIXEL, 2)


def save_banana_data(data: dict):
    try:
        clean = _sanitize(data)
        with open(LOG_FILE, "a") as f:
            json.dump(clean, f); f.write("\n")
    except Exception as e:
        print(f"[Log] {e}")


# =========================================
# SORTER SERVO
# =========================================
_servo_lock = threading.Lock()


def fire_servo_bad():
    def _fire():
        with _servo_lock:
            print(f"  [Servo] BAD — waiting {SERVO_DELAY_S}s ...")
            time.sleep(SERVO_DELAY_S)
            print(f"  [Servo] EXTEND -> {SERVO_PUSH_DEG}°")
            _set_servo_angle(SERVO_PUSH_DEG)
            print(f"  [Servo] Holding {SERVO_HOLD_S}s ...")
            time.sleep(SERVO_HOLD_S)
            print(f"  [Servo] RETRACT HOME -> {SERVO_HOME_DEG}°")
            _servo_home()
    threading.Thread(target=_fire, daemon=True).start()


# =========================================
# PIPELINE
# =========================================
class BananaInFlight:
    _seq = 0; _seq_lock = threading.Lock()

    def __init__(self, t: float):
        with BananaInFlight._seq_lock:
            BananaInFlight._seq += 1; seq = BananaInFlight._seq
        self.label            = f"Banana#{seq}"
        self.scan_start_time  = t
        self.scan1_done       = False; self.scan1_grade = None
        self.scan1_defect     = 0.0;   self.scan1_frame = None; self.scan1_debug = {}
        self.scan2_done       = False; self.scan2_grade = None
        self.scan2_defect     = 0.0;   self.scan2_frame = None; self.scan2_debug = {}
        self.grade_determined = False
        self.banana_data      = None


pipeline       = []
_pipeline_lock = threading.Lock()


# =========================================
# GRADE FINALISATION
# =========================================
def finalise_grade(b: BananaInFlight):
    def effective(g): return "GOOD" if g in ("GOOD", "EMPTY", None) else "BAD"
    g1 = effective(b.scan1_grade if b.scan1_done else b.scan2_grade)
    g2 = effective(b.scan2_grade if b.scan2_done else b.scan1_grade)
    d1 = b.scan1_defect if b.scan1_done else b.scan2_defect
    d2 = b.scan2_defect if b.scan2_done else b.scan1_defect

    final_grade = "GOOD" if (g1 == "GOOD" and g2 == "GOOD") else "BAD"
    defect_pct  = round((d1 + d2) / 2, 2)
    ref         = b.scan1_frame or b.scan2_frame
    sz          = estimate_size(ref) if ref is not None else 0.0
    bid         = make_banana_id()
    ts          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    classifier  = "ML" if _ml_ready else "HSV-fallback"

    size_rejected = False
    if not is_size_acceptable(sz):
        final_grade = "BAD"; size_rejected = True

    b.banana_data = {
        "timestamp": ts, "banana_id": bid, "size_cm": sz,
        "size_rejected": size_rejected, "defect_pct": defect_pct,
        "grade": final_grade, "defect": final_grade == "BAD",
        "cam1_grade": b.scan1_grade, "cam2_grade": b.scan2_grade,
        "cam1_debug": b.scan1_debug, "cam2_debug": b.scan2_debug,
        "classifier": classifier,
    }

    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  {b.label}  ->  {bid}  [{classifier}]")
    print(f"  Cam1 : {b.scan1_grade}  ({b.scan1_defect}%)")
    print(f"  Cam2 : {b.scan2_grade}  ({b.scan2_defect}%)")
    print(f"  FINAL: {final_grade}  |  {defect_pct}%  |  {sz}cm  "
          f"{'REJECTED' if size_rejected else 'OK'}")
    print(bar)

    save_banana_data(b.banana_data)
    for frame, lbl in ((b.scan1_frame, "cam1"), (b.scan2_frame, "cam2")):
        if frame is not None:
            threading.Thread(target=save_image, args=(frame, bid, lbl), daemon=True).start()

    upload_to_firestore(b.banana_data)

    if final_grade == "BAD":
        fire_servo_bad()
    else:
        print("  [GOOD] → servo stays home")

    c1_display = b.scan1_grade if b.scan1_done else (f"~{b.scan2_grade}" if b.scan2_grade else "N/A")
    c2_display = b.scan2_grade if b.scan2_done else (f"~{b.scan1_grade}" if b.scan1_grade else "N/A")

    with _state_lock:
        _shared["last_grade1"]    = str(b.scan1_grade or "")
        _shared["last_grade2"]    = str(b.scan2_grade or "")
        _shared["last_final"]     = final_grade
        _shared["last_banana_id"] = bid
        _shared["cam1_busy"]      = False
        _shared["cam2_busy"]      = False
        _shared["results"].insert(0, {
            "time": ts, "id": bid, "size": sz,
            "size_ok": "NO" if size_rejected else "YES",
            "defect": f"{defect_pct}%", "grade": final_grade,
            "c1": c1_display, "c2": c2_display,
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
            if b.grade_determined: continue
            if (now - b.scan_start_time) > SAME_BANANA_WINDOW: continue
            if cam_id == 1 and b.scan1_frame is None: target = b; break
            if cam_id == 2 and b.scan2_frame is None: target = b; break

        if target is None:
            target = BananaInFlight(now)
            pipeline.append(target)
            state_write(pipeline_count=len(pipeline))
            print(f"  -> new {target.label}  [{len(pipeline)} in pipeline]")
        else:
            print(f"  -> attached to {target.label}")

        if cam_id == 1: target.scan1_frame = frame
        else:           target.scan2_frame = frame

    def _worker():
        t0 = time.time()
        grade, defect, dbg = classify_banana(frame, cam_label)
        elapsed = time.time() - t0
        print(f"  [TIMING] {cam_label} classify_banana took {elapsed:.3f}s")

        with _pipeline_lock:
            if cam_id == 1:
                target.scan1_grade = grade; target.scan1_defect = defect
                target.scan1_debug = dbg;   target.scan1_done   = True
            else:
                target.scan2_grade = grade; target.scan2_defect = defect
                target.scan2_debug = dbg;   target.scan2_done   = True

        method = dbg.get("method", "HSV")
        icon   = "OK" if grade in ("GOOD", "EMPTY") else "!!"
        msg    = f"[{icon}] {grade}  ({defect:.1f}%)  [{method}]"
        if cam_id == 1: state_write(cam1_busy=False, ml_status1=msg)
        else:           state_write(cam2_busy=False, ml_status2=msg)
        _check_finalise(target)

    threading.Thread(target=_worker, daemon=True).start()


def _check_finalise(b: BananaInFlight):
    with _pipeline_lock:
        if b.grade_determined: return
        if b.scan1_done and b.scan2_done:
            b.grade_determined = True
        else:
            return
    threading.Thread(target=finalise_grade, args=(b,), daemon=True).start()


# =========================================
# RUNNING FLAG + CLEANUP
# =========================================
_running    = True
_cleaned_up = False


def cleanup():
    global _running, _cleaned_up
    if _cleaned_up: return
    _cleaned_up = True
    _running    = False
    _stop_event.set()
    time.sleep(0.4)
    try:
        push_device_status("OFFLINE")
    except Exception:
        pass
    try:
        _servo_home()
        lgpio.tx_pwm(h, SERVO_PIN, 0, 0)
        lgpio.gpiochip_close(h)
    except Exception:
        pass
    try:
        cam1.release(); cam2.release()
    except Exception:
        pass
    print("[System] Cleanup complete.")


# =========================================
# TIMEOUT WATCHER
# =========================================
def timeout_watcher():
    while _running:
        now = time.time()
        with _pipeline_lock:
            for b in list(pipeline):
                if b.grade_determined: continue
                elapsed = now - b.scan_start_time
                s1 = b.scan1_frame is not None
                s2 = b.scan2_frame is not None
                if (s1 or s2) and elapsed >= SCAN_TIMEOUT:
                    b.grade_determined = True
                    missed = "Cam2" if (s1 and not s2) else ("Cam1" if (s2 and not s1) else "none")
                    print(f"  [Timeout] {missed} missed — partial grade ({elapsed:.1f}s)")
                    threading.Thread(target=finalise_grade, args=(b,), daemon=True).start()

            alive = [b for b in pipeline
                     if not b.grade_determined or (now - b.scan_start_time) < FULL_CYCLE_TIME]
            if len(alive) != len(pipeline):
                pipeline[:] = alive
                state_write(pipeline_count=len(pipeline))
        time.sleep(0.25)


# =========================================
# VISUAL CENTER WATCHER
# =========================================
def visual_center_watcher():
    armed       = {1: True,  2: True}
    roi_present = {1: False, 2: False}
    last_trigger = {1: 0.0, 2: 0.0}
    frame_cx = SCAN_W // 2; frame_cy = SCAN_H // 2

    cam_configs = [
        (1, "frame1", "cam1_busy", "vis1_offset", "vis1_state", "Cam1"),
        (2, "frame2", "cam2_busy", "vis2_offset", "vis2_state", "Cam2"),
    ]

    while _running:
        st = state_read()
        for cam_id, frame_key, busy_key, offset_key, vstate_key, label in cam_configs:
            frame = st.get(frame_key)
            if frame is None: continue

            roi = find_banana_roi(frame, min_area=MIN_TRIGGER_PX)

            if roi is None:
                if roi_present[cam_id]:
                    armed[cam_id] = True; roi_present[cam_id] = False
                    print(f"  [VIS{cam_id}] Banana left FOV — re-armed")
                state_write(**{offset_key: 0, vstate_key: "ARMED" if armed[cam_id] else "DISARMED"})
                continue

            roi_present[cam_id] = True
            rx, ry, rw, rh = roi
            roi_cx   = rx + rw // 2; roi_cy = ry + rh // 2
            offset_x = roi_cx - frame_cx; offset_y = roi_cy - frame_cy

            now         = time.time()
            centred_x   = abs(offset_x) <= CENTER_TOL_PX
            centred_y   = abs(offset_y) <= CENTER_TOL_Y_PX
            centred     = centred_x and centred_y
            cooldown_ok = (now - last_trigger[cam_id]) >= VIS_COOLDOWN
            busy        = st[busy_key]

            if centred and armed[cam_id] and cooldown_ok and not busy:
                if not _is_sharp(frame, roi):
                    state_write(**{offset_key: offset_x, vstate_key: "BLURRY"})
                    continue
                armed[cam_id] = False; last_trigger[cam_id] = now
                snapshot = frame.copy()
                state_write(**{busy_key: True, offset_key: offset_x, vstate_key: "TRIGGERED"})
                print(f"\n[VIS{cam_id}] {label} centred X={offset_x:+d} Y={offset_y:+d} → capture")
                _dispatch_classify(cam_id, snapshot, label)
            else:
                if not armed[cam_id]:         vstate = "DISARMED"
                elif centred:                 vstate = "CENTRED"
                elif not centred_x:           vstate = f"ARMED X{offset_x:+d}"
                else:                         vstate = f"ARMED Y{offset_y:+d}"
                state_write(**{offset_key: offset_x, vstate_key: vstate})

        time.sleep(0.033)


# =========================================
# CAMERA LOOP
# =========================================
GRID_COLS = 8; GRID_ROWS = 6
GRID_COL  = (50, 50, 50); CROSS_COL  = (0, 180, 255)
BBOX_COL  = (0, 255, 120); DEF_COL   = (0, 60, 255)
CENTRE_COL = (255, 200, 0)


def draw_fov_overlay(frame: np.ndarray, grade_label: str = "",
                     offset_px: int = 0) -> np.ndarray:
    disp       = cv2.resize(frame, (DISP_W, DISP_H))
    h_px, w_px = disp.shape[:2]
    sx = DISP_W / SCAN_W; sy = DISP_H / SCAN_H

    cw, ch = w_px // GRID_COLS, h_px // GRID_ROWS
    for c in range(1, GRID_COLS): cv2.line(disp, (c*cw, 0), (c*cw, h_px), GRID_COL, 1)
    for r in range(1, GRID_ROWS): cv2.line(disp, (0, r*ch), (w_px, r*ch), GRID_COL, 1)
    cv2.rectangle(disp, (0, 0), (w_px-1, h_px-1), GRID_COL, 2)

    cx = w_px // 2; cy = h_px // 2
    tol_x_disp = int(CENTER_TOL_PX   * sx)
    tol_y_disp = int(CENTER_TOL_Y_PX * sy)
    cv2.rectangle(disp, (cx - tol_x_disp, 0), (cx + tol_x_disp, h_px), CENTRE_COL, 1)
    cv2.rectangle(disp, (0, cy - tol_y_disp), (w_px-1, cy + tol_y_disp), CENTRE_COL, 1)
    cv2.line(disp, (cx-20, cy), (cx+20, cy), CROSS_COL, 2)
    cv2.line(disp, (cx, cy-20), (cx, cy+20), CROSS_COL, 2)

    roi = find_banana_roi(frame, min_area=MIN_BBOX_AREA)
    if roi is not None:
        rx, ry, rw, rh = roi
        drx = int(rx*sx); dry = int(ry*sy)
        drw = int(rw*sx); drh = int(rh*sy)
        roi_cx_disp = drx + drw // 2; roi_cy_disp = dry + drh // 2
        offset_x = (rx + rw//2) - (SCAN_W//2); offset_y = (ry + rh//2) - (SCAN_H//2)
        in_zone = abs(offset_x) <= CENTER_TOL_PX and abs(offset_y) <= CENTER_TOL_Y_PX

        hsv_chk = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gm  = cv2.inRange(hsv_chk, SEG_GREEN_LOW,  SEG_GREEN_HIGH)
        ym  = cv2.inRange(hsv_chk, SEG_YELLOW_LOW, SEG_YELLOW_HIGH)
        bm  = cv2.bitwise_or(gm, ym)
        trigger_ok = int(np.sum(bm > 0)) >= MIN_TRIGGER_PX
        sharp      = _is_sharp(frame, roi)

        bbox_col = ((0, 255, 60) if (in_zone and sharp and trigger_ok)
                    else (0, 140, 255) if not trigger_ok
                    else BBOX_COL)
        cv2.rectangle(disp, (drx, dry), (drx+drw, dry+drh), bbox_col, 2)
        cv2.putText(disp, f"{rw}x{rh}px", (drx, max(dry-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_col, 1)
        cv2.circle(disp, (roi_cx_disp, roi_cy_disp), 5, bbox_col, -1)
        sx_sign = "+" if offset_x >= 0 else ""; sy_sign = "+" if offset_y >= 0 else ""
        cv2.putText(disp, f"X{sx_sign}{offset_x} Y{sy_sign}{offset_y}px",
                    (drx, dry + drh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 60) if in_zone else (180, 180, 180), 1)
        if not trigger_ok:
            cv2.putText(disp, "too small", (drx, dry+drh+26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,140,255), 1)
        elif not sharp:
            cv2.putText(disp, "blurry", (drx, dry+drh+26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,140,255), 1)

        if drw > 0 and drh > 0:
            crop_disp = disp[dry:dry+drh, drx:drx+drw]
            if crop_disp.size > 0:
                hsv_c = cv2.cvtColor(crop_disp, cv2.COLOR_BGR2HSV)
                bm_c  = _segment_banana(hsv_c); dm_c = _detect_defects(hsv_c, bm_c)
                crop_disp[dm_c > 0] = DEF_COL
                disp[dry:dry+drh, drx:drx+drw] = crop_disp
                bp = int(np.sum(bm_c > 0)); dp = int(np.sum(dm_c > 0))
                ratio = dp / bp * 100 if bp > 0 else 0.0
                cv2.putText(disp, f"HSV {ratio:.1f}%", (drx+2, dry+drh-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140,140,140), 1)
    else:
        cv2.putText(disp, "no banana", (w_px-80, h_px-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1)

    if _ml_ready:
        badge_txt = f"ML {_ml_info.get('accuracy_pct','?')}%"; badge_col = (0, 220, 100)
    else:
        badge_txt = "HSV-fallback"; badge_col = (0, 160, 255)
    cv2.putText(disp, badge_txt, (w_px-115, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_col, 1)

    if grade_label in ("GOOD", "BAD"):
        col = (0, 210, 0) if grade_label == "GOOD" else (0, 0, 220)
        cv2.putText(disp, grade_label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    return disp


def camera_loop():
    while _running:
        ret1, f1 = cam1.read()
        ret2, f2 = cam2.read()
        st = state_read()
        if ret1:
            f1  = cv2.resize(f1, (SCAN_W, SCAN_H))
            ov1 = draw_fov_overlay(f1, st["last_grade1"], st.get("vis1_offset", 0))
            state_write(frame1=f1.copy(), overlay1=ov1)
        if ret2:
            f2  = cv2.resize(f2, (SCAN_W, SCAN_H))
            ov2 = draw_fov_overlay(f2, st["last_grade2"], st.get("vis2_offset", 0))
            state_write(frame2=f2.copy(), overlay2=ov2)
        time.sleep(0.033)


# =========================================
# HARDWARE BUTTON WATCHER
# =========================================
_btn_hold_start: dict = {SHUTDOWN_PIN: None, REBOOT_PIN: None}


def _hardware_teardown():
    cleanup()
    import os as _os
    _os.kill(_os.getpid(), signal.SIGTERM)


def _do_shutdown():
    print("\n[BUTTON] SHUTDOWN triggered — cleaning up ...")
    cleanup()
    push_device_status("SHUTTING DOWN")
    subprocess.run(["sudo", "shutdown", "-h", "now"])


def _do_reboot():
    print("\n[BUTTON] REBOOT triggered — cleaning up ...")
    cleanup()
    push_device_status("REBOOTING")
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
            try: level = lgpio.gpio_read(h, pin)
            except Exception: return

            if level == 0:
                if _btn_hold_start[pin] is None:
                    _btn_hold_start[pin] = time.time()
                held = time.time() - _btn_hold_start[pin]
                pct  = min(100, int(held / BUTTON_HOLD_S * 100))
                state_write(btn_hold_pct=pct, system_action=label if pct < 100 else "")
                if held >= BUTTON_HOLD_S and not fired[pin]:
                    fired[pin] = True
                    state_write(system_action=label, btn_hold_pct=100)
                    threading.Thread(target=action_fn, daemon=False).start()
                    return
            else:
                if _btn_hold_start[pin] is not None:
                    held = time.time() - _btn_hold_start[pin]
                    print(f"[BTN] GPIO{pin} released after {held:.1f}s (no action)")
                _btn_hold_start[pin] = None
                fired[pin]           = False
                state_write(btn_hold_pct=0, system_action="")
        time.sleep(poll_s)


# =========================================
# FLASK APPLICATION
# =========================================
flask_app = Flask(__name__, template_folder="templates")


def _mjpeg_stream(overlay_key: str):
    """Generator: yields MJPEG frames from pre-computed overlay."""
    while _running:
        try:
            st    = state_read()
            frame = st.get(overlay_key)
            if frame is not None:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
        except GeneratorExit:
            break
        time.sleep(0.067)   # ~15 fps cap


@flask_app.route("/")
def index():
    return render_template(
        "livefeed.html",
        firebase_url=FIREBASE_URL,
        ml_ready=_ml_ready,
        ml_info=_ml_info,
    )


@flask_app.route("/video_feed/1")
def video_feed_1():
    return Response(
        _mjpeg_stream("overlay1"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@flask_app.route("/video_feed/2")
def video_feed_2():
    return Response(
        _mjpeg_stream("overlay2"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@flask_app.route("/events")
def events():
    """SSE stream — sends state every 150 ms."""
    def _stream():
        tick = 0
        while _running:
            st = state_read()
            results = st.get("results", [])
            payload = {
                "last_grade1":    st.get("last_grade1", ""),
                "last_grade2":    st.get("last_grade2", ""),
                "last_final":     st.get("last_final", ""),
                "last_banana_id": st.get("last_banana_id", ""),
                "pipeline_count": st.get("pipeline_count", 0),
                "cam1_busy":      st.get("cam1_busy", False),
                "cam2_busy":      st.get("cam2_busy", False),
                "ml_status1":     st.get("ml_status1", ""),
                "ml_status2":     st.get("ml_status2", ""),
                "vis1_state":     st.get("vis1_state", "ARMED"),
                "vis2_state":     st.get("vis2_state", "ARMED"),
                "vis1_offset":    st.get("vis1_offset", 0),
                "vis2_offset":    st.get("vis2_offset", 0),
                "btn_hold_pct":   st.get("btn_hold_pct", 0),
                "system_action":  st.get("system_action", ""),
                # Only send the most recent result — JS adds it to its table
                "last_result": results[0] if results else None,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            tick += 1
            if tick % 20 == 0:       # heartbeat every ~3 s
                yield ": heartbeat\n\n"
            time.sleep(0.15)
        yield 'data: {"shutdown": true}\n\n'

    return Response(
        stream_with_context(_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@flask_app.route("/capture/<int:cam_id>", methods=["POST"])
def manual_capture(cam_id: int):
    if cam_id not in (1, 2):
        return jsonify({"ok": False, "msg": "Invalid cam_id"}), 400
    st = state_read()
    busy_key  = f"cam{cam_id}_busy"
    frame_key = f"frame{cam_id}"
    if st[busy_key]:
        return jsonify({"ok": False, "msg": "Camera busy"})
    if st[frame_key] is None:
        return jsonify({"ok": False, "msg": "No frame available"})
    state_write(**{busy_key: True})
    snapshot = st[frame_key].copy()
    print(f"\n[WEB] Manual capture — Cam{cam_id}")
    _dispatch_classify(cam_id, snapshot, f"Cam{cam_id}")
    return jsonify({"ok": True})


@flask_app.route("/system/shutdown", methods=["POST"])
def web_shutdown():
    threading.Thread(target=_do_shutdown, daemon=False).start()
    return jsonify({"ok": True, "msg": "Shutting down..."})


@flask_app.route("/system/reboot", methods=["POST"])
def web_reboot():
    threading.Thread(target=_do_reboot, daemon=False).start()
    return jsonify({"ok": True, "msg": "Rebooting..."})


@flask_app.route("/health")
def health():
    return jsonify({
        "running":    _running,
        "ml_ready":   _ml_ready,
        "ml_info":    _ml_info,
        "cam1_open":  cam1.isOpened(),
        "cam2_open":  cam2.isOpened(),
        "pipeline":   len(pipeline),
        "queue_depth": len(_upload_queue),
    })


# =========================================
# STARTUP
# =========================================
def start_background_threads():
    load_local_model()
    recover_dlq_on_startup()
    threading.Thread(target=camera_loop,           daemon=True).start()
    threading.Thread(target=visual_center_watcher, daemon=True).start()
    threading.Thread(target=timeout_watcher,       daemon=True).start()
    threading.Thread(target=hardware_button_watcher, daemon=True).start()
    push_device_status("ACTIVE")
    mode = "ML (local SVM)" if _ml_ready else "HSV fallback"
    print(f"[System] Ready — classifier: {mode}")
    print(f"[System] Servo: HOME={SERVO_HOME_DEG}° PUSH={SERVO_PUSH_DEG}°  "
          f"delay={SERVO_DELAY_S}s  hold={SERVO_HOLD_S}s")
    print(f"[System] Vision trigger: tol={CENTER_TOL_PX}px  cd={VIS_COOLDOWN}s  "
          f"timeout={SCAN_TIMEOUT}s")
    print(f"[System] Flask live feed: http://0.0.0.0:5000  (local network only)")
    print(f"[System] Firebase site:   {FIREBASE_URL}")


# =========================================
# SIGNAL HANDLERS + MAIN
# =========================================
def _signal_handler(signum, frame):
    print(f"\n[System] Signal {signum} received — shutting down ...")
    cleanup()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    start_background_threads()

    print("[Flask] Starting on http://0.0.0.0:5000 ...")
    try:
        flask_app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False,
            debug=False,
        )
    finally:
        cleanup()
