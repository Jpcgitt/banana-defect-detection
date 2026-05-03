"""
train_model.py — Banana Sorting SVM Trainer
============================================
Dataset layout expected
-----------------------
  banana_sorting/backend/
      dataset/
          good/   ← 60 GOOD banana images
          bad/    ← 64 BAD  banana images  (dark spots, cuts)
      train_model.py          ← this file
      banana_model.pkl        ← generated here
      label_encoder.pkl       ← generated here
      model_info.json         ← generated here

Run
---
  cd /banana_sorting/backend
  python3 train_model.py

Output
------
  banana_model.pkl    — sklearn Pipeline (StandardScaler + SVM)
  label_encoder.pkl   — LabelEncoder  ("bad" / "good" ↔ 0 / 1)
  model_info.json     — accuracy, sample counts, trained timestamp

Features extracted per image (must stay in sync with main system)
-----------------------------------------------------------------
  HOG descriptor  — captures shape/texture (banana curves, cut edges, spots)
  HSV histogram   — captures colour distribution (browning, dark patches)

Both are concatenated into a single feature vector fed to the SVM.
The SVM uses an RBF kernel with probability=True so predict_proba()
returns a confidence score the main system uses as "defect_pct".

Why SVM for 60+64 images?
--------------------------
  Deep learning needs thousands of images.  SVM + hand-crafted features
  (HOG + colour histogram) is the classic approach for small datasets and
  runs in milliseconds on RPi5 — no GPU needed, fully offline.

Tips for improving accuracy
---------------------------
  • Collect more photos — aim for 150+ per class
  • Vary lighting, angles, and stages of ripeness
  • Re-run this script any time you add photos
  • Check the confusion matrix printed at the end — if BAD recall is low,
    lower DEFECT_RATIO_MAX in the main system as a complementary guard
"""

import cv2
import numpy as np
import pickle
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

# =========================================
# PATHS
# =========================================
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
GOOD_DIR    = DATASET_DIR / "good"
BAD_DIR     = DATASET_DIR / "bad"

MODEL_OUT   = BASE_DIR / "banana_model.pkl"
ENCODER_OUT = BASE_DIR / "label_encoder.pkl"
INFO_OUT    = BASE_DIR / "model_info.json"

# =========================================
# FEATURE EXTRACTION SETTINGS
# Must stay identical to the main sorting system.
# =========================================
IMG_SIZE   = 64          # resize every image to 64×64 before feature extraction
HOG_CELL   = (8, 8)
HOG_BLOCK  = (16, 16)
HOG_STRIDE = (8, 8)
HOG_BINS   = 9
HIST_BINS  = 32          # bins per HSV channel histogram

_HOG = cv2.HOGDescriptor(
    (IMG_SIZE, IMG_SIZE),
    HOG_BLOCK, HOG_STRIDE, HOG_CELL, HOG_BINS,
)


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a 1-D float32 feature vector combining:
      - HOG descriptor  (shape / texture — catches cuts, spot edges)
      - Normalised HSV histograms  (colour — catches browning, dark spots)
    """
    img  = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG — shape and texture
    hog_vec = _HOG.compute(gray).flatten()

    # HSV colour histograms (normalised so brightness doesn't dominate)
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [HIST_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [HIST_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [HIST_BINS], [0, 256]).flatten()
    colour_vec = np.concatenate([h_hist, s_hist, v_hist])
    total = colour_vec.sum()
    if total > 0:
        colour_vec /= total   # normalise to sum=1

    return np.concatenate([hog_vec, colour_vec]).astype(np.float32)


# =========================================
# AUGMENTATION
# -----------------------------------------
# With only ~60 images per class, augmentation
# gives the SVM more variety to learn from.
# Each image produces 6 variants:
#   original, flip-H, flip-V, flip-both,
#   +brightness, -brightness
# This 6× expansion → ~360 GOOD + ~384 BAD
# which is enough for a robust SVM.
# =========================================
def augment(img: np.ndarray) -> list:
    variants = [img]

    # Horizontal and vertical flips
    variants.append(cv2.flip(img, 1))
    variants.append(cv2.flip(img, 0))
    variants.append(cv2.flip(img, -1))

    # Slight brightness shift  (+40 and -40 in V channel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    for delta in (+40, -40):
        shifted = hsv.copy()
        shifted[:, :, 2] = np.clip(shifted[:, :, 2] + delta, 0, 255)
        variants.append(cv2.cvtColor(shifted.astype(np.uint8),
                                      cv2.COLOR_HSV2BGR))
    return variants


# =========================================
# LOAD DATASET
# =========================================
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class(folder: Path, label: str, augment_data: bool = True):
    X, y = [], []
    files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED]

    if not files:
        print(f"  [WARN] No images found in {folder}")
        return X, y

    for fpath in files:
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  [SKIP] Could not read {fpath.name}")
            continue

        imgs = augment(img) if augment_data else [img]
        for variant in imgs:
            feat = extract_features(variant)
            X.append(feat)
            y.append(label)

    print(f"  [{label.upper():4s}]  {len(files)} images  "
          f"→  {len(X)} samples after augmentation")
    return X, y


# =========================================
# MAIN TRAINING ROUTINE
# =========================================
def train():
    print("\n" + "=" * 60)
    print("  BANANA SORTING  —  SVM TRAINER")
    print("=" * 60)

    # ── Verify dataset folders exist ──────────────────────────────
    for folder, name in ((GOOD_DIR, "good"), (BAD_DIR, "bad")):
        if not folder.exists():
            print(f"\n[ERROR] Dataset folder not found: {folder}")
            print(f"        Create it and add your {name} banana photos.")
            sys.exit(1)

    print(f"\n[Dataset] {DATASET_DIR}")

    # ── Load + augment ────────────────────────────────────────────
    print("\n[Loading images]")
    Xg, yg = load_class(GOOD_DIR, "good", augment_data=True)
    Xb, yb = load_class(BAD_DIR,  "bad",  augment_data=True)

    if not Xg or not Xb:
        print("\n[ERROR] Need at least 1 image in each class.")
        sys.exit(1)

    X = np.array(Xg + Xb, dtype=np.float32)
    y = np.array(yg + yb)

    good_raw = len([f for f in GOOD_DIR.iterdir()
                    if f.suffix.lower() in SUPPORTED])
    bad_raw  = len([f for f in BAD_DIR.iterdir()
                    if f.suffix.lower() in SUPPORTED])

    print(f"\n  Raw images : {good_raw} GOOD  +  {bad_raw} BAD"
          f"  =  {good_raw + bad_raw} total")
    print(f"  After aug  : {len(Xg)} GOOD  +  {len(Xb)} BAD"
          f"  =  {len(X)} total")
    print(f"  Feature dim: {X.shape[1]}")

    # ── Label encoding ────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\n[LabelEncoder] classes = {list(le.classes_)}")

    # ── Build SVM pipeline ────────────────────────────────────────
    #  StandardScaler normalises HOG + histogram magnitudes so the
    #  SVM treats both feature types equally.
    #  RBF kernel handles non-linear banana defect boundaries.
    #  C=10 and gamma='scale' work well for small-medium datasets.
    #  probability=True enables predict_proba() for confidence scores.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, class_weight="balanced",
                       random_state=42)),
    ])

    # ── Cross-validation (5-fold stratified) ─────────────────────
    #  Gives an honest accuracy estimate without a separate test set
    #  (important when dataset is small).
    print("\n[Cross-validation]  5-fold stratified ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(pipeline, X, y_enc, cv=cv)

    cv_acc = accuracy_score(y_enc, y_pred_cv)
    print(f"\n  CV Accuracy : {cv_acc * 100:.1f}%")

    # Per-class report
    print("\n  Classification report (cross-validated):")
    report = classification_report(
        y_enc, y_pred_cv,
        target_names=le.classes_,
        digits=3)
    for line in report.splitlines():
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(y_enc, y_pred_cv)
    print("\n  Confusion matrix:")
    print(f"    Labels : {list(le.classes_)}")
    for i, row in enumerate(cm):
        print(f"    {le.classes_[i]:4s}   {row}")

    # Warn if BAD recall is dangerously low
    bad_idx   = list(le.classes_).index("bad")
    bad_recall = cm[bad_idx, bad_idx] / cm[bad_idx].sum()
    if bad_recall < 0.80:
        print(f"\n  [WARN] BAD recall = {bad_recall:.1%}  (<80%).")
        print("         Consider adding more BAD photos or lowering")
        print("         DEFECT_RATIO_MAX in the main system as a backup.")
    else:
        print(f"\n  [OK]  BAD recall = {bad_recall:.1%}")

    # ── Final fit on ALL data ─────────────────────────────────────
    print("\n[Training] Fitting final model on full dataset ...")
    pipeline.fit(X, y_enc)
    print("  Done.")

    # ── Save artefacts ────────────────────────────────────────────
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[Saved] {MODEL_OUT.name}")

    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)
    print(f"[Saved] {ENCODER_OUT.name}")

    info = {
        "trained_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "good_images":  good_raw,
        "bad_images":   bad_raw,
        "good_samples": len(Xg),
        "bad_samples":  len(Xb),
        "total_samples": len(X),
        "feature_dim":  int(X.shape[1]),
        "accuracy_pct": round(cv_acc * 100, 1),
        "bad_recall_pct": round(bad_recall * 100, 1),
        "model":        "SVM RBF  C=10  gamma=scale  balanced",
        "augmentation": "flip-H flip-V flip-HV bright+40 bright-40",
        "img_size":     IMG_SIZE,
        "hog_bins":     HOG_BINS,
        "hist_bins":    HIST_BINS,
    }
    with open(INFO_OUT, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[Saved] {INFO_OUT.name}")

    print("\n" + "=" * 60)
    print(f"  MODEL READY  —  {cv_acc * 100:.1f}% CV accuracy")
    print(f"  Start the main system — it will auto-load the model.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()