"""
Task C: does blurring/pixelating a detected face actually remove the face
signal, or can a detector still find it? For a sample of images, take the
faces MobileNet-SSD finds (it has the best precision per evaluate.py, so its
boxes are the least noisy reference set to anonymize), apply several blur
kernel sizes / pixelation block counts, then re-run ALL THREE detectors on
the anonymized image and report the fraction of anonymized faces that are
STILL detected, per setting. Also reports a simple distortion measure (mean
absolute difference, 0-1 scale) between the original and anonymized face
crop, next to the re-detection rate.

This is a DETECTOR-BASED check only: it tests whether common CPU face
detectors still fire on the anonymized region, not whether a human or a
different (e.g. re-identification) model could still recognize the person.
It is not a proof of irreversibility.
"""
import os
import random

import cv2
import numpy as np
import pandas as pd

import gt_utils as gtu
from anonymize import blur_faces, pixelate_faces
from detectors import ALL_DETECTORS, mobilenet_ssd
from geometry import iou

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")

SAMPLE_SIZE = 300  # full 3226-image re-detection x 4 settings x 3 detectors is not
# needed for this secondary check; a documented random subset keeps runtime bounded.
RANDOM_SEED = 7

BLUR_KSIZES = [15, 31, 51, 75]
PIXELATE_BLOCKS = [4, 8, 12, 20]
REDETECT_IOU = 0.5
TARGET_MAX_REDETECTION = 0.02  # recommend the mildest setting under this, across all 3 detectors


def pick_sample(records):
    with_faces = [r for r in records if len(gtu.valid_boxes(r)) > 0]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(with_faces)
    return with_faces[:SAMPLE_SIZE]


def mean_abs_diff(a, b):
    """0-1 scale: 0 = identical, 1 = maximally different (every channel flipped 0<->255)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def apply_setting(frame, boxes, setting):
    kind, strength = setting
    if kind == "blur":
        return blur_faces(frame, boxes, ksize=strength)
    return pixelate_faces(frame, boxes, blocks=strength)


def evaluate_setting(records_with_boxes, setting):
    kind, strength = setting
    setting_name = f"{kind}_{strength}"

    redetected = {det.NAME: 0 for det in ALL_DETECTORS}
    total_faces = 0
    distortions = []

    for rec, img, boxes in records_with_boxes:
        anon = apply_setting(img, boxes, setting)

        for (x, y, w, h) in boxes:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            orig_crop = img[y0:y1, x0:x1]
            anon_crop = anon[y0:y1, x0:x1]
            distortions.append(mean_abs_diff(orig_crop, anon_crop))

        total_faces += len(boxes)

        for det in ALL_DETECTORS:
            found = det.detect_faces(anon)
            for (x, y, w, h) in boxes:
                if any(iou((x, y, w, h), f) >= REDETECT_IOU for f in found):
                    redetected[det.NAME] += 1

    rows = []
    mean_distortion = round(float(np.mean(distortions)), 4) if distortions else float("nan")
    for det_name, n_redetected in redetected.items():
        rows.append({
            "setting": setting_name, "kind": kind, "strength": strength,
            "detector": det_name, "n_faces": total_faces,
            "n_redetected": n_redetected,
            "redetection_rate": round(n_redetected / total_faces, 4) if total_faces else float("nan"),
            "mean_distortion": mean_distortion,
        })
    return rows


def main():
    records = gtu.load_ground_truth()
    sample = pick_sample(records)
    print(f"Privacy check sample: {len(sample)} images (seed={RANDOM_SEED}) out of {len(records)}")

    # reference faces to anonymize = MobileNet-SSD's own operating-point detections
    # (NOT ground truth -- we need the detector's view of "a detected face" to test
    # whether anonymizing it defeats detection; using GT boxes would test a
    # slightly different, less realistic question).
    records_with_boxes = []
    for rec in sample:
        img = cv2.imread(gtu.image_abspath(rec))
        if img is None:
            continue
        boxes = mobilenet_ssd.detect_faces(img)
        if boxes:
            records_with_boxes.append((rec, img, boxes))
    print(f"{len(records_with_boxes)} images had >=1 MobileNet-SSD detection to anonymize")

    settings = [("blur", k) for k in BLUR_KSIZES] + [("pixelate", b) for b in PIXELATE_BLOCKS]
    all_rows = []
    for setting in settings:
        rows = evaluate_setting(records_with_boxes, setting)
        all_rows.extend(rows)
        print(f"{setting[0]}_{setting[1]}: " + ", ".join(f"{r['detector']}={r['redetection_rate']}" for r in rows)
              + f", distortion={rows[0]['mean_distortion']}")

    df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "privacy_check.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    # recommend the mildest (least-distortion) setting where EVERY detector's
    # re-detection rate is at or below TARGET_MAX_REDETECTION
    per_setting_max = df.groupby("setting").agg(
        max_redetection=("redetection_rate", "max"),
        mean_distortion=("mean_distortion", "first"),
        kind=("kind", "first"), strength=("strength", "first"),
    ).reset_index()
    passing = per_setting_max[per_setting_max["max_redetection"] <= TARGET_MAX_REDETECTION]

    rec_path = os.path.join(RESULTS_DIR, "privacy_recommendation.txt")
    with open(rec_path, "w") as f:
        if passing.empty:
            msg = (f"No tested setting got every detector's re-detection rate <= {TARGET_MAX_REDETECTION}. "
                   f"Strongest tested setting was blur_{max(BLUR_KSIZES)} / pixelate_{max(PIXELATE_BLOCKS)}; "
                   f"see privacy_check.csv for the full table.")
        else:
            best = passing.sort_values("mean_distortion").iloc[0]
            msg = (f"Recommended default: {best['setting']} "
                   f"(max re-detection across detectors = {best['max_redetection']}, "
                   f"mean distortion = {best['mean_distortion']}). "
                   f"This is a detector-based check only -- not proof of irreversibility, "
                   f"and not a re-identification attack test.")
        print(msg)
        f.write(msg + "\n")
    print(f"Saved -> {rec_path}")


if __name__ == "__main__":
    main()
