"""
Evaluate each detector's accuracy (precision / recall / F1) against the
WIDER FACE sample ground truth, matching predicted boxes to ground-truth
boxes via IoU >= IOU_THRESHOLD (greedy one-to-one matching).
"""
import os

import cv2
import pandas as pd

from detectors import ALL_DETECTORS
from geometry import match_boxes, IOU_THRESHOLD

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "..", "data", "sample")
GT_PATH = os.path.join(SAMPLE_DIR, "ground_truth.txt")
RESULTS_DIR = os.path.join(HERE, "..", "results")


def load_ground_truth(path):
    with open(path) as f:
        lines = [l.strip() for l in f]
    gt = {}
    i = 0
    while i < len(lines):
        fname = lines[i]
        i += 1
        n = int(lines[i])
        i += 1
        boxes = [tuple(map(int, lines[i + j].split())) for j in range(n)]
        i += n
        gt[fname] = boxes
    return gt


def evaluate_detector(det, gt):
    total_tp = total_fp = total_fn = 0
    for fname, gt_boxes in gt.items():
        img = cv2.imread(os.path.join(SAMPLE_DIR, fname))
        if img is None:
            continue
        pred_boxes = det.detect_faces(img)
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "detector": det.NAME,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main():
    gt = load_ground_truth(GT_PATH)
    print(f"Evaluating on {len(gt)} images, IoU threshold = {IOU_THRESHOLD}")

    rows = []
    for det in ALL_DETECTORS:
        row = evaluate_detector(det, gt)
        print(row)
        rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "evaluation.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
