"""
Task B: try to cheaply improve recall on CPU via (1) multi-scale inference
and (2) detector union, measured on the SAME full WIDER FACE validation set
and IoU 0.5 matching as evaluate.py. Reports recall by size bucket and the
extra latency each configuration costs -- including if a configuration does
NOT help. Tiled-crop inference (mentioned as optional in the brief) was not
implemented; see README limitations.

CPU-only throughout. Detector union reuses evaluate.py's cached single-scale
operating-point predictions (results/raw_detections/*.jsonl) -- no need to
re-run those detectors. Multi-scale requires fresh inference at each extra
scale, since evaluate.py never ran detectors on upscaled images.
"""
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

import gt_utils as gtu
from detectors import ALL_DETECTORS
from geometry import IOU_THRESHOLD, match_per_prediction, nms

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw_detections")  # shared with evaluate.py
MULTISCALE_RAW_DIR = os.path.join(RESULTS_DIR, "raw_detections_multiscale")

SCALES = [1.0, 1.5, 2.0]
NMS_IOU = 0.4
SIZE_LABELS = gtu.SIZE_BUCKET_LABELS


def detect_scaled(det, img, scale):
    """Run one detector at one scale; return [(x,y,w,h,score)] mapped back to
    original-image coordinates. score is the detector's own confidence, or
    None mapped to 1.0 for detectors with no confidence score (Haar) -- NMS
    ranking among same-confidence boxes then falls back to detection order,
    an acknowledged limitation for combining a non-scored detector."""
    h, w = img.shape[:2]
    resized = img if scale == 1.0 else cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))

    if getattr(det, "HAS_CONFIDENCE_SCORE", False):
        raw = det.detect_faces_with_scores(resized, conf_threshold=0.0)
        thr = det.OPERATING_THRESHOLD
        raw = [(x, y, bw, bh, s) for (x, y, bw, bh, s) in raw if s >= thr]
    else:
        raw = [(x, y, bw, bh, 1.0) for (x, y, bw, bh) in det.detect_faces(resized)]

    return [(x / scale, y / scale, bw / scale, bh / scale, s) for (x, y, bw, bh, s) in raw]


def run_multiscale_inference(det, records):
    os.makedirs(MULTISCALE_RAW_DIR, exist_ok=True)
    out_path = os.path.join(MULTISCALE_RAW_DIR, f"{det.NAME}.jsonl")
    if os.path.isfile(out_path):
        print(f"[cache] {det.NAME} multi-scale: reusing {out_path}")
        rows = []
        with open(out_path) as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    print(f"[infer] {det.NAME}: multi-scale inference ({SCALES}) on {len(records)} images ...")
    rows = []
    for n, rec in enumerate(records):
        img = cv2.imread(gtu.image_abspath(rec))
        if img is None:
            rows.append({"path": rec["path"], "per_scale": {}, "ms_per_scale": {}})
            continue
        per_scale = {}
        ms_per_scale = {}
        for scale in SCALES:
            t0 = time.perf_counter()
            boxes = detect_scaled(det, img, scale)
            ms_per_scale[str(scale)] = round((time.perf_counter() - t0) * 1000, 3)
            per_scale[str(scale)] = [[round(v, 2) for v in b[:4]] + [b[4]] for b in boxes]
        rows.append({"path": rec["path"], "per_scale": per_scale, "ms_per_scale": ms_per_scale})
        if (n + 1) % 500 == 0:
            print(f"  {det.NAME}: {n + 1}/{len(records)}")

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[infer] {det.NAME}: multi-scale done -> {out_path}")
    return rows


def recall_by_bucket(gt_box_records, matched_gt):
    counts = {lbl: {"tp": 0, "fn": 0} for lbl in SIZE_LABELS}
    for idx, b in enumerate(gt_box_records):
        lbl = gtu.size_bucket(b["h"])
        counts[lbl]["tp" if idx in matched_gt else "fn"] += 1
    return counts


def merge_counts(acc, new):
    for lbl in SIZE_LABELS:
        acc[lbl]["tp"] += new[lbl]["tp"]
        acc[lbl]["fn"] += new[lbl]["fn"]


def safe_div(a, b):
    return a / b if b > 0 else float("nan")


# --------------------------------------------------------------------------
# Experiment 1: multi-scale (single-scale subset == the detector's baseline)
# --------------------------------------------------------------------------
def eval_multiscale(det, records):
    ms_rows = run_multiscale_inference(det, records)
    ms_by_path = {r["path"]: r for r in ms_rows}

    configs = {
        "scale_1.0_only": [1.0],
        "scale_1.0_1.5": [1.0, 1.5],
        "scale_1.0_1.5_2.0": [1.0, 1.5, 2.0],
    }

    out_rows = []
    for config_name, scales_used in configs.items():
        total_ms = 0.0
        overall_tp = overall_fp = overall_fn = 0
        size_counts = {lbl: {"tp": 0, "fn": 0} for lbl in SIZE_LABELS}

        for rec in records:
            entry = ms_by_path.get(rec["path"])
            if entry is None or not entry["per_scale"]:
                continue
            boxes, scores = [], []
            for s in scales_used:
                key = str(s)
                for (x, y, bw, bh, sc) in entry["per_scale"].get(key, []):
                    boxes.append((x, y, bw, bh))
                    scores.append(sc)
                total_ms += entry["ms_per_scale"].get(key, 0.0)

            keep_idx = nms(boxes, scores, iou_threshold=NMS_IOU)
            final_boxes = [boxes[i] for i in keep_idx]

            gt_box_records = gtu.valid_box_records(rec)
            gt_boxes = [(b["x"], b["y"], b["w"], b["h"]) for b in gt_box_records]
            pred_matches, matched_gt = match_per_prediction(final_boxes, gt_boxes, IOU_THRESHOLD)

            tp = sum(1 for m in pred_matches if m is not None)
            overall_tp += tp
            overall_fp += len(final_boxes) - tp
            overall_fn += len(gt_boxes) - len(matched_gt)
            merge_counts(size_counts, recall_by_bucket(gt_box_records, matched_gt))

        n_images = len(records)
        row = {
            "detector": det.NAME, "config": config_name,
            "recall_overall": round(safe_div(overall_tp, overall_tp + overall_fn), 4),
            "precision_overall": round(safe_div(overall_tp, overall_tp + overall_fp), 4),
            "ms_per_image": round(total_ms / n_images, 3),
        }
        for lbl in SIZE_LABELS:
            c = size_counts[lbl]
            row[f"recall_{lbl}"] = round(safe_div(c["tp"], c["tp"] + c["fn"]), 4)
            row[f"gt_count_{lbl}"] = c["tp"] + c["fn"]
        out_rows.append(row)

    return out_rows


# --------------------------------------------------------------------------
# Experiment 2: detector union (reuses evaluate.py's single-scale cache)
# --------------------------------------------------------------------------
def load_single_scale_cache(det_name):
    path = os.path.join(RAW_DIR, f"{det_name}.jsonl")
    if not os.path.isfile(path):
        return None
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["path"]] = r["preds"]
    return rows


def eval_union(records):
    caches = {}
    thresholds = {}
    for det in ALL_DETECTORS:
        cache = load_single_scale_cache(det.NAME)
        if cache is None:
            print(f"[skip] {det.NAME}: no evaluate.py cache found, run evaluate.py first")
            continue
        caches[det.NAME] = cache
        thresholds[det.NAME] = getattr(det, "OPERATING_THRESHOLD", None)

    if len(caches) < 2:
        print("[skip] union experiment needs at least 2 cached detectors")
        return []

    from itertools import combinations
    out_rows = []
    det_names = list(caches.keys())

    for r in range(2, len(det_names) + 1):
        for combo in combinations(det_names, r):
            overall_tp = overall_fp = overall_fn = 0
            size_counts = {lbl: {"tp": 0, "fn": 0} for lbl in SIZE_LABELS}

            for rec in records:
                boxes, scores = [], []
                for name in combo:
                    preds = caches[name].get(rec["path"], [])
                    thr = thresholds[name]
                    for p in preds:
                        x, y, bw, bh, s = p
                        # evaluate.py's cache stores ALL raw candidates (score floor 0.0)
                        # for scored detectors -- apply that detector's own operating
                        # threshold here; Haar's cache is already its final output.
                        if thr is None or (s is not None and s >= thr):
                            boxes.append((x, y, bw, bh))
                            scores.append(s if s is not None else 1.0)

                keep_idx = nms(boxes, scores, iou_threshold=NMS_IOU)
                final_boxes = [boxes[i] for i in keep_idx]

                gt_box_records = gtu.valid_box_records(rec)
                gt_boxes = [(b["x"], b["y"], b["w"], b["h"]) for b in gt_box_records]
                pred_matches, matched_gt = match_per_prediction(final_boxes, gt_boxes, IOU_THRESHOLD)

                tp = sum(1 for m in pred_matches if m is not None)
                overall_tp += tp
                overall_fp += len(final_boxes) - tp
                overall_fn += len(gt_boxes) - len(matched_gt)
                merge_counts(size_counts, recall_by_bucket(gt_box_records, matched_gt))

            row = {
                "config": " + ".join(combo),
                "n_detectors": len(combo),
                "recall_overall": round(safe_div(overall_tp, overall_tp + overall_fn), 4),
                "precision_overall": round(safe_div(overall_tp, overall_tp + overall_fp), 4),
            }
            for lbl in SIZE_LABELS:
                c = size_counts[lbl]
                row[f"recall_{lbl}"] = round(safe_div(c["tp"], c["tp"] + c["fn"]), 4)
            out_rows.append(row)

    return out_rows


def main():
    records = gtu.load_ground_truth()
    print(f"Loaded {len(records)} images for Task B (recall improvement) experiments")

    multiscale_rows = []
    for det in ALL_DETECTORS:
        multiscale_rows.extend(eval_multiscale(det, records))
    df_ms = pd.DataFrame(multiscale_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_ms.to_csv(os.path.join(RESULTS_DIR, "improve_multiscale.csv"), index=False)
    print(f"Saved -> {RESULTS_DIR}/improve_multiscale.csv")
    print(df_ms.to_string(index=False))

    union_rows = eval_union(records)
    if union_rows:
        df_union = pd.DataFrame(union_rows)
        df_union.to_csv(os.path.join(RESULTS_DIR, "improve_union.csv"), index=False)
        print(f"Saved -> {RESULTS_DIR}/improve_union.csv")
        print(df_union.to_string(index=False))


if __name__ == "__main__":
    main()
