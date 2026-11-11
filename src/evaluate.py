"""
Rigorous evaluation of each CPU-only detector against the FULL WIDER FACE
validation split (3226 images), matching WIDER FACE's own evaluation
philosophy: for a face-anonymization tool, a missed face stays visible, so
RECALL -- not F1 -- is the privacy-critical number. This script reports:

  1. Recall and precision broken down by ground-truth face-height bucket and
     by WIDER FACE's own blur attribute, plus overall, with 95% bootstrap
     confidence intervals (resampling images, 1000x).
     - Recall is bucketed by the GROUND-TRUTH box's own height -- "of all
       faces this small, how many did we find".
     - Precision is bucketed by the PREDICTED box's own height -- a false
       positive has no ground-truth box to bucket it by, so this is the only
       well-defined way to report precision per size bucket. This choice is
       stated explicitly here and in the README; it is not interchangeable
       with GT-bucketed recall.
  2. Full precision-recall curves (confidence threshold sweep) for the two
     detectors that expose a continuous confidence score (MobileNet-SSD,
     MediaPipe BlazeFace), plus a recall-at-fixed-precision table.
     Haar Cascade has NO continuous confidence score (minNeighbors is a
     discrete knob) -- this is reported as an explicit limitation, not faked.

Every number in results/*.csv comes from running the detectors; nothing here
is hand-typed into the README/report.
"""
import json
import os
import warnings

import cv2
import numpy as np
import pandas as pd

# Expected/benign: a bootstrap resample can land on a bucket with zero
# tp+fp (or tp+fn) by chance (e.g. a rare blur level) -- nanpercentile
# correctly returns NaN for that resample rather than fabricating a value.
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

import gt_utils as gtu
from detectors import ALL_DETECTORS
from geometry import IOU_THRESHOLD, match_per_prediction

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw_detections")

N_BOOTSTRAP = 1000
BOOT_SEED = 42
PR_THRESHOLDS = np.round(np.arange(0.0, 1.0, 0.04), 2)
TARGET_PRECISIONS = [0.9, 0.8, 0.7]

SIZE_LABELS = gtu.SIZE_BUCKET_LABELS
BLUR_LABELS = ["clear", "normal", "heavy"]


# --------------------------------------------------------------------------
# Step 1: run each detector once per image, cache raw (boxes [+ score]).
# Re-running this script re-uses the cache instead of re-running inference.
# --------------------------------------------------------------------------
def run_inference(det, records):
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"{det.NAME}.jsonl")
    if os.path.isfile(out_path):
        print(f"[cache] {det.NAME}: reusing {out_path}")
        return load_raw(out_path)

    print(f"[infer] {det.NAME}: running on {len(records)} images ...")
    has_score = getattr(det, "HAS_CONFIDENCE_SCORE", False)
    rows = []
    for n, rec in enumerate(records):
        img = cv2.imread(gtu.image_abspath(rec))
        if img is None:
            rows.append({"path": rec["path"], "preds": []})
            continue
        if has_score:
            preds = det.detect_faces_with_scores(img, conf_threshold=0.0)
            preds = [[x, y, w, h, round(float(s), 4)] for (x, y, w, h, s) in preds]
        else:
            preds = [[x, y, w, h, None] for (x, y, w, h) in det.detect_faces(img)]
        rows.append({"path": rec["path"], "preds": preds})
        if (n + 1) % 500 == 0:
            print(f"  {det.NAME}: {n + 1}/{len(records)}")

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[infer] {det.NAME}: done -> {out_path}")
    return rows


def load_raw(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def operating_boxes(det, preds):
    """preds: list of [x,y,w,h,score_or_None]. Returns boxes at the detector's
    fixed operating threshold (the same one detect_faces()/benchmark.py use)."""
    if getattr(det, "HAS_CONFIDENCE_SCORE", False):
        thr = det.OPERATING_THRESHOLD
        return [(p[0], p[1], p[2], p[3]) for p in preds if p[4] is not None and p[4] >= thr]
    return [(p[0], p[1], p[2], p[3]) for p in preds]


# --------------------------------------------------------------------------
# Step 2: per-image bucketed stats at the operating threshold.
# --------------------------------------------------------------------------
def per_image_stats(det, record, preds):
    gt_box_records = gtu.valid_box_records(record)  # invalid=1 boxes excluded
    gt_boxes = [(b["x"], b["y"], b["w"], b["h"]) for b in gt_box_records]
    pred_boxes = operating_boxes(det, preds)

    pred_matches, matched_gt = match_per_prediction(pred_boxes, gt_boxes, IOU_THRESHOLD)

    tp = sum(1 for m in pred_matches if m is not None)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    recall_size = {label: {"tp": 0, "fn": 0} for label in SIZE_LABELS}
    recall_blur = {label: {"tp": 0, "fn": 0} for label in BLUR_LABELS}
    precision_size = {label: {"tp": 0, "fp": 0} for label in SIZE_LABELS}

    for idx, b in enumerate(gt_box_records):
        sbucket = gtu.size_bucket(b["h"])
        blabel = gtu.blur_label(b["blur"])
        is_tp = idx in matched_gt
        recall_size[sbucket]["tp" if is_tp else "fn"] += 1
        recall_blur[blabel]["tp" if is_tp else "fn"] += 1

    for i, pb in enumerate(pred_boxes):
        sbucket = gtu.size_bucket(pb[3])  # predicted box's own height
        is_tp = pred_matches[i] is not None
        precision_size[sbucket]["tp" if is_tp else "fp"] += 1

    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn},
        "recall_size": recall_size,
        "recall_blur": recall_blur,
        "precision_size": precision_size,
    }


def stats_to_arrays(all_stats):
    """Stack per-image stat dicts into numpy arrays for fast bootstrap resampling."""
    n = len(all_stats)
    overall = np.array([[s["overall"]["tp"], s["overall"]["fp"], s["overall"]["fn"]] for s in all_stats])

    rs_tp = np.array([[s["recall_size"][lbl]["tp"] for lbl in SIZE_LABELS] for s in all_stats])
    rs_fn = np.array([[s["recall_size"][lbl]["fn"] for lbl in SIZE_LABELS] for s in all_stats])

    rb_tp = np.array([[s["recall_blur"][lbl]["tp"] for lbl in BLUR_LABELS] for s in all_stats])
    rb_fn = np.array([[s["recall_blur"][lbl]["fn"] for lbl in BLUR_LABELS] for s in all_stats])

    ps_tp = np.array([[s["precision_size"][lbl]["tp"] for lbl in SIZE_LABELS] for s in all_stats])
    ps_fp = np.array([[s["precision_size"][lbl]["fp"] for lbl in SIZE_LABELS] for s in all_stats])

    assert overall.shape[0] == n
    return {
        "overall": overall, "rs_tp": rs_tp, "rs_fn": rs_fn,
        "rb_tp": rb_tp, "rb_fn": rb_fn, "ps_tp": ps_tp, "ps_fp": ps_fp,
    }


def safe_div(a, b):
    return a / b if b > 0 else float("nan")


def bootstrap_ci(tp_col, fn_or_fp_col, kind, n_images, rng, n_boot=N_BOOTSTRAP):
    """kind: 'recall' (fn_or_fp_col is FN) or 'precision' (fn_or_fp_col is FP).
    Resamples IMAGES with replacement; returns (lo, hi) 95% percentile CI."""
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_images, n_images)
        tp = tp_col[idx].sum()
        other = fn_or_fp_col[idx].sum()
        denom = tp + other
        vals[b] = tp / denom if denom > 0 else np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def evaluate_detector(det, records):
    raw = run_inference(det, records)
    raw_by_path = {r["path"]: r["preds"] for r in raw}

    all_stats = []
    for rec in records:
        preds = raw_by_path.get(rec["path"], [])
        all_stats.append(per_image_stats(det, rec, preds))

    arr = stats_to_arrays(all_stats)
    n_images = len(records)
    rng = np.random.default_rng(BOOT_SEED)

    rows = []

    # overall
    tp, fp, fn = arr["overall"].sum(axis=0)
    precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision + recall > 0 else float("nan")
    p_lo, p_hi = bootstrap_ci(arr["overall"][:, 0], arr["overall"][:, 1], "precision", n_images, rng)
    r_lo, r_hi = bootstrap_ci(arr["overall"][:, 0], arr["overall"][:, 2], "recall", n_images, rng)
    rows.append({
        "detector": det.NAME, "metric_type": "overall", "bucket": "overall",
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "gt_count": int(tp + fn),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "precision_ci_lo": p_lo, "precision_ci_hi": p_hi,
        "recall_ci_lo": r_lo, "recall_ci_hi": r_hi,
    })

    # recall by size bucket
    for i, lbl in enumerate(SIZE_LABELS):
        tp_, fn_ = arr["rs_tp"][:, i].sum(), arr["rs_fn"][:, i].sum()
        recall = safe_div(tp_, tp_ + fn_)
        r_lo, r_hi = bootstrap_ci(arr["rs_tp"][:, i], arr["rs_fn"][:, i], "recall", n_images, rng)
        rows.append({
            "detector": det.NAME, "metric_type": "recall_by_size", "bucket": lbl,
            "tp": int(tp_), "fp": float("nan"), "fn": int(fn_), "gt_count": int(tp_ + fn_),
            "precision": float("nan"), "recall": round(recall, 4), "f1": float("nan"),
            "precision_ci_lo": float("nan"), "precision_ci_hi": float("nan"),
            "recall_ci_lo": r_lo, "recall_ci_hi": r_hi,
        })

    # recall by blur level
    for i, lbl in enumerate(BLUR_LABELS):
        tp_, fn_ = arr["rb_tp"][:, i].sum(), arr["rb_fn"][:, i].sum()
        recall = safe_div(tp_, tp_ + fn_)
        r_lo, r_hi = bootstrap_ci(arr["rb_tp"][:, i], arr["rb_fn"][:, i], "recall", n_images, rng)
        rows.append({
            "detector": det.NAME, "metric_type": "recall_by_blur", "bucket": lbl,
            "tp": int(tp_), "fp": float("nan"), "fn": int(fn_), "gt_count": int(tp_ + fn_),
            "precision": float("nan"), "recall": round(recall, 4), "f1": float("nan"),
            "precision_ci_lo": float("nan"), "precision_ci_hi": float("nan"),
            "recall_ci_lo": r_lo, "recall_ci_hi": r_hi,
        })

    # precision by (predicted-box) size bucket
    for i, lbl in enumerate(SIZE_LABELS):
        tp_, fp_ = arr["ps_tp"][:, i].sum(), arr["ps_fp"][:, i].sum()
        precision = safe_div(tp_, tp_ + fp_)
        p_lo, p_hi = bootstrap_ci(arr["ps_tp"][:, i], arr["ps_fp"][:, i], "precision", n_images, rng)
        rows.append({
            "detector": det.NAME, "metric_type": "precision_by_size", "bucket": lbl,
            "tp": int(tp_), "fp": int(fp_), "fn": float("nan"), "gt_count": float("nan"),
            "precision": round(precision, 4), "recall": float("nan"), "f1": float("nan"),
            "precision_ci_lo": p_lo, "precision_ci_hi": p_hi,
            "recall_ci_lo": float("nan"), "recall_ci_hi": float("nan"),
        })

    return rows, raw_by_path


# --------------------------------------------------------------------------
# Step 3: PR curve + recall-at-fixed-precision, for scored detectors only.
# --------------------------------------------------------------------------
def pr_curve(det, records, raw_by_path):
    if not getattr(det, "HAS_CONFIDENCE_SCORE", False):
        return [], []

    gt_per_image = [[(b["x"], b["y"], b["w"], b["h"]) for b in gtu.valid_box_records(rec)] for rec in records]
    preds_per_image = [raw_by_path.get(rec["path"], []) for rec in records]

    curve_rows = []
    for thr in PR_THRESHOLDS:
        tp_sum = fp_sum = fn_sum = 0
        for gt_boxes, preds in zip(gt_per_image, preds_per_image):
            boxes = [(p[0], p[1], p[2], p[3]) for p in preds if p[4] is not None and p[4] >= thr]
            pred_matches, matched_gt = match_per_prediction(boxes, gt_boxes, IOU_THRESHOLD)
            tp = sum(1 for m in pred_matches if m is not None)
            tp_sum += tp
            fp_sum += len(boxes) - tp
            fn_sum += len(gt_boxes) - len(matched_gt)
        precision = safe_div(tp_sum, tp_sum + fp_sum)
        recall = safe_div(tp_sum, tp_sum + fn_sum)
        curve_rows.append({
            "detector": det.NAME, "threshold": float(thr),
            "tp": tp_sum, "fp": fp_sum, "fn": fn_sum,
            "precision": round(precision, 4) if precision == precision else precision,
            "recall": round(recall, 4) if recall == recall else recall,
        })

    # recall at fixed precision: among thresholds achieving >= target precision,
    # pick the one with the highest recall (lowest threshold that clears the bar).
    rap_rows = []
    for target in TARGET_PRECISIONS:
        candidates = [r for r in curve_rows if r["precision"] == r["precision"] and r["precision"] >= target]
        if not candidates:
            rap_rows.append({
                "detector": det.NAME, "target_precision": target,
                "achieved_precision": float("nan"), "recall": float("nan"),
                "threshold": float("nan"), "note": "not achievable at any sweeped threshold",
            })
            continue
        best = max(candidates, key=lambda r: r["recall"])
        rap_rows.append({
            "detector": det.NAME, "target_precision": target,
            "achieved_precision": best["precision"], "recall": best["recall"],
            "threshold": best["threshold"], "note": "",
        })

    return curve_rows, rap_rows


def main():
    records = gtu.load_ground_truth()
    print(f"Loaded ground truth for {len(records)} images (full WIDER FACE validation split)")
    print(f"IoU threshold = {IOU_THRESHOLD}, bootstrap resamples = {N_BOOTSTRAP}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_bucket_rows = []
    all_curve_rows = []
    all_rap_rows = []

    for det in ALL_DETECTORS:
        bucket_rows, raw_by_path = evaluate_detector(det, records)
        all_bucket_rows.extend(bucket_rows)
        print(f"{det.NAME}: overall recall={bucket_rows[0]['recall']}, precision={bucket_rows[0]['precision']}")

        curve_rows, rap_rows = pr_curve(det, records, raw_by_path)
        all_curve_rows.extend(curve_rows)
        all_rap_rows.extend(rap_rows)
        if not getattr(det, "HAS_CONFIDENCE_SCORE", False):
            print(f"{det.NAME}: no continuous confidence score -- PR curve skipped (see README limitation)")

    pd.DataFrame(all_bucket_rows).to_csv(os.path.join(RESULTS_DIR, "eval_by_bucket.csv"), index=False)
    print(f"Saved -> {RESULTS_DIR}/eval_by_bucket.csv")

    if all_curve_rows:
        pd.DataFrame(all_curve_rows).to_csv(os.path.join(RESULTS_DIR, "pr_curves.csv"), index=False)
        print(f"Saved -> {RESULTS_DIR}/pr_curves.csv")
    if all_rap_rows:
        pd.DataFrame(all_rap_rows).to_csv(os.path.join(RESULTS_DIR, "recall_at_precision.csv"), index=False)
        print(f"Saved -> {RESULTS_DIR}/recall_at_precision.csv")


if __name__ == "__main__":
    main()
