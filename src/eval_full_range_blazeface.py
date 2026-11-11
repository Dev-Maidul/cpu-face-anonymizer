"""
Task B.3: evaluate the full-range BlazeFace model on the SAME bucketed
recall/precision methodology as evaluate.py, to test whether the main
project's short-range BlazeFace's poor recall on small/crowd faces is a
short-range-specific limitation. Appends its rows to eval_by_bucket.csv
rather than re-running the original 3-detector evaluation (which was already
completed and committed before this model was downloaded).
"""
import os

import pandas as pd

import evaluate as ev
import gt_utils as gtu
from detectors import mediapipe_blazeface_full as full_range


def main():
    records = gtu.load_ground_truth()
    print(f"Evaluating {full_range.NAME} on {len(records)} images (same methodology as evaluate.py)")

    bucket_rows, raw_by_path = ev.evaluate_detector(full_range, records)
    curve_rows, rap_rows = ev.pr_curve(full_range, records, raw_by_path)

    bucket_path = os.path.join(ev.RESULTS_DIR, "eval_by_bucket.csv")
    existing = pd.read_csv(bucket_path)
    existing = existing[existing["detector"] != full_range.NAME]
    combined = pd.concat([existing, pd.DataFrame(bucket_rows)], ignore_index=True)
    combined.to_csv(bucket_path, index=False)
    print(f"Updated -> {bucket_path} (added {full_range.NAME})")

    if curve_rows:
        curve_path = os.path.join(ev.RESULTS_DIR, "pr_curves.csv")
        existing_curve = pd.read_csv(curve_path)
        existing_curve = existing_curve[existing_curve["detector"] != full_range.NAME]
        combined_curve = pd.concat([existing_curve, pd.DataFrame(curve_rows)], ignore_index=True)
        combined_curve.to_csv(curve_path, index=False)

        rap_path = os.path.join(ev.RESULTS_DIR, "recall_at_precision.csv")
        existing_rap = pd.read_csv(rap_path)
        existing_rap = existing_rap[existing_rap["detector"] != full_range.NAME]
        combined_rap = pd.concat([existing_rap, pd.DataFrame(rap_rows)], ignore_index=True)
        combined_rap.to_csv(rap_path, index=False)
        print(f"Updated -> {curve_path}, {rap_path}")

    print(f"{full_range.NAME} overall: {bucket_rows[0]}")


if __name__ == "__main__":
    main()
