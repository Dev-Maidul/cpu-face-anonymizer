"""
Internal worker for benchmark.py: runs ONE detector, in its OWN process, so
its peak RSS reflects only that detector (not whatever ran before it in a
shared process). Prints a single JSON result line to stdout.

Usage: python _benchmark_worker.py <detector_name> <repeats> <n_images> <num_threads> [video_path]
"""
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    name = sys.argv[1]
    repeats = int(sys.argv[2])
    n_images = int(sys.argv[3])
    num_threads = int(sys.argv[4])
    video_path = sys.argv[5] if len(sys.argv) > 5 else None

    cv2.setNumThreads(num_threads)  # explicit, so runs are comparable (not left to OpenCV's default)

    import gt_utils as gtu
    from detectors import haar, mediapipe_blazeface, mobilenet_ssd

    det = {"haar_cascade": haar, "mobilenet_ssd": mobilenet_ssd, "mediapipe_blazeface": mediapipe_blazeface}[name]

    frames = []
    if video_path:
        cap = cv2.VideoCapture(video_path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        source = "own_video"
    else:
        records = gtu.load_ground_truth()[:n_images]
        for rec in records:
            img = cv2.imread(gtu.image_abspath(rec))
            if img is not None:
                frames.append(img)
        source = "wider_face_subset"

    process = psutil.Process(os.getpid())

    for f in frames[:5]:  # warm-up, not timed
        det.detect_faces(f)

    repeat_ms_per_frame = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for f in frames:
            det.detect_faces(f)
        elapsed = time.perf_counter() - t0
        repeat_ms_per_frame.append(1000 * elapsed / len(frames))

    peak_rss_mb = process.memory_info().rss / (1024 * 1024)  # isolated process -> per-detector peak
    median_ms = statistics.median(repeat_ms_per_frame)
    q1, q3 = np.percentile(repeat_ms_per_frame, [25, 75])

    result = {
        "detector": name,
        "source": source,
        "n_frames": len(frames),
        "n_repeats": repeats,
        "median_latency_ms": round(median_ms, 3),
        "iqr_latency_ms": round(float(q3 - q1), 3),
        "fps_median": round(1000 / median_ms, 2) if median_ms > 0 else float("inf"),
        "peak_memory_mb": round(peak_rss_mb, 2),
        "model_size_mb": round(getattr(det, "MODEL_SIZE_MB", 0), 3),
        "num_threads": num_threads,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
