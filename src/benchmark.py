"""
Benchmark each detector's CPU-only runtime cost: FPS, per-frame latency,
CPU usage, and peak memory, over the sample image set (used as a frame
stream in place of a live video since no camera is available in this
environment). If a real video file is later placed in data/, point
VIDEO_PATH at it instead for a true video FPS benchmark.
"""
import glob
import os
import time

import cv2
import pandas as pd
import psutil

from detectors import ALL_DETECTORS

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "..", "data", "sample")
RESULTS_DIR = os.path.join(HERE, "..", "results")
VIDEO_PATH = os.path.join(HERE, "..", "data", "own_video.mp4")  # optional, see note above
WARMUP_RUNS = 5


def load_frames_from_images():
    paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))
    frames = [cv2.imread(p) for p in paths]
    return [f for f in frames if f is not None]


def load_frames_from_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def benchmark_detector(det, frames):
    process = psutil.Process(os.getpid())

    # warm-up (model init / first-call overhead shouldn't count)
    for frame in frames[:WARMUP_RUNS]:
        det.detect_faces(frame)

    process.cpu_percent(interval=None)  # reset baseline
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()
    for frame in frames:
        det.detect_faces(frame)
    elapsed = time.perf_counter() - start

    cpu_percent = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)

    n = len(frames)
    return {
        "detector": det.NAME,
        "model_size_mb": round(det.MODEL_SIZE_MB, 3),
        "num_frames": n,
        "total_time_s": round(elapsed, 3),
        "fps": round(n / elapsed, 2) if elapsed > 0 else float("inf"),
        "latency_ms_per_frame": round(1000 * elapsed / n, 2) if n else 0,
        "cpu_percent": round(cpu_percent, 1),
        "peak_memory_mb": round(max(mem_before, mem_after), 2),
    }


def main():
    if os.path.isfile(VIDEO_PATH):
        print(f"Using real video: {VIDEO_PATH}")
        frames = load_frames_from_video(VIDEO_PATH)
    else:
        print("No own_video.mp4 found in data/ -- using WIDER FACE sample images as the frame stream.")
        frames = load_frames_from_images()

    print(f"Benchmarking on {len(frames)} frames...")
    rows = []
    for det in ALL_DETECTORS:
        row = benchmark_detector(det, frames)
        print(row)
        rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
