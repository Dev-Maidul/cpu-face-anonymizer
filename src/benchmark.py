"""
Task D: benchmark fix. Each detector is launched in its OWN subprocess via
_benchmark_worker.py, so peak RSS is per-detector rather than cumulative
across one shared process (Phase 1's bug). Each reports median latency + IQR
over 5 repeats (not a single run), with cv2.setNumThreads set explicitly so
the three detectors are compared under the same threading configuration.

Energy was never measured in Phase 1, and this project does NOT call itself
"green" without measuring it: powermetrics (the only plausible energy-
measurement path on macOS) was checked and requires root
(`powermetrics must be invoked as the superuser`), which this automated run
does not use. So the title/claims in this project were changed from "green"
to "CPU-efficient", and this file records that explicitly rather than
quietly estimating energy from CPU% (CPU% is not energy and is not reported
as if it were).

No own video was available in data/ when this was run (data/own_video.mp4
absent) -- this benchmarks on a fixed 200-image subset of the WIDER FACE
validation set instead, and says so in its own output.
"""
import json
import os
import platform
import subprocess
import sys

import pandas as pd
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
WORKER = os.path.join(HERE, "_benchmark_worker.py")
VIDEO_PATH = os.path.join(HERE, "..", "data", "own_video.mp4")

DETECTOR_NAMES = ["haar_cascade", "mobilenet_ssd", "mediapipe_blazeface"]
N_REPEATS = 5
N_IMAGES = 200
NUM_THREADS = 1  # explicit and fixed, so detectors are compared under identical threading


def run_benchmarks(using_video):
    rows = []
    for name in DETECTOR_NAMES:
        print(f"[benchmark] {name}: launching isolated subprocess (num_threads={NUM_THREADS}) ...")
        cmd = [sys.executable, WORKER, name, str(N_REPEATS), str(N_IMAGES), str(NUM_THREADS)]
        if using_video:
            cmd.append(VIDEO_PATH)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[error] {name} subprocess failed:\n{proc.stderr}")
            continue
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(result)
        print(result)
    return rows


def write_environment_info(using_video):
    lines = []
    try:
        if platform.system() == "Darwin":
            cpu_model = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        else:
            cpu_model = "unknown (non-macOS; parse /proc/cpuinfo if needed)"
    except Exception:
        cpu_model = "unknown"

    lines.append(f"CPU model: {cpu_model}")
    lines.append(f"Physical cores: {psutil.cpu_count(logical=False)}")
    lines.append(f"Logical cores: {psutil.cpu_count(logical=True)}")
    lines.append(f"OS: {platform.platform()}")
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"cv2.setNumThreads used for benchmark: {NUM_THREADS}")
    lines.append(
        "Benchmark source: "
        + ("data/own_video.mp4" if using_video else "fixed 200-image WIDER FACE validation subset (no own video present)")
    )
    lines.append("")
    lines.append("Energy measurement: NOT performed. powermetrics is present on this machine")
    lines.append("but requires root ('powermetrics must be invoked as the superuser'); this")
    lines.append("automated run does not use sudo. CPU% / latency are reported as efficiency")
    lines.append("proxies only -- not energy -- and the project is described as")
    lines.append('"CPU-efficient", not "green", because of this.')
    lines.append("")
    lines.append("Installed library versions (subset):")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        wanted = ("opencv-python", "mediapipe", "numpy", "pandas", "matplotlib", "psutil", "pytest", "tabulate")
        for line in freeze.splitlines():
            if line.split("==")[0].lower() in wanted:
                lines.append(f"  {line}")
    except Exception as e:
        lines.append(f"  (pip freeze failed: {e})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "environment.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved -> {RESULTS_DIR}/environment.txt")


def main():
    using_video = os.path.isfile(VIDEO_PATH)
    if not using_video:
        print("No data/own_video.mp4 found -- benchmarking on a fixed WIDER FACE image subset instead.")

    rows = run_benchmarks(using_video)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "benchmark.csv"), index=False)
    print(f"Saved -> {RESULTS_DIR}/benchmark.csv")

    write_environment_info(using_video)


if __name__ == "__main__":
    main()
