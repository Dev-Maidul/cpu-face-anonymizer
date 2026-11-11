"""Shared helpers for loading the full WIDER FACE ground truth (data_prep.py output)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "WIDER_val", "images")
GT_PATH = os.path.join(DATA_DIR, "ground_truth_full.jsonl")

# WIDER FACE's own height-bucket boundaries don't exist; these are a reasonable
# standard choice (small/medium/large-ish split) used throughout this project.
SIZE_BUCKETS = [(0, 20), (20, 50), (50, 100), (100, float("inf"))]
SIZE_BUCKET_LABELS = ["<20px", "20-50px", "50-100px", ">100px"]

BLUR_LABELS = {0: "clear", 1: "normal", 2: "heavy"}


def load_ground_truth(path=GT_PATH):
    """Returns list of {"path": str, "boxes": [{"x","y","w","h","blur",...}, ...]}."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def image_abspath(record, images_dir=IMAGES_DIR):
    return os.path.join(images_dir, record["path"])


def valid_boxes(record):
    """Ground-truth boxes with invalid=1 excluded, as (x, y, w, h) tuples."""
    return [(b["x"], b["y"], b["w"], b["h"]) for b in record["boxes"] if b["invalid"] == 0]


def valid_box_records(record):
    """Same as valid_boxes but keeps the full attribute dict per box."""
    return [b for b in record["boxes"] if b["invalid"] == 0]


def size_bucket(h):
    for (lo, hi), label in zip(SIZE_BUCKETS, SIZE_BUCKET_LABELS):
        if lo <= h < hi:
            return label
    return SIZE_BUCKET_LABELS[-1]


def blur_label(blur_value):
    return BLUR_LABELS.get(blur_value, "unknown")
