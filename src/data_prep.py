"""
Parse WIDER FACE validation ground-truth and select a small, manageable
sample of images (with their bounding boxes) for detector evaluation.

Run after data/WIDER_val.zip has finished downloading and been extracted.
"""
import os
import random
import shutil
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
SPLIT_DIR = os.path.join(DATA_DIR, "wider_face_split", "wider_face_split")
GT_FILE = os.path.join(SPLIT_DIR, "wider_face_val_bbx_gt.txt")
VAL_ZIP = os.path.join(DATA_DIR, "WIDER_val.zip")
SPLIT_ZIP = os.path.join(DATA_DIR, "wider_face_split.zip")
VAL_EXTRACT_DIR = os.path.join(DATA_DIR, "WIDER_val")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
N_IMAGES = 60  # keep it small and fast for a CPU-only course project
RANDOM_SEED = 42

HF_BASE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"


def download_if_missing():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(SPLIT_ZIP):
        print("Downloading wider_face_split.zip (annotations, ~3.6MB)...")
        urllib.request.urlretrieve(f"{HF_BASE}/wider_face_split.zip", SPLIT_ZIP)
    if not os.path.isdir(SPLIT_DIR):
        with zipfile.ZipFile(SPLIT_ZIP, "r") as z:
            z.extractall(os.path.join(DATA_DIR, "wider_face_split"))
    if not os.path.isfile(VAL_ZIP) and not os.path.isdir(VAL_EXTRACT_DIR):
        print("Downloading WIDER_val.zip (validation images, ~363MB)...")
        urllib.request.urlretrieve(f"{HF_BASE}/WIDER_val.zip", VAL_ZIP)


def parse_gt(gt_path):
    """Returns dict: {relative_image_path: [(x, y, w, h), ...]}"""
    annotations = {}
    with open(gt_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        img_path = lines[i]
        i += 1
        num_boxes = int(lines[i])
        i += 1
        boxes = []
        # WIDER FACE quirk: num_boxes can be 0 but still has one dummy line
        n_lines_to_read = max(num_boxes, 1)
        for j in range(n_lines_to_read):
            parts = lines[i + j].split()
            x, y, w, h = map(int, parts[:4])
            if num_boxes > 0:
                boxes.append((x, y, w, h))
        i += n_lines_to_read
        annotations[img_path] = boxes
    return annotations


def extract_val_zip():
    if os.path.isdir(VAL_EXTRACT_DIR):
        print("WIDER_val already extracted.")
        return
    print("Extracting WIDER_val.zip ...")
    with zipfile.ZipFile(VAL_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("Done extracting.")


def build_sample():
    annotations = parse_gt(GT_FILE)
    # keep only images that actually have at least 1 face, for a meaningful eval
    valid_items = [(k, v) for k, v in annotations.items() if len(v) > 0]
    random.seed(RANDOM_SEED)
    random.shuffle(valid_items)
    chosen = valid_items[:N_IMAGES]

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    gt_out_path = os.path.join(SAMPLE_DIR, "ground_truth.txt")
    with open(gt_out_path, "w") as out:
        copied = 0
        for rel_path, boxes in chosen:
            src = os.path.join(VAL_EXTRACT_DIR, "images", rel_path)
            if not os.path.isfile(src):
                continue
            flat_name = rel_path.replace("/", "__")
            dst = os.path.join(SAMPLE_DIR, flat_name)
            shutil.copy(src, dst)
            out.write(flat_name + "\n")
            out.write(str(len(boxes)) + "\n")
            for (x, y, w, h) in boxes:
                out.write(f"{x} {y} {w} {h}\n")
            copied += 1
    print(f"Copied {copied} images into {SAMPLE_DIR}")
    print(f"Ground truth written to {gt_out_path}")


if __name__ == "__main__":
    download_if_missing()
    extract_val_zip()
    build_sample()
    # clean up the large intermediates, keep only data/sample
    for path in (VAL_ZIP, SPLIT_ZIP, VAL_EXTRACT_DIR):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
