"""
Download and parse the FULL WIDER FACE validation split (3226 images) with
all ground-truth attribute flags (blur, expression, illumination, invalid,
occlusion, pose), for a rigorous evaluation.

Images are NOT copied/flattened into a separate folder (that was fine for a
60-image sample in Phase 1, not for 3226 images / ~1GB) -- the extracted
WIDER_val/images tree is kept as-is, and the ground-truth file stores paths
relative to it. Run this once; it downloads ~363MB and keeps the extracted
images on disk (not committed to git -- see .gitignore).
"""
import json
import os
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
SPLIT_DIR = os.path.join(DATA_DIR, "wider_face_split", "wider_face_split")
GT_FILE = os.path.join(SPLIT_DIR, "wider_face_val_bbx_gt.txt")
VAL_ZIP = os.path.join(DATA_DIR, "WIDER_val.zip")
SPLIT_ZIP = os.path.join(DATA_DIR, "wider_face_split.zip")
VAL_EXTRACT_DIR = os.path.join(DATA_DIR, "WIDER_val")
IMAGES_DIR = os.path.join(VAL_EXTRACT_DIR, "images")
GT_OUT_PATH = os.path.join(DATA_DIR, "ground_truth_full.jsonl")

HF_BASE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"

# WIDER FACE attribute column order, per wider_face_split/readme.txt:
# x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
ATTR_NAMES = ["blur", "expression", "illumination", "invalid", "occlusion", "pose"]


def download_if_missing():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(SPLIT_ZIP) and not os.path.isdir(SPLIT_DIR):
        print("Downloading wider_face_split.zip (annotations, ~3.6MB)...")
        urllib.request.urlretrieve(f"{HF_BASE}/wider_face_split.zip", SPLIT_ZIP)
    if not os.path.isdir(SPLIT_DIR):
        with zipfile.ZipFile(SPLIT_ZIP, "r") as z:
            z.extractall(os.path.join(DATA_DIR, "wider_face_split"))
    if not os.path.isfile(VAL_ZIP) and not os.path.isdir(VAL_EXTRACT_DIR):
        print("Downloading WIDER_val.zip (validation images, ~363MB)...")
        urllib.request.urlretrieve(f"{HF_BASE}/WIDER_val.zip", VAL_ZIP)


def extract_val_zip():
    if os.path.isdir(VAL_EXTRACT_DIR):
        print("WIDER_val already extracted.")
        return
    print("Extracting WIDER_val.zip (this is the full ~3226-image validation set)...")
    with zipfile.ZipFile(VAL_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("Done extracting.")


def parse_gt(gt_path):
    """Returns list of {"path": str, "boxes": [{"x","y","w","h", <attrs>}, ...]}."""
    with open(gt_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    records = []
    i = 0
    while i < len(lines):
        img_path = lines[i]
        i += 1
        num_boxes = int(lines[i])
        i += 1
        n_lines_to_read = max(num_boxes, 1)  # WIDER FACE quirk: 0 boxes still has 1 dummy line
        boxes = []
        for j in range(n_lines_to_read):
            parts = lines[i + j].split()
            if num_boxes > 0 and len(parts) >= 10:
                x, y, w, h = map(int, parts[:4])
                attrs = dict(zip(ATTR_NAMES, map(int, parts[4:10])))
                boxes.append({"x": x, "y": y, "w": w, "h": h, **attrs})
        i += n_lines_to_read
        records.append({"path": img_path, "boxes": boxes})
    return records


def build_ground_truth():
    records = parse_gt(GT_FILE)
    with open(GT_OUT_PATH, "w") as out:
        for rec in records:
            out.write(json.dumps(rec) + "\n")

    n_images = len(records)
    n_boxes = sum(len(r["boxes"]) for r in records)
    n_invalid = sum(1 for r in records for b in r["boxes"] if b["invalid"] == 1)
    print(f"Wrote {GT_OUT_PATH}")
    print(f"  {n_images} images, {n_boxes} boxes total, {n_invalid} flagged invalid=1 (excluded from eval)")


if __name__ == "__main__":
    download_if_missing()
    extract_val_zip()
    build_ground_truth()
    # remove only the zip archives; keep the extracted images (needed for the full eval)
    for path in (VAL_ZIP, SPLIT_ZIP):
        if os.path.isfile(path):
            os.remove(path)
