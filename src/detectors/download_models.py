"""
Download the pretrained weight files the detectors need.
Not committed to the repo (large binaries) -- run this once after cloning:

    python src/detectors/download_models.py
"""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FILES = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.isfile(dest):
            print(f"[skip] {filename} already present")
            continue
        print(f"[download] {filename} <- {url}")
        urllib.request.urlretrieve(url, dest)
    print("All model files ready.")


if __name__ == "__main__":
    main()
