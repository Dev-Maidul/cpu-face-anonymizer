"""MediaPipe BlazeFace -- FULL RANGE variant (vs. the short-range model used
as the project's main 'mediapipe_blazeface' detector). Added specifically to
test whether the short-range model's poor recall on WIDER FACE's small/crowd
faces (see results/eval_by_bucket.csv) is a fundamental BlazeFace limitation
or specific to the short-range variant. Not part of the original 3-detector
benchmark/eval (that was already run before this model was added) -- see
README for how its numbers are reported separately.
"""
import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "blaze_face_full_range.tflite")

NAME = "mediapipe_blazeface_full"
MODEL_SIZE_MB = os.path.getsize(_MODEL_PATH) / (1024 * 1024)
HAS_CONFIDENCE_SCORE = True
OPERATING_THRESHOLD = 0.5

_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=_MODEL_PATH),
    min_detection_confidence=OPERATING_THRESHOLD,
)
_detector = vision.FaceDetector.create_from_options(_options)

_options_low = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=_MODEL_PATH),
    min_detection_confidence=0.01,
)
_detector_low = vision.FaceDetector.create_from_options(_options_low)


def _run(detector, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return detector.detect(mp_image)


def detect_faces_with_scores(frame_bgr, conf_threshold=0.0):
    result = _run(_detector_low, frame_bgr)
    out = []
    for det in result.detections:
        score = det.categories[0].score if det.categories else 0.0
        if score < conf_threshold:
            continue
        bbox = det.bounding_box
        out.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height, float(score)))
    return out


def detect_faces(frame_bgr):
    result = _run(_detector, frame_bgr)
    return [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)
            for d in result.detections]
