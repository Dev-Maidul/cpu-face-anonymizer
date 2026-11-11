"""MediaPipe BlazeFace (short-range) face detector — lightweight deep model."""
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "blaze_face_short_range.tflite")

NAME = "mediapipe_blazeface"
MODEL_SIZE_MB = os.path.getsize(_MODEL_PATH) / (1024 * 1024)

HAS_CONFIDENCE_SCORE = True
OPERATING_THRESHOLD = 0.5  # public, read by evaluate.py

_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=_MODEL_PATH),
    min_detection_confidence=OPERATING_THRESHOLD,
)
_detector = vision.FaceDetector.create_from_options(_options)

# Separate low-floor instance for PR-curve sweeps: MediaPipe filters internally
# at min_detection_confidence, so the fixed-threshold _detector above would
# truncate the sweep before we ever see lower-confidence candidates.
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
    """Boxes only, at the fixed operating threshold used by benchmark/anonymize."""
    result = _run(_detector, frame_bgr)
    return [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)
            for d in result.detections]
