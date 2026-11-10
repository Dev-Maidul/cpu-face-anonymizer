"""MediaPipe BlazeFace (short-range) face detector — lightweight deep model."""
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "blaze_face_short_range.tflite")

NAME = "mediapipe_blazeface"
MODEL_SIZE_MB = os.path.getsize(_MODEL_PATH) / (1024 * 1024)

_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=_MODEL_PATH),
    min_detection_confidence=0.5,
)
_detector = vision.FaceDetector.create_from_options(_options)


def detect_faces(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = _detector.detect(mp_image)
    boxes = []
    for det in result.detections:
        bbox = det.bounding_box
        boxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))
    return boxes
