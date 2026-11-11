"""Haar Cascade (Viola-Jones) face detector — classical CV, no deep learning."""
import cv2

_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_detector = cv2.CascadeClassifier(_cascade_path)

NAME = "haar_cascade"
MODEL_SIZE_MB = 0  # ships inside opencv-python, no separate file to measure
HAS_CONFIDENCE_SCORE = False  # minNeighbors is a discrete knob, not a continuous
# score -- no honest PR-curve sweep is possible for this detector (see README).


def detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
