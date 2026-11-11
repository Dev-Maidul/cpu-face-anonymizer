"""MobileNet-SSD (res10, Caffe) face detector via OpenCV DNN module — deep but compact."""
import os
import cv2

_DIR = os.path.join(os.path.dirname(__file__), "models")
_PROTOTXT = os.path.join(_DIR, "deploy.prototxt")
_WEIGHTS = os.path.join(_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

NAME = "mobilenet_ssd"
MODEL_SIZE_MB = os.path.getsize(_WEIGHTS) / (1024 * 1024)

_net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _WEIGHTS)
_CONF_THRESHOLD = 0.5
OPERATING_THRESHOLD = _CONF_THRESHOLD  # public alias, read by evaluate.py
HAS_CONFIDENCE_SCORE = True


def detect_faces_with_scores(frame_bgr, conf_threshold=0.0):
    """All raw detections with their confidence score, for PR-curve sweeps.
    conf_threshold=0.0 returns essentially everything the network proposes."""
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    _net.setInput(blob)
    detections = _net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        results.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence))
    return results


def detect_faces(frame_bgr):
    """Boxes only, at the fixed operating threshold used by benchmark/anonymize."""
    return [(x, y, w, h) for (x, y, w, h, score) in detect_faces_with_scores(frame_bgr, _CONF_THRESHOLD)]
