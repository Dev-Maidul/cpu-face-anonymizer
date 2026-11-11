"""Anonymize detected face regions in a frame via Gaussian blur or pixelation."""
import cv2


def blur_faces(frame_bgr, boxes, ksize=51):
    out = frame_bgr.copy()
    k = ksize if ksize % 2 == 1 else ksize + 1
    for (x, y, w, h) in boxes:
        x, y = max(0, x), max(0, y)
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def pixelate_faces(frame_bgr, boxes, blocks=10):
    out = frame_bgr.copy()
    for (x, y, w, h) in boxes:
        x, y = max(0, x), max(0, y)
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        # resize back to the ROI's *actual* (possibly border-clipped) shape,
        # not the requested (w, h) -- those can differ when the box runs past
        # the frame edge, which previously crashed with a broadcast error.
        roi_h, roi_w = roi.shape[:2]
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        out[y:y + roi_h, x:x + roi_w] = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    return out


if __name__ == "__main__":
    import os
    from detectors import ALL_DETECTORS

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "anonymized_samples")
    os.makedirs(results_dir, exist_ok=True)

    # pick a sample image with several visible faces for a clear before/after demo
    demo_filename = "2--Demonstration__2_Demonstration_Demonstrators_2_488.jpg"
    img_path = os.path.join(sample_dir, demo_filename)
    frame = cv2.imread(img_path)
    cv2.imwrite(os.path.join(results_dir, "original.jpg"), frame)

    for det in ALL_DETECTORS:
        boxes = det.detect_faces(frame)
        blurred = blur_faces(frame, boxes)
        pixelated = pixelate_faces(frame, boxes)
        cv2.imwrite(os.path.join(results_dir, f"{det.NAME}_blur.jpg"), blurred)
        cv2.imwrite(os.path.join(results_dir, f"{det.NAME}_pixelate.jpg"), pixelated)
        print(f"{det.NAME}: {len(boxes)} faces anonymized -> saved samples")
