import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anonymize import blur_faces, pixelate_faces  # noqa: E402


def make_frame(h=100, w=100):
    # deterministic non-uniform content so "it actually changed the pixels" is a real check
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


class TestBlurFaces:
    def test_no_boxes_returns_unchanged_copy(self):
        frame = make_frame()
        out = blur_faces(frame, [])
        assert np.array_equal(out, frame)
        assert out is not frame  # must be a copy, not the same array

    def test_normal_box_changes_pixels_inside_it(self):
        frame = make_frame()
        out = blur_faces(frame, [(10, 10, 30, 30)], ksize=15)
        assert not np.array_equal(out[10:40, 10:40], frame[10:40, 10:40])
        # outside the box must be untouched
        assert np.array_equal(out[:10, :], frame[:10, :])

    def test_box_partially_outside_right_bottom_edge_does_not_crash(self):
        frame = make_frame(h=50, w=50)
        # box extends past the frame's right/bottom edge
        out = blur_faces(frame, [(40, 40, 30, 30)], ksize=15)
        assert out.shape == frame.shape

    def test_box_fully_outside_frame_does_not_crash(self):
        frame = make_frame(h=50, w=50)
        out = blur_faces(frame, [(200, 200, 10, 10)], ksize=15)
        assert np.array_equal(out, frame)

    def test_zero_width_box_does_not_crash(self):
        frame = make_frame()
        out = blur_faces(frame, [(10, 10, 0, 20)], ksize=15)
        assert np.array_equal(out, frame)

    def test_zero_height_box_does_not_crash(self):
        frame = make_frame()
        out = blur_faces(frame, [(10, 10, 20, 0)], ksize=15)
        assert np.array_equal(out, frame)

    def test_negative_origin_box_is_clamped_not_crashed(self):
        frame = make_frame(h=50, w=50)
        out = blur_faces(frame, [(-5, -5, 20, 20)], ksize=15)
        assert out.shape == frame.shape

    def test_even_ksize_is_made_odd_internally(self):
        frame = make_frame()
        # must not raise (cv2.GaussianBlur requires an odd kernel size)
        out = blur_faces(frame, [(10, 10, 30, 30)], ksize=10)
        assert out.shape == frame.shape


class TestPixelateFaces:
    def test_no_boxes_returns_unchanged_copy(self):
        frame = make_frame()
        out = pixelate_faces(frame, [])
        assert np.array_equal(out, frame)
        assert out is not frame

    def test_normal_box_changes_pixels_inside_it(self):
        frame = make_frame()
        out = pixelate_faces(frame, [(10, 10, 30, 30)], blocks=4)
        assert not np.array_equal(out[10:40, 10:40], frame[10:40, 10:40])
        assert np.array_equal(out[:10, :], frame[:10, :])

    def test_box_partially_outside_frame_does_not_crash(self):
        frame = make_frame(h=50, w=50)
        out = pixelate_faces(frame, [(40, 40, 30, 30)], blocks=4)
        assert out.shape == frame.shape

    def test_zero_size_box_does_not_crash(self):
        frame = make_frame()
        out = pixelate_faces(frame, [(10, 10, 0, 0)], blocks=4)
        assert np.array_equal(out, frame)

    def test_multiple_boxes_each_anonymized(self):
        frame = make_frame(h=200, w=200)
        boxes = [(10, 10, 30, 30), (100, 100, 30, 30)]
        out = pixelate_faces(frame, boxes, blocks=4)
        assert not np.array_equal(out[10:40, 10:40], frame[10:40, 10:40])
        assert not np.array_equal(out[100:130, 100:130], frame[100:130, 100:130])
