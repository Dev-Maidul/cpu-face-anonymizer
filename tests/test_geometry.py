import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import iou, match_boxes, match_boxes_detailed, nms  # noqa: E402


class TestIoU:
    def test_identical_boxes(self):
        assert iou((10, 10, 20, 20), (10, 10, 20, 20)) == 1.0

    def test_no_overlap(self):
        assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0

    def test_partial_overlap(self):
        # box A: (0,0)-(10,10), box B: (5,5)-(15,15) -> intersection 5x5=25, union 200-25=175
        val = iou((0, 0, 10, 10), (5, 5, 10, 10))
        assert abs(val - 25 / 175) < 1e-9

    def test_touching_edges_no_overlap(self):
        # box B starts exactly where box A ends -> zero-area intersection
        assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0

    def test_zero_size_box_returns_zero(self):
        assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0
        assert iou((0, 0, 10, 10), (0, 0, 0, 5)) == 0.0

    def test_contained_box(self):
        # small box fully inside a larger one
        val = iou((0, 0, 10, 10), (2, 2, 4, 4))
        assert abs(val - (4 * 4) / (10 * 10)) < 1e-9


class TestMatchBoxes:
    def test_empty_predictions(self):
        tp, fp, fn = match_boxes([], [(0, 0, 10, 10), (20, 20, 10, 10)])
        assert (tp, fp, fn) == (0, 0, 2)

    def test_empty_ground_truth(self):
        tp, fp, fn = match_boxes([(0, 0, 10, 10)], [])
        assert (tp, fp, fn) == (0, 1, 0)

    def test_both_empty(self):
        assert match_boxes([], []) == (0, 0, 0)

    def test_perfect_match(self):
        boxes = [(0, 0, 10, 10), (50, 50, 10, 10)]
        tp, fp, fn = match_boxes(boxes, boxes)
        assert (tp, fp, fn) == (2, 0, 0)

    def test_one_to_one_greedy_no_double_match(self):
        # two predictions both overlap the SAME single ground-truth box well;
        # only one may claim it as a true positive, the other is a false positive
        gt = [(0, 0, 10, 10)]
        preds = [(0, 0, 10, 10), (1, 1, 10, 10)]
        tp, fp, fn = match_boxes(preds, gt)
        assert tp == 1
        assert fp == 1
        assert fn == 0

    def test_below_threshold_is_not_a_match(self):
        # small overlap, IoU well under 0.5
        tp, fp, fn = match_boxes([(0, 0, 10, 10)], [(8, 8, 10, 10)], threshold=0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 1

    def test_detailed_variant_reports_matched_and_unmatched_indices(self):
        gt = [(0, 0, 10, 10), (100, 100, 10, 10)]
        preds = [(0, 0, 10, 10)]
        matched, unmatched, tp, fp = match_boxes_detailed(preds, gt)
        assert matched == {0}
        assert unmatched == {1}
        assert tp == 1
        assert fp == 0


class TestNMS:
    def test_keeps_highest_score_and_suppresses_overlap(self):
        boxes = [(0, 0, 10, 10), (1, 1, 10, 10), (100, 100, 10, 10)]
        scores = [0.9, 0.95, 0.5]
        keep = nms(boxes, scores, iou_threshold=0.3)
        # box 1 (highest score) suppresses overlapping box 0; box 2 is far away and kept
        assert set(keep) == {1, 2}

    def test_no_boxes(self):
        assert nms([], []) == []

    def test_non_overlapping_boxes_all_kept(self):
        boxes = [(0, 0, 5, 5), (100, 100, 5, 5), (200, 200, 5, 5)]
        scores = [0.5, 0.6, 0.7]
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert set(keep) == {0, 1, 2}
