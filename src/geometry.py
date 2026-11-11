"""Pure box-geometry helpers shared by evaluate.py, improve.py, privacy_check.py.

Extracted out of evaluate.py (Phase 1) so they can be unit-tested in isolation
(tests/test_geometry.py) and reused without importing the whole eval pipeline.
"""

IOU_THRESHOLD = 0.5


def iou(box_a, box_b):
    """(x, y, w, h) boxes -> intersection-over-union in [0, 1]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, threshold=IOU_THRESHOLD):
    """Greedy one-to-one IoU matching.

    Each predicted box is matched to its best remaining (unmatched) ground-truth
    box; a match counts as a true positive only if IoU >= threshold. Returns
    (true_positives, false_positives, false_negatives) as counts.
    """
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            cur_iou = iou(pb, gb)
            if cur_iou > best_iou:
                best_iou, best_j = cur_iou, j
        if best_iou >= threshold:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def match_boxes_detailed(pred_boxes, gt_boxes, threshold=IOU_THRESHOLD):
    """Like match_boxes, but also returns which gt indices were matched (TPs)
    and which were missed (FNs), for bucketed / per-attribute analysis.

    Returns (matched_gt_indices: set[int], unmatched_gt_indices: set[int], tp, fp).
    """
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            cur_iou = iou(pb, gb)
            if cur_iou > best_iou:
                best_iou, best_j = cur_iou, j
        if best_iou >= threshold:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_boxes) - tp
    unmatched = set(range(len(gt_boxes))) - matched_gt
    return matched_gt, unmatched, tp, fp


def nms(boxes, scores, iou_threshold=0.4):
    """Greedy non-maximum suppression. boxes: list of (x,y,w,h), scores: list of float.
    Returns the indices (into the input lists) to keep, highest score first.
    """
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_threshold]
    return keep
