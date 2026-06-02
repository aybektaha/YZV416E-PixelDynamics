"""
metrics.py
==========
DAVIS evaluation primitives: region similarity J (Jaccard / IoU) and boundary
accuracy F (contour F-measure), following the official DAVIS definitions
(Perazzi et al., CVPR 2016). These are reimplemented here so the pipeline runs
without external packages; for the final reported numbers you may cross-check
against the official davis2017-evaluation toolkit.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Region similarity (J) -- Jaccard index between binary masks
# --------------------------------------------------------------------------- #
def db_eval_iou(gt, pred):
    """Jaccard / IoU between two binary masks. Empty-vs-empty counts as 1.0."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0
    return inter / union


# --------------------------------------------------------------------------- #
# Boundary accuracy (F) -- contour F-measure with a small matching tolerance
# --------------------------------------------------------------------------- #
def _seg2bmap(seg):
    """Boundary map of a binary segmentation (1px contour)."""
    seg = seg.astype(bool)
    h, w = seg.shape
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def db_eval_boundary(gt, pred, bound_th=0.008):
    """Boundary F-measure between two binary masks (DAVIS definition).

    Tolerance matching uses a Euclidean distance transform (O(N)) instead of
    iterative dilation -- same result, ~20x faster, which matters across the
    ablation grid (tens of thousands of frames).
    """
    from scipy.ndimage import distance_transform_edt

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(gt.shape))
    radius = max(int(bound_pix), 1)

    gt_b = _seg2bmap(gt)
    pred_b = _seg2bmap(pred)

    # "within tolerance of a boundary pixel" == distance to nearest boundary <= radius
    gt_dil = distance_transform_edt(~gt_b) <= radius
    pred_dil = distance_transform_edt(~pred_b) <= radius

    gt_match = gt_b & pred_dil
    pred_match = pred_b & gt_dil

    n_gt = gt_b.sum()
    n_pred = pred_b.sum()

    if n_pred == 0 and n_gt > 0:
        return 0.0
    if n_pred > 0 and n_gt == 0:
        return 0.0
    if n_pred == 0 and n_gt == 0:
        return 1.0

    precision = pred_match.sum() / n_pred
    recall = gt_match.sum() / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Multi-instance matching (DAVIS unsupervised track)
# --------------------------------------------------------------------------- #
def _instances(label_img):
    """Sorted list of non-zero instance ids present in a label image."""
    ids = np.unique(label_img)
    return [i for i in ids if i != 0]


def evaluate_sequence(gt_masks, pred_masks):
    """Evaluate one sequence (list of label images) -> mean J, F, J&F.

    Unsupervised metric: predicted instances are matched to GT instances with a
    one-to-one assignment that maximises mean J (Hungarian), shared across all
    frames of the sequence; then J and F are averaged over GT objects & frames.
    """
    from scipy.optimize import linear_sum_assignment

    gt_ids = sorted(set().union(*[_instances(g) for g in gt_masks]))
    pred_ids = sorted(set().union(*[_instances(p) for p in pred_masks]))

    if not gt_ids:
        return {"J": float("nan"), "F": float("nan"), "JF": float("nan")}
    if not pred_ids:
        return {"J": 0.0, "F": 0.0, "JF": 0.0}

    # cost = -mean IoU over frames for every (gt, pred) instance pair
    cost = np.zeros((len(gt_ids), len(pred_ids)))
    for gi, g_id in enumerate(gt_ids):
        for pi, p_id in enumerate(pred_ids):
            ious = [db_eval_iou(g == g_id, p == p_id)
                    for g, p in zip(gt_masks, pred_masks)]
            cost[gi, pi] = -np.mean(ious)

    row, col = linear_sum_assignment(cost)
    assign = {gt_ids[r]: pred_ids[c] for r, c in zip(row, col)}

    j_per_obj, f_per_obj = [], []
    for g_id in gt_ids:
        p_id = assign.get(g_id, None)
        js, fs = [], []
        for g, p in zip(gt_masks, pred_masks):
            g_bin = (g == g_id)
            p_bin = (p == p_id) if p_id is not None else np.zeros_like(g_bin)
            js.append(db_eval_iou(g_bin, p_bin))
            fs.append(db_eval_boundary(g_bin, p_bin))
        j_per_obj.append(np.mean(js))
        f_per_obj.append(np.mean(fs))

    J = float(np.mean(j_per_obj))
    F = float(np.mean(f_per_obj))
    return {"J": J, "F": F, "JF": (J + F) / 2.0}
