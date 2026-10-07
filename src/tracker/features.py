"""Pairwise motion and box features for the association model.

Feature order is ``tracker.constants.FEATURE_NAMES`` and is the training and
inference contract. Distances are divided by the last observed box height.
"""

from __future__ import annotations

import math

import numpy as np

from tracker.constants import FEATURE_NAMES, LEARNED_MAX_AGE
from tracker.geometry import box_iou, center, height, width


def _log_ratio(a: float, b: float) -> float:
    if a <= 1e-6 or b <= 1e-6:
        return 0.0
    return math.log(a / b)


def pair_features(
    pred_xyxy: list[float],
    last_xyxy: list[float],
    obs_vel: tuple[float, float] | list[float],
    gap: int,
    det_xyxy: list[float],
    det_conf: float,
) -> list[float]:
    """Features for one track-detection pair. ``gap`` is frames since the last observation."""
    last_h = height(last_xyxy)
    scale = last_h if last_h > 1.0 else 1.0
    pc = center(pred_xyxy)
    lc = center(last_xyxy)
    dc = center(det_xyxy)
    dist_pred = math.hypot(dc[0] - pc[0], dc[1] - pc[1]) / scale
    dist_last = math.hypot(dc[0] - lc[0], dc[1] - lc[1]) / scale
    gap_f = float(gap if gap > 0 else 1)
    expected_x = lc[0] + float(obs_vel[0]) * gap_f
    expected_y = lc[1] + float(obs_vel[1]) * gap_f
    vel_error = math.hypot(dc[0] - expected_x, dc[1] - expected_y) / scale
    last_w = width(last_xyxy)
    det_w = width(det_xyxy)
    det_h = height(det_xyxy)
    last_aspect = 0.0 if last_h <= 1e-6 else last_w / last_h
    det_aspect = 0.0 if det_h <= 1e-6 else det_w / det_h
    return [
        box_iou(pred_xyxy, det_xyxy),
        box_iou(last_xyxy, det_xyxy),
        dist_pred,
        dist_last,
        _log_ratio(det_h, last_h if last_h > 1e-6 else 1.0),
        _log_ratio(det_w, last_w if last_w > 1e-6 else 1.0),
        abs(det_aspect - last_aspect),
        float(gap) / float(LEARNED_MAX_AGE),
        vel_error,
        float(det_conf),
    ]


def feature_matrix(rows: list[list[float]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float64), std.astype(np.float64)


def apply_normalizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std
