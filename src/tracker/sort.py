"""SORT: constant-velocity Kalman filter and IoU Hungarian matching.

Track lifetime follows Bewley et al.: ``max_age=1``, ``min_hits=3``,
IoU gate 0.3. A new track does not count its initializing detection as a hit.
The C++ core in ``cpp/bindings.cpp`` follows this function step for step.
"""

from __future__ import annotations

import math

import numpy as np

from tracker.constants import FORBID_COST, SORT_IOU_THRESHOLD, SORT_MAX_AGE, SORT_MIN_HITS
from tracker.geometry import box_iou, xywh_to_xyxy, xyxy_to_xywh
from tracker.hungarian import hungarian
from tracker.kalman import KalmanBox


class _SortTrack:
    def __init__(self, xyxy: list[float], track_id: int) -> None:
        self.kf = KalmanBox(xyxy)
        self.id = track_id
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, xyxy: list[float]) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(xyxy)


def _associate(dets: list[list[float]], tracks: list[_SortTrack], iou_threshold: float) -> tuple[list[tuple[int, int]], list[int]]:
    n_d = len(dets)
    n_t = len(tracks)
    if n_d == 0 or n_t == 0:
        return [], list(range(n_d))
    preds = [tr.kf.xyxy() for tr in tracks]
    cost = np.empty((n_d, n_t), dtype=np.float64)
    for i in range(n_d):
        for j in range(n_t):
            iou = box_iou(dets[i], preds[j])
            if iou < iou_threshold:
                cost[i, j] = FORBID_COST
            else:
                cost[i, j] = 1.0 - iou
    pairs = hungarian(cost)
    matched: list[tuple[int, int]] = []
    matched_d: set[int] = set()
    for i, j in pairs:
        if cost[i, j] >= FORBID_COST * 0.5:
            continue
        matched.append((i, j))
        matched_d.add(i)
    unmatched_d = [i for i in range(n_d) if i not in matched_d]
    return matched, unmatched_d


def track_sort_frames(
    frames_xyxy: list[list[list[float]]],
    *,
    iou_threshold: float = SORT_IOU_THRESHOLD,
    max_age: int = SORT_MAX_AGE,
    min_hits: int = SORT_MIN_HITS,
) -> list[tuple[int, int, float, float, float, float]]:
    """Run SORT. ``frames_xyxy[t]`` is frame ``t + 1``.

    Returns rows ``(frame, id, x, y, w, h)`` with 1-based frame numbers.
    """
    tracks: list[_SortTrack] = []
    next_id = 1
    rows: list[tuple[int, int, float, float, float, float]] = []
    for frame_index, dets in enumerate(frames_xyxy):
        frame_count = frame_index + 1
        for tr in tracks:
            tr.predict()
        alive: list[_SortTrack] = []
        for tr in tracks:
            box = tr.kf.xyxy()
            if any(math.isnan(v) for v in box):
                continue
            alive.append(tr)
        tracks = alive
        matched, unmatched_d = _associate(dets, tracks, iou_threshold)
        for i, j in matched:
            tracks[j].update(dets[i])
        for i in unmatched_d:
            tracks.append(_SortTrack(dets[i], next_id))
            next_id += 1
        kept: list[_SortTrack] = []
        for tr in tracks:
            if tr.time_since_update < 1 and (tr.hit_streak >= min_hits or frame_count <= min_hits):
                xywh = xyxy_to_xywh(tr.kf.xyxy())
                rows.append((frame_count, tr.id, xywh[0], xywh[1], xywh[2], xywh[3]))
            if tr.time_since_update <= max_age:
                kept.append(tr)
        tracks = kept
    return rows


def track_sort_array(
    dets: np.ndarray,
    seq_length: int,
    iou_threshold: float = SORT_IOU_THRESHOLD,
    max_age: int = SORT_MAX_AGE,
    min_hits: int = SORT_MIN_HITS,
) -> np.ndarray:
    """SORT on an ``(N, 5)`` array of ``frame, x, y, w, h`` (1-based frames)."""
    frames: list[list[list[float]]] = [[] for _ in range(seq_length)]
    if dets.size:
        array = np.asarray(dets, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        for row in array:
            frame = int(row[0])
            if frame < 1 or frame > seq_length:
                continue
            frames[frame - 1].append(xywh_to_xyxy([float(row[1]), float(row[2]), float(row[3]), float(row[4])]))
    rows = track_sort_frames(
        frames,
        iou_threshold=iou_threshold,
        max_age=max_age,
        min_hits=min_hits,
    )
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)
