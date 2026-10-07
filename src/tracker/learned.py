"""Learned association on top of the same Kalman state model as SORT.

Differences from SORT, fixed before looking at held-out sequences:

* A track may be re-identified for up to ``max_age`` frames (30).
* The initializing detection counts as a hit, so confirmation matches the
  number of observations rather than observations after birth.
* Eligible pairs must pass a geometric gate. A PyTorch model ranks them.
  The acceptance threshold is the only value selected on the validation sequence.
* While a pair is eligible, cost is ``(1 - p) + 0.15 * min(normalized distances)``.
  The distance term is a fixed tie break, not a tuned weight.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tracker.constants import FORBID_COST, LEARNED_MAX_AGE, LEARNED_MIN_HITS
from tracker.features import apply_normalizer, pair_features
from tracker.geometry import xywh_to_xyxy, xyxy_to_xywh
from tracker.hungarian import hungarian
from tracker.kalman import KalmanBox
from tracker.model import AssociationNet

_TIE_BREAK = 0.15


class _LearnedTrack:
    def __init__(self, xyxy: list[float], conf: float, track_id: int) -> None:
        self.kf = KalmanBox(xyxy)
        self.id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.last_xyxy = [float(v) for v in xyxy]
        self.conf = float(conf)
        cx = (self.last_xyxy[0] + self.last_xyxy[2]) * 0.5
        cy = (self.last_xyxy[1] + self.last_xyxy[3]) * 0.5
        self.prev_center = (cx, cy)
        self.obs_vel = [0.0, 0.0]

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, xyxy: list[float], conf: float) -> None:
        gap = self.time_since_update if self.time_since_update > 0 else 1
        cx = (float(xyxy[0]) + float(xyxy[2])) * 0.5
        cy = (float(xyxy[1]) + float(xyxy[3])) * 0.5
        self.obs_vel = [(cx - self.prev_center[0]) / gap, (cy - self.prev_center[1]) / gap]
        self.prev_center = (cx, cy)
        self.last_xyxy = [float(v) for v in xyxy]
        self.conf = float(conf)
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(xyxy)


def _eligible(feats: list[float]) -> bool:
    iou_pred, iou_last, _dist_pred, dist_last = feats[0], feats[1], feats[2], feats[3]
    if iou_pred >= 0.15:
        return True
    if iou_last >= 0.20 and dist_last <= 2.0:
        return True
    return False


def _score_matrix(
    tracks: list[_LearnedTrack],
    dets: list[list[float]],
    confs: list[float],
    model: AssociationNet,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    n_d = len(dets)
    n_t = len(tracks)
    cost = np.full((n_d, n_t), FORBID_COST, dtype=np.float64)
    raw: list[list[float]] = []
    index: list[tuple[int, int]] = []
    for i, det in enumerate(dets):
        for j, tr in enumerate(tracks):
            feats = pair_features(
                tr.kf.xyxy(),
                tr.last_xyxy,
                tr.obs_vel,
                tr.time_since_update,
                det,
                confs[i],
            )
            if not _eligible(feats):
                continue
            raw.append(feats)
            index.append((i, j))
    if not raw:
        return cost
    arr = np.asarray(raw, dtype=np.float64)
    normed = apply_normalizer(arr, mean, std)
    with torch.inference_mode():
        logits = model(torch.as_tensor(normed, dtype=torch.float32, device=device))
        prob = torch.sigmoid(logits).detach().cpu().numpy()
    for (i, j), feats, p in zip(index, raw, prob):
        if float(p) < threshold:
            continue
        spatial = feats[2] if feats[2] < feats[3] else feats[3]
        cost[i, j] = (1.0 - float(p)) + _TIE_BREAK * spatial
    return cost


def track_learned_frames(
    frames_xyxy: list[list[list[float]]],
    frames_conf: list[list[float]],
    model: AssociationNet,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    threshold: float,
    max_age: int = LEARNED_MAX_AGE,
    min_hits: int = LEARNED_MIN_HITS,
    device: torch.device | None = None,
) -> list[tuple[int, int, float, float, float, float]]:
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    tracks: list[_LearnedTrack] = []
    next_id = 1
    rows: list[tuple[int, int, float, float, float, float]] = []
    for frame_index, dets in enumerate(frames_xyxy):
        frame_count = frame_index + 1
        confs = frames_conf[frame_index]
        for tr in tracks:
            tr.predict()
        alive: list[_LearnedTrack] = []
        for tr in tracks:
            box = tr.kf.xyxy()
            if any(math.isnan(v) for v in box):
                continue
            alive.append(tr)
        tracks = alive
        if dets and tracks:
            cost = _score_matrix(tracks, dets, confs, model, mean, std, threshold, device)
            pairs = hungarian(cost)
            matched_d: set[int] = set()
            matched: list[tuple[int, int]] = []
            for i, j in pairs:
                if cost[i, j] >= FORBID_COST * 0.5:
                    continue
                matched.append((i, j))
                matched_d.add(i)
            unmatched_d = [i for i in range(len(dets)) if i not in matched_d]
        else:
            matched = []
            unmatched_d = list(range(len(dets)))
        for i, j in matched:
            tracks[j].update(dets[i], confs[i])
        for i in unmatched_d:
            tracks.append(_LearnedTrack(dets[i], confs[i], next_id))
            next_id += 1
        kept: list[_LearnedTrack] = []
        for tr in tracks:
            if tr.time_since_update < 1 and (tr.hits >= min_hits or frame_count <= min_hits):
                xywh = xyxy_to_xywh(tr.kf.xyxy())
                rows.append((frame_count, tr.id, xywh[0], xywh[1], xywh[2], xywh[3]))
            if tr.time_since_update <= max_age:
                kept.append(tr)
        tracks = kept
    return rows


def frames_from_array(dets: np.ndarray, seq_length: int) -> tuple[list[list[list[float]]], list[list[float]]]:
    """``dets`` is ``frame, x, y, w, h, conf`` with 1-based frames."""
    frames: list[list[list[float]]] = [[] for _ in range(seq_length)]
    confs: list[list[float]] = [[] for _ in range(seq_length)]
    if dets.size == 0:
        return frames, confs
    array = np.asarray(dets, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    for row in array:
        frame = int(row[0])
        if frame < 1 or frame > seq_length:
            continue
        frames[frame - 1].append(xywh_to_xyxy([float(row[1]), float(row[2]), float(row[3]), float(row[4])]))
        confs[frame - 1].append(float(row[5]) if row.shape[0] > 5 else 1.0)
    return frames, confs
