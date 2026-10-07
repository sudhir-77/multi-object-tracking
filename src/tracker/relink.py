"""Link SORT tracklets whose motion and box features agree.

SORT's boxes stay as they are. A second pass scores pairs of tracklets that
do not overlap in time, with a gap of at most ``max_age`` frames, and merges
the ids when the association model is confident and the match is unique.
"""

from __future__ import annotations

import numpy as np
import torch

from tracker.constants import LEARNED_MAX_AGE
from tracker.features import apply_normalizer, pair_features
from tracker.geometry import center, z_to_xyxy
from tracker.kalman import KalmanBox, predict_state
from tracker.model import AssociationNet


class Tracklet:
    def __init__(self, track_id: int) -> None:
        self.id = track_id
        self.frames: list[int] = []
        self.boxes: list[list[float]] = []
        self.end_x: list[float] | None = None
        self.end_P: list[list[float]] | None = None
        self.last_xyxy: list[float] | None = None
        self.obs_vel = [0.0, 0.0]


def motion_gate(feats: list[float]) -> bool:
    """Keep pairs the Kalman prediction could still explain."""
    iou_pred = feats[0]
    dist_pred = feats[2]
    dist_last = feats[3]
    if iou_pred >= 0.05:
        return True
    if dist_pred <= 2.0 or dist_last <= 3.0:
        return True
    return False


def tracklets_from_rows(rows: list[tuple] | np.ndarray) -> list[Tracklet]:
    grouped: dict[int, Tracklet] = {}
    array = np.asarray(rows, dtype=np.float64) if not isinstance(rows, list) else None
    iterable = array if array is not None else rows
    if array is not None and array.size == 0:
        return []
    for row in iterable:
        frame = int(row[0])
        track_id = int(row[1])
        xywh = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
        x1, y1, w, h = xywh
        box = [x1, y1, x1 + w, y1 + h]
        track = grouped.get(track_id)
        if track is None:
            track = Tracklet(track_id)
            grouped[track_id] = track
        track.frames.append(frame)
        track.boxes.append(box)
    tracklets = list(grouped.values())
    for track in tracklets:
        order = np.argsort(track.frames)
        track.frames = [track.frames[i] for i in order]
        track.boxes = [track.boxes[i] for i in order]
        _freeze_end_state(track)
    return tracklets


def _freeze_end_state(track: Tracklet) -> None:
    kf = KalmanBox(track.boxes[0])
    prev = center(track.boxes[0])
    vel = [0.0, 0.0]
    last = track.boxes[0]
    for index in range(1, len(track.frames)):
        step = track.frames[index] - track.frames[index - 1]
        for _ in range(step):
            kf.predict()
        cx, cy = center(track.boxes[index])
        vel = [(cx - prev[0]) / step, (cy - prev[1]) / step]
        prev = (cx, cy)
        last = track.boxes[index]
        kf.update(track.boxes[index])
    track.end_x = kf.x[:]
    track.end_P = [row[:] for row in kf.P]
    track.last_xyxy = [float(v) for v in last]
    track.obs_vel = vel


def junction_features(earlier: Tracklet, later: Tracklet, max_age: int = LEARNED_MAX_AGE) -> list[float] | None:
    """Features for linking ``earlier`` into ``later``. None when the gap is illegal."""
    gap = later.frames[0] - earlier.frames[-1]
    if gap < 1 or gap > max_age or earlier.end_x is None or earlier.end_P is None or earlier.last_xyxy is None:
        return None
    x = earlier.end_x[:]
    P = [row[:] for row in earlier.end_P]
    for _ in range(gap):
        predict_state(x, P)
    pred_box = z_to_xyxy(x)
    if any(np.isnan(pred_box)):
        return None
    return pair_features(pred_box, earlier.last_xyxy, earlier.obs_vel, gap, later.boxes[0], 1.0)


def relink_rows(
    rows: list[tuple] | np.ndarray,
    model: AssociationNet,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
    max_age: int = LEARNED_MAX_AGE,
) -> list[tuple[int, int, float, float, float, float]]:
    """Greedy one-to-one merges, highest model score first."""
    if isinstance(rows, np.ndarray):
        if rows.size == 0:
            return []
        source = [tuple(float(v) for v in row[:6]) for row in rows]
        source = [(int(r[0]), int(r[1]), r[2], r[3], r[4], r[5]) for r in source]
    else:
        source = [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]
    tracklets = tracklets_from_rows(source)
    if len(tracklets) < 2:
        return source
    raw: list[list[float]] = []
    pairs: list[tuple[int, int]] = []
    for i, earlier in enumerate(tracklets):
        for j, later in enumerate(tracklets):
            if i == j:
                continue
            feats = junction_features(earlier, later, max_age=max_age)
            if feats is None or not motion_gate(feats):
                continue
            raw.append(feats)
            pairs.append((i, j))
    if not raw:
        return source
    normed = apply_normalizer(np.asarray(raw, dtype=np.float64), mean, std)
    with torch.inference_mode():
        logits = model(torch.as_tensor(normed, dtype=torch.float32, device=device))
        prob = torch.sigmoid(logits).detach().cpu().numpy()
    dists = np.asarray(raw, dtype=np.float64)[:, 2]
    # Prefer the closest motion-consistent tracklet. A low model score vetoes
    # the link; the distance cutoff is the threshold selected on validation.
    order = np.argsort(dists)
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for index in order:
        if float(dists[index]) > threshold:
            break
        if float(prob[index]) < 0.40:
            continue
        earlier, later = pairs[int(index)]
        if earlier in successor or later in predecessor:
            continue
        successor[earlier] = later
        predecessor[later] = earlier
    parent = {child: earlier for earlier, child in successor.items()}

    def root(index: int) -> int:
        seen: set[int] = set()
        while index in parent and index not in seen:
            seen.add(index)
            index = parent[index]
        return index

    rewritten: list[tuple[int, int, float, float, float, float]] = []
    id_to_index = {track.id: i for i, track in enumerate(tracklets)}
    for frame, track_id, x, y, w, h in source:
        keep = tracklets[root(id_to_index[track_id])].id
        rewritten.append((frame, keep, x, y, w, h))
    return rewritten
