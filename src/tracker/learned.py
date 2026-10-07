"""SORT boxes with a learned second pass that merges broken tracklets.

Frame-to-frame association is the SORT baseline, so the boxes match that
baseline. After the sequence is tracked, a PyTorch model scores pairs of
tracklets separated by a short gap. A unique high score reuses the earlier
id. The score threshold is selected on the validation sequence.
"""

from __future__ import annotations

import numpy as np
import torch

from tracker.constants import LEARNED_MAX_AGE, LEARNED_MIN_HITS, SORT_IOU_THRESHOLD, SORT_MAX_AGE
from tracker.geometry import xywh_to_xyxy
from tracker.model import AssociationNet
from tracker.relink import relink_rows
from tracker.sort import track_sort_frames


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
    del frames_conf  # SORT does not use detection confidence. The linker uses box motion.
    sort_rows = track_sort_frames(
        frames_xyxy,
        iou_threshold=SORT_IOU_THRESHOLD,
        max_age=SORT_MAX_AGE,
        min_hits=min_hits,
    )
    return relink_rows(sort_rows, model, mean, std, threshold, device, max_age=max_age)


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
