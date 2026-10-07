"""MOTA, IDF1, and ID switches via motmetrics. HOTA via TrackEval.

Preprocessing follows TrackEval's MOT17 config: distractor classes
(person on vehicle, static person, distractor, reflection) absorb matched
predictions, and only pedestrian boxes with a non-zero flag are scored.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.constants import DISTRACTOR_CLASSES, PEDESTRIAN_CLASS
from tracker.geometry import box_iou, xywh_to_xyxy

_EPS = float(np.finfo(float).eps)


def _patch_numpy_aliases() -> None:
    # TrackEval 1.x still refers to aliases removed in NumPy 2.
    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = np.int64  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]


def _iou_matrix(gt_xyxy: np.ndarray, pr_xyxy: np.ndarray) -> np.ndarray:
    n = len(gt_xyxy)
    m = len(pr_xyxy)
    out = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            out[i, j] = box_iou(gt_xyxy[i], pr_xyxy[j])
    return out


def preprocess_frame(
    gt_ids: np.ndarray,
    gt_xywh: np.ndarray,
    gt_conf: np.ndarray,
    gt_cls: np.ndarray,
    pr_ids: np.ndarray,
    pr_xywh: np.ndarray,
) -> dict[str, np.ndarray]:
    """One frame of MOT17 preprocessing. Boxes are xywh."""
    gt_xyxy = np.array([xywh_to_xyxy(b) for b in gt_xywh], dtype=np.float64) if len(gt_xywh) else np.zeros((0, 4))
    pr_xyxy = np.array([xywh_to_xyxy(b) for b in pr_xywh], dtype=np.float64) if len(pr_xywh) else np.zeros((0, 4))
    remove = np.zeros(len(pr_ids), dtype=bool)
    if len(gt_ids) and len(pr_ids):
        scores = _iou_matrix(gt_xyxy, pr_xyxy)
        gated = scores.copy()
        gated[gated < 0.5 - _EPS] = 0.0
        rows, cols = linear_sum_assignment(-gated)
        keep = gated[rows, cols] > _EPS
        rows = rows[keep]
        cols = cols[keep]
        for r, c in zip(rows, cols):
            if int(gt_cls[r]) in DISTRACTOR_CLASSES:
                remove[c] = True
    keep_pr = ~remove
    pr_ids = pr_ids[keep_pr]
    pr_xyxy = pr_xyxy[keep_pr]
    if len(gt_ids):
        keep_gt = (gt_conf != 0) & (gt_cls.astype(np.int64) == PEDESTRIAN_CLASS)
    else:
        keep_gt = np.zeros(0, dtype=bool)
    gt_ids = gt_ids[keep_gt]
    gt_xyxy = gt_xyxy[keep_gt]
    return {
        "gt_ids": gt_ids.astype(np.int64),
        "gt_xyxy": gt_xyxy,
        "pr_ids": pr_ids.astype(np.int64),
        "pr_xyxy": pr_xyxy,
    }


def prepare_sequence(gt: np.ndarray, predictions: np.ndarray, seq_length: int) -> list[dict[str, np.ndarray]]:
    """``gt`` is frame,id,x,y,w,h,conf,class. ``predictions`` is frame,id,x,y,w,h."""
    gt_by = [[] for _ in range(seq_length)]
    pr_by = [[] for _ in range(seq_length)]
    for row in gt:
        frame = int(row[0])
        if 1 <= frame <= seq_length:
            gt_by[frame - 1].append(row)
    for row in np.asarray(predictions, dtype=np.float64).reshape(-1, 6) if np.size(predictions) else []:
        frame = int(row[0])
        if 1 <= frame <= seq_length:
            pr_by[frame - 1].append(row)
    frames = []
    for t in range(seq_length):
        g = np.asarray(gt_by[t], dtype=np.float64) if gt_by[t] else np.zeros((0, 8))
        p = np.asarray(pr_by[t], dtype=np.float64) if pr_by[t] else np.zeros((0, 6))
        if len(g):
            # Drop duplicate gt ids in a frame, keeping the first row.
            _, uniq = np.unique(g[:, 1].astype(np.int64), return_index=True)
            g = g[np.sort(uniq)]
        if len(p):
            _, uniq = np.unique(p[:, 1].astype(np.int64), return_index=True)
            p = p[np.sort(uniq)]
        frames.append(
            preprocess_frame(
                g[:, 1].astype(np.int64) if len(g) else np.zeros(0, dtype=np.int64),
                g[:, 2:6] if len(g) else np.zeros((0, 4)),
                g[:, 6] if len(g) else np.zeros(0),
                g[:, 7].astype(np.int64) if len(g) else np.zeros(0, dtype=np.int64),
                p[:, 1].astype(np.int64) if len(p) else np.zeros(0, dtype=np.int64),
                p[:, 2:6] if len(p) else np.zeros((0, 4)),
            )
        )
    return frames


def _distance(gt_xyxy: np.ndarray, pr_xyxy: np.ndarray) -> np.ndarray:
    n, m = len(gt_xyxy), len(pr_xyxy)
    dist = np.full((n, m), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou = box_iou(gt_xyxy[i], pr_xyxy[j])
            if iou >= 0.5:
                dist[i, j] = 1.0 - iou
    return dist


def motmetrics_score(frames: list[dict[str, np.ndarray]], id_offset: int = 0) -> dict[str, float]:
    """CLEAR MOT and IDF1 for one already-preprocessed sequence."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for t, frame in enumerate(frames):
        gt_ids = [int(v) + id_offset for v in frame["gt_ids"]]
        pr_ids = [int(v) + id_offset for v in frame["pr_ids"]]
        dist = _distance(frame["gt_xyxy"], frame["pr_xyxy"])
        acc.update(gt_ids, pr_ids, dist)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "num_objects",
            "num_predictions",
            "num_matches",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mota",
            "idf1",
            "precision",
            "recall",
        ],
        name="seq",
    )
    row = summary.loc["seq"]

    def _get(name: str) -> float:
        value = row[name]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        return float(value)

    return {
        "num_frames": _get("num_frames"),
        "num_objects": _get("num_objects"),
        "num_predictions": _get("num_predictions"),
        "num_matches": _get("num_matches"),
        "id_switches": _get("num_switches"),
        "false_positives": _get("num_false_positives"),
        "false_negatives": _get("num_misses"),
        "mota": _get("mota"),
        "idf1": _get("idf1"),
        "precision": _get("precision"),
        "recall": _get("recall"),
        "accumulator": acc,
    }


def hota_score(frames: list[dict[str, np.ndarray]]) -> float:
    """Mean HOTA over localization thresholds, using TrackEval's implementation."""
    _patch_numpy_aliases()
    from trackeval.metrics.hota import HOTA

    gt_ids_all: list[int] = []
    pr_ids_all: list[int] = []
    for frame in frames:
        gt_ids_all.extend(int(v) for v in frame["gt_ids"])
        pr_ids_all.extend(int(v) for v in frame["pr_ids"])
    gt_uniq = sorted(set(gt_ids_all))
    pr_uniq = sorted(set(pr_ids_all))
    gt_map = {gid: i for i, gid in enumerate(gt_uniq)}
    pr_map = {pid: i for i, pid in enumerate(pr_uniq)}
    data = {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_uniq),
        "num_tracker_ids": len(pr_uniq),
        "num_gt_dets": int(sum(len(f["gt_ids"]) for f in frames)),
        "num_tracker_dets": int(sum(len(f["pr_ids"]) for f in frames)),
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }
    for frame in frames:
        gt_ids = np.asarray([gt_map[int(v)] for v in frame["gt_ids"]], dtype=int)
        pr_ids = np.asarray([pr_map[int(v)] for v in frame["pr_ids"]], dtype=int)
        sim = _iou_matrix(frame["gt_xyxy"], frame["pr_xyxy"])
        # TrackEval's association accumulation divides by a broadcast of row and
        # column sums. Give empty axes a positive length of zero that still
        # matches the id vectors (both length 0). Non-empty empty-axis frames
        # use a (n, 0) or (0, m) matrix, which the installed HOTA accepts.
        data["gt_ids"].append(gt_ids)
        data["tracker_ids"].append(pr_ids)
        data["similarity_scores"].append(sim)
    metric = HOTA()
    result = metric.eval_sequence(data)
    return float(np.mean(result["HOTA"]))


def combine_hota(sequence_frames: list[list[dict[str, np.ndarray]]]) -> float:
    """HOTA combined the way TrackEval combines sequences (sum of counts)."""
    if len(sequence_frames) == 1:
        return hota_score(sequence_frames[0])
    _patch_numpy_aliases()
    from trackeval.metrics.hota import HOTA

    metric = HOTA()
    results = {}
    for i, frames in enumerate(sequence_frames):
        # Rebuild through hota_score's data construction by evaluating each
        # sequence, then combine. eval_sequence is called inside a local copy.
        gt_ids_all: list[int] = []
        pr_ids_all: list[int] = []
        for frame in frames:
            gt_ids_all.extend(int(v) for v in frame["gt_ids"])
            pr_ids_all.extend(int(v) for v in frame["pr_ids"])
        gt_uniq = sorted(set(gt_ids_all))
        pr_uniq = sorted(set(pr_ids_all))
        gt_map = {gid: j for j, gid in enumerate(gt_uniq)}
        pr_map = {pid: j for j, pid in enumerate(pr_uniq)}
        data = {
            "num_timesteps": len(frames),
            "num_gt_ids": len(gt_uniq),
            "num_tracker_ids": len(pr_uniq),
            "num_gt_dets": int(sum(len(f["gt_ids"]) for f in frames)),
            "num_tracker_dets": int(sum(len(f["pr_ids"]) for f in frames)),
            "gt_ids": [],
            "tracker_ids": [],
            "similarity_scores": [],
        }
        for frame in frames:
            data["gt_ids"].append(np.asarray([gt_map[int(v)] for v in frame["gt_ids"]], dtype=int))
            data["tracker_ids"].append(np.asarray([pr_map[int(v)] for v in frame["pr_ids"]], dtype=int))
            data["similarity_scores"].append(_iou_matrix(frame["gt_xyxy"], frame["pr_xyxy"]))
        results[str(i)] = metric.eval_sequence(data)
    combined = metric.combine_sequences(results)
    return float(np.mean(combined["HOTA"]))


def score_sequences(prepared: list[list[dict[str, np.ndarray]]]) -> dict[str, float]:
    """Aggregate MOTChallenge metrics across sequences with disjoint ids."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for seq_i, frames in enumerate(prepared):
        offset = seq_i * 1_000_000
        for frame in frames:
            gt_ids = [int(v) + offset for v in frame["gt_ids"]]
            pr_ids = [int(v) + offset for v in frame["pr_ids"]]
            dist = _distance(frame["gt_xyxy"], frame["pr_xyxy"])
            acc.update(gt_ids, pr_ids, dist)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "num_objects",
            "num_predictions",
            "num_matches",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mota",
            "idf1",
            "precision",
            "recall",
        ],
        name="overall",
    )
    row = summary.loc["overall"]

    def _get(name: str) -> float:
        value = row[name]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        return float(value)

    hota = combine_hota(prepared)
    return {
        "num_frames": _get("num_frames"),
        "num_objects": _get("num_objects"),
        "num_predictions": _get("num_predictions"),
        "num_matches": _get("num_matches"),
        "id_switches": _get("num_switches"),
        "false_positives": _get("num_false_positives"),
        "false_negatives": _get("num_misses"),
        "mota": _get("mota"),
        "idf1": _get("idf1"),
        "precision": _get("precision"),
        "recall": _get("recall"),
        "hota": hota,
    }
