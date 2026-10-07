"""Train the association model on training sequences only."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tracker.constants import FEATURE_NAMES, LEARNED_MAX_AGE, PEDESTRIAN_CLASS, SEED
from tracker.features import apply_normalizer, feature_matrix, fit_normalizer, pair_features
from tracker.geometry import box_iou, center, xywh_to_xyxy
from tracker.hungarian import hungarian
from tracker.kalman import KalmanBox
from tracker.model import AssociationNet


class _SimTrack:
    def __init__(self, xyxy: list[float]) -> None:
        self.kf = KalmanBox(xyxy)
        self.last_xyxy = [float(v) for v in xyxy]
        self.time_since_update = 0
        cx, cy = center(xyxy)
        self.prev_center = (cx, cy)
        self.obs_vel = [0.0, 0.0]

    def predict(self) -> None:
        self.kf.predict()
        self.time_since_update += 1

    def update(self, xyxy: list[float]) -> None:
        gap = self.time_since_update if self.time_since_update > 0 else 1
        cx, cy = center(xyxy)
        self.obs_vel = [(cx - self.prev_center[0]) / gap, (cy - self.prev_center[1]) / gap]
        self.prev_center = (cx, cy)
        self.last_xyxy = [float(v) for v in xyxy]
        self.time_since_update = 0
        self.kf.update(xyxy)


def _match_dets_to_gt(
    det_xyxy: list[list[float]], gt_xyxy: list[list[float]], gt_ids: list[int]
) -> dict[int, int]:
    """Map gt id -> detection index for pairs with IoU >= 0.5."""
    if not det_xyxy or not gt_xyxy:
        return {}
    cost = np.zeros((len(gt_xyxy), len(det_xyxy)), dtype=np.float64)
    for i, gt_box in enumerate(gt_xyxy):
        for j, det_box in enumerate(det_xyxy):
            iou = box_iou(gt_box, det_box)
            cost[i, j] = 1.0 - iou if iou >= 0.5 else 1.0e6
    assigned: dict[int, int] = {}
    for i, j in hungarian(cost):
        if cost[i, j] >= 1.0e5:
            continue
        assigned[gt_ids[i]] = j
    return assigned


def pairs_from_sequence(sequence: dict, rng: np.random.RandomState, negatives: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Positive and negative pairs from pedestrian-aligned detections."""
    seq_length = int(sequence["seq_length"])
    dets = sequence["dets"]
    gt = sequence["gt"]
    det_frames: list[list[list[float]]] = [[] for _ in range(seq_length)]
    det_conf: list[list[float]] = [[] for _ in range(seq_length)]
    for row in dets:
        frame = int(row[0])
        if 1 <= frame <= seq_length:
            det_frames[frame - 1].append(xywh_to_xyxy(row[1:5]))
            det_conf[frame - 1].append(float(row[5]))
    gt_frames_box: list[list[list[float]]] = [[] for _ in range(seq_length)]
    gt_frames_id: list[list[int]] = [[] for _ in range(seq_length)]
    for row in gt:
        frame = int(row[0])
        if not (1 <= frame <= seq_length):
            continue
        if int(row[7]) != PEDESTRIAN_CLASS or float(row[6]) == 0.0:
            continue
        gt_frames_box[frame - 1].append(xywh_to_xyxy(row[2:6]))
        gt_frames_id[frame - 1].append(int(row[1]))

    sims: dict[int, _SimTrack] = {}
    features: list[list[float]] = []
    labels: list[float] = []
    for t in range(seq_length):
        for sim in sims.values():
            sim.predict()
        assigned = _match_dets_to_gt(det_frames[t], gt_frames_box[t], gt_frames_id[t])
        seen: set[int] = set()
        for gt_id, det_index in assigned.items():
            seen.add(gt_id)
            det_box = det_frames[t][det_index]
            conf = det_conf[t][det_index]
            sim = sims.get(gt_id)
            if sim is None:
                sims[gt_id] = _SimTrack(det_box)
                continue
            pos = pair_features(sim.kf.xyxy(), sim.last_xyxy, sim.obs_vel, sim.time_since_update, det_box, conf)
            features.append(pos)
            labels.append(1.0)
            others = [k for k in range(len(det_frames[t])) if k != det_index]
            if others:
                last_c = center(sim.last_xyxy)
                ranked = sorted(
                    others,
                    key=lambda k: (center(det_frames[t][k])[0] - last_c[0]) ** 2
                    + (center(det_frames[t][k])[1] - last_c[1]) ** 2,
                )
                chosen = ranked[:1]
                if len(ranked) > 1 and negatives > 1:
                    rest = ranked[1:]
                    pick = int(rng.randint(0, len(rest)))
                    chosen.append(rest[pick])
                for k in chosen:
                    neg = pair_features(
                        sim.kf.xyxy(),
                        sim.last_xyxy,
                        sim.obs_vel,
                        sim.time_since_update,
                        det_frames[t][k],
                        det_conf[t][k],
                    )
                    features.append(neg)
                    labels.append(0.0)
            sim.update(det_box)
        drop = [gid for gid, sim in sims.items() if gid not in seen and sim.time_since_update > LEARNED_MAX_AGE]
        for gid in drop:
            del sims[gid]
    if not features:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return feature_matrix(features), np.asarray(labels, dtype=np.float64)


def _balance(features: np.ndarray, labels: np.ndarray, rng: np.random.RandomState, cap: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(labels == 1.0)
    neg = np.flatnonzero(labels == 0.0)
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("association training produced only one class of pairs")
    n = min(len(pos), len(neg), cap)
    take_pos = rng.choice(pos, size=n, replace=False)
    take_neg = rng.choice(neg, size=n, replace=False)
    idx = np.concatenate([take_pos, take_neg])
    rng.shuffle(idx)
    return features[idx], labels[idx]


def train_association(
    sequences: list[dict],
    *,
    epochs: int = 8,
    batch_size: int = 256,
    cap_per_class: int = 20000,
    lr: float = 1e-3,
    seed: int = SEED,
    device: torch.device | None = None,
) -> tuple[AssociationNet, np.ndarray, np.ndarray, list[float]]:
    if device is None:
        device = torch.device("cpu")
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    for sequence in sequences:
        x, y = pairs_from_sequence(sequence, rng)
        if len(y):
            chunks_x.append(x)
            chunks_y.append(y)
    if not chunks_x:
        raise RuntimeError("no association pairs were built from the training sequences")
    features = np.concatenate(chunks_x, axis=0)
    labels = np.concatenate(chunks_y, axis=0)
    features, labels = _balance(features, labels, rng, cap_per_class)
    mean, std = fit_normalizer(features)
    normed = apply_normalizer(features, mean, std).astype(np.float32)
    model = AssociationNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    n = len(labels)
    for _epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            xb = torch.as_tensor(normed[batch], dtype=torch.float32, device=device)
            yb = torch.as_tensor(labels[batch], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
        losses.append(total / max(seen, 1))
    model.eval()
    return model, mean, std, losses
