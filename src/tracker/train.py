"""Train the association model on training sequences only."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from tracker.constants import FEATURE_NAMES, SEED
from tracker.features import apply_normalizer, feature_matrix, fit_normalizer
from tracker.geometry import box_iou
from tracker.metrics import prepare_sequence
from tracker.model import AssociationNet
from tracker.relink import junction_features, motion_gate, tracklets_from_rows
from tracker.sort import track_sort_array


def _majority_identity(frames: list[dict[str, np.ndarray]]) -> dict[int, int | None]:
    """Map a SORT id to the pedestrian it overlaps most often. Ambiguous ids are dropped."""
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    lengths: Counter[int] = Counter()
    for frame in frames:
        taken: set[int] = set()
        for j, pid in enumerate(frame["pr_ids"]):
            best_iou = 0.5
            best_gid = None
            for i, gid in enumerate(frame["gt_ids"]):
                gid_i = int(gid)
                if gid_i in taken:
                    continue
                iou = box_iou(frame["pr_xyxy"][j], frame["gt_xyxy"][i])
                if iou >= best_iou:
                    best_iou = iou
                    best_gid = gid_i
            lengths[int(pid)] += 1
            if best_gid is not None:
                taken.add(best_gid)
                votes[int(pid)][best_gid] += 1
    assigned: dict[int, int | None] = {}
    for pid, length in lengths.items():
        if not votes[pid]:
            assigned[pid] = None
            continue
        gid, count = votes[pid].most_common(1)[0]
        unanimous = count == length and count >= 2
        established = count >= 3 and count >= 0.5 * length
        if unanimous or established:
            assigned[pid] = int(gid)
        else:
            assigned[pid] = None
    return assigned


def pairs_from_sequence(sequence: dict, rng: np.random.RandomState, negatives: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Tracklet junctions labeled by ground-truth identity. ``rng`` and ``negatives`` keep the call stable."""
    del negatives
    seq_length = int(sequence["seq_length"])
    dets = np.asarray(sequence["dets"], dtype=np.float64)
    rows = track_sort_array(dets[:, :5] if dets.size else dets, seq_length)
    if rows.size == 0:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    identities = _majority_identity(prepare_sequence(sequence["gt"], rows, seq_length))
    tracklets = tracklets_from_rows(rows)
    features: list[list[float]] = []
    labels: list[float] = []
    for earlier in tracklets:
        gid = identities.get(earlier.id)
        for later in tracklets:
            if earlier.id == later.id:
                continue
            feats = junction_features(earlier, later)
            if feats is None or not motion_gate(feats):
                continue
            other = identities.get(later.id)
            if gid is None or other is None:
                continue
            features.append(feats)
            labels.append(1.0 if gid == other else 0.0)
    del rng
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
