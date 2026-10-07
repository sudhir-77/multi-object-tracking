import math

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from tracker.constants import HELDOUT_SEQUENCES, TRAIN_SEQUENCES, VAL_SEQUENCES, split_dict
from tracker.features import pair_features
from tracker.geometry import box_iou, xywh_to_xyxy, xyxy_to_xywh
from tracker.hungarian import hungarian
from tracker.kalman import invert_4x4, predict_state
from tracker.constants import Q_DIAG
from tracker.learned import track_learned_frames
from tracker.model import AssociationNet
from tracker.serialize import format_mot
from tracker.sort import track_sort_frames


def test_iou_hand_value():
    # 10x10 boxes overlapping on a 5x10 region: inter=50, union=150, IoU=1/3.
    a = [0.0, 0.0, 10.0, 10.0]
    b = [5.0, 0.0, 15.0, 10.0]
    assert box_iou(a, b) == pytest.approx(1.0 / 3.0)
    assert box_iou(a, [20.0, 20.0, 30.0, 30.0]) == 0.0


def test_xywh_roundtrip():
    box = [1.5, 2.5, 10.0, 4.0]
    assert xyxy_to_xywh(xywh_to_xyxy(box)) == pytest.approx(box)


def test_invert_matches_numpy():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 4))
    matrix = base @ base.T + np.eye(4)
    inv = np.asarray(invert_4x4(matrix.tolist()))
    assert inv @ matrix == pytest.approx(np.eye(4), abs=1e-8)


def test_predict_matches_transition_matrix():
    x = [10.0, 20.0, 100.0, 1.0, 1.5, -0.5, 0.25]
    P = np.eye(7).tolist()
    predict_state(x, P)
    state = np.array([10.0, 20.0, 100.0, 1.0, 1.5, -0.5, 0.25])
    cov = np.eye(7)
    transition = np.eye(7)
    transition[0, 4] = transition[1, 5] = transition[2, 6] = 1.0
    expected_x = transition @ state
    expected_p = transition @ cov @ transition.T + np.diag(Q_DIAG)
    assert x == pytest.approx(expected_x)
    assert np.asarray(P) == pytest.approx(expected_p, abs=1e-10)


def test_hungarian_optimal_cost_matches_scipy():
    rng = np.random.default_rng(1)
    for shape in ((1, 1), (3, 3), (2, 5), (6, 3), (8, 8)):
        cost = rng.normal(size=shape)
        pairs = hungarian(cost)
        rows, cols = linear_sum_assignment(cost)
        assert len(pairs) == len(rows)
        ours = sum(cost[i, j] for i, j in pairs)
        theirs = cost[rows, cols].sum()
        assert ours == pytest.approx(theirs)


def test_split_has_no_overlap_and_two_heldout_sequences():
    split = split_dict()
    assert set(split) == {"train", "val", "heldout"}
    assert len(split["heldout"]) == 2
    assert set(split["train"]) == set(TRAIN_SEQUENCES)
    groups = [set(split["train"]), set(split["val"]), set(split["heldout"])]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert len(groups[0] | groups[1] | groups[2]) == 7
    assert set(VAL_SEQUENCES).isdisjoint(HELDOUT_SEQUENCES)


def test_pair_features_for_identical_boxes():
    box = [0.0, 0.0, 10.0, 10.0]
    feats = pair_features(box, box, (0.0, 0.0), 1, box, 0.8)
    assert feats[0] == pytest.approx(1.0)
    assert feats[1] == pytest.approx(1.0)
    assert feats[2] == pytest.approx(0.0)
    assert feats[7] == pytest.approx(1.0 / 30.0)
    assert feats[9] == pytest.approx(0.8)
    assert all(math.isfinite(v) for v in feats)


def test_sort_breaks_identity_across_a_long_gap_and_learned_keeps_it():
    frames = []
    for t in range(8):
        x = 10.0 + t * 2.0
        frames.append([[x, 10.0, x + 40.0, 50.0]])
    for _ in range(4):
        frames.append([])
    for t in range(8, 14):
        x = 10.0 + t * 2.0
        frames.append([[x, 10.0, x + 40.0, 50.0]])
    sort_rows = track_sort_frames(frames, iou_threshold=0.3, max_age=1, min_hits=1)
    before = [row[1] for row in sort_rows if row[0] <= 8]
    after = [row[1] for row in sort_rows if row[0] >= 13]
    assert before and after
    assert before[-1] != after[0]

    class Close(torch.nn.Module):
        def forward(self, x):
            return 8.0 * x[:, 0] - 2.0

    confs = [[1.0] if frame else [] for frame in frames]
    learned_rows = track_learned_frames(
        frames,
        confs,
        Close(),
        np.zeros(10),
        np.ones(10),
        threshold=0.2,
        max_age=30,
        min_hits=1,
    )
    learned_before = [row[1] for row in learned_rows if row[0] <= 8]
    learned_after = [row[1] for row in learned_rows if row[0] >= 13]
    assert learned_before and learned_after
    assert learned_before[-1] == learned_after[0]


def test_model_output_shape():
    net = AssociationNet()
    out = net(torch.zeros(4, 10))
    assert tuple(out.shape) == (4,)


def test_format_mot_sorts_and_uses_mot_columns():
    text = format_mot([(2, 3, 1.0, 2.0, 3.0, 4.0), (1, 1, 0.0, 0.0, 5.0, 6.0)])
    lines = text.strip().splitlines()
    assert lines[0].startswith("1,1,0.000000,0.000000,5.000000,6.000000,1,-1,-1,-1")
    assert lines[1].startswith("2,3,")
