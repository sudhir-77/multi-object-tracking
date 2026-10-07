import numpy as np
import pytest

from tracker.metrics import hota_score, motmetrics_score, prepare_sequence, preprocess_frame


def _box_frame(gt_ids, pred_ids, xywh=(0.0, 0.0, 10.0, 10.0)):
    n_gt = len(gt_ids)
    n_pr = len(pred_ids)
    return {
        "gt_ids": np.asarray(gt_ids, dtype=np.int64),
        "gt_xyxy": np.tile(np.array(xywh, dtype=np.float64), (n_gt, 1)) if n_gt else np.zeros((0, 4)),
        "pr_ids": np.asarray(pred_ids, dtype=np.int64),
        "pr_xyxy": np.tile(
            np.array([xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]], dtype=np.float64),
            (n_pr, 1),
        )
        if n_pr
        else np.zeros((0, 4)),
    }


def test_mota_and_idf1_against_hand_counts():
    # Two perfect localizations of one pedestrian. The hypothesis id changes,
    # so there is one identity switch, no false positive, and no miss.
    # MOTA = 1 - 1/2 = 0.5. IDF1 assigns the ground-truth id to one hypothesis:
    # IDTP=1, IDFP=1, IDFN=1, IDF1 = 2/4 = 0.5.
    frames = [
        _box_frame([1], [1]),
        _box_frame([1], [2]),
    ]
    # _box_frame stored xyxy already, but the helper's gt_xyxy uses xywh by mistake
    # for gt. Rebuild explicitly.
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    frames = [
        {"gt_ids": np.array([1]), "gt_xyxy": box.copy(), "pr_ids": np.array([1]), "pr_xyxy": box.copy()},
        {"gt_ids": np.array([1]), "gt_xyxy": box.copy(), "pr_ids": np.array([2]), "pr_xyxy": box.copy()},
    ]
    scores = motmetrics_score(frames)
    assert scores["id_switches"] == 1
    assert scores["false_positives"] == 0
    assert scores["false_negatives"] == 0
    assert scores["mota"] == pytest.approx(0.5)
    assert scores["idf1"] == pytest.approx(0.5)


def test_perfect_track_scores_one():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    frames = [
        {"gt_ids": np.array([1]), "gt_xyxy": box.copy(), "pr_ids": np.array([7]), "pr_xyxy": box.copy()},
        {"gt_ids": np.array([1]), "gt_xyxy": box.copy(), "pr_ids": np.array([7]), "pr_xyxy": box.copy()},
    ]
    scores = motmetrics_score(frames)
    assert scores["mota"] == pytest.approx(1.0)
    assert scores["idf1"] == pytest.approx(1.0)
    assert scores["id_switches"] == 0
    assert hota_score(frames) == pytest.approx(1.0, abs=1e-6)


def test_distractor_prediction_is_not_a_false_positive():
    # One static person (class 7) and one pedestrian. The only prediction sits
    # on the distractor, so preprocessing removes it and the pedestrian is a miss.
    gt = np.array(
        [
            [1, 1, 0, 0, 10, 10, 1, 7],
            [1, 2, 100, 100, 10, 10, 1, 1],
        ],
        dtype=np.float64,
    )
    pred = np.array([[1, 5, 0, 0, 10, 10]], dtype=np.float64)
    frames = prepare_sequence(gt, pred, seq_length=1)
    assert len(frames[0]["pr_ids"]) == 0
    assert list(frames[0]["gt_ids"]) == [2]
    scores = motmetrics_score(frames)
    assert scores["false_positives"] == 0
    assert scores["false_negatives"] == 1
    assert scores["mota"] == pytest.approx(0.0)


def test_preprocess_keeps_pedestrian_match():
    frame = preprocess_frame(
        gt_ids=np.array([3]),
        gt_xywh=np.array([[0.0, 0.0, 10.0, 10.0]]),
        gt_conf=np.array([1.0]),
        gt_cls=np.array([1]),
        pr_ids=np.array([8]),
        pr_xywh=np.array([[0.0, 0.0, 10.0, 10.0]]),
    )
    assert list(frame["gt_ids"]) == [3]
    assert list(frame["pr_ids"]) == [8]
