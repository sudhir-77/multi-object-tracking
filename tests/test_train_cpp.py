import numpy as np
import torch

from tracker.cpp_build import ensure_extension
from tracker.failure import classify_switches
from tracker.serialize import format_mot
from tracker.sort import track_sort_array
from tracker.train import train_association


def test_training_on_a_tiny_sequence_returns_a_model():
    dets = []
    gt = []
    for frame in range(1, 9):
        x = 10.0 + frame * 3.0
        dets.append([frame, x, 20.0, 18.0, 40.0, 0.95])
        dets.append([frame, 500.0, 20.0, 18.0, 40.0, 0.4])
        gt.append([frame, 1, x, 20.0, 18.0, 40.0, 1, 1])
        gt.append([frame, 2, 500.0, 20.0, 18.0, 40.0, 1, 1])
    sequence = {
        "name": "toy",
        "seq_length": 8,
        "dets": np.asarray(dets, dtype=np.float64),
        "gt": np.asarray(gt, dtype=np.float64),
    }
    torch.manual_seed(0)
    model, mean, std, losses = train_association([sequence], epochs=2, cap_per_class=32, batch_size=8)
    assert len(losses) == 2
    assert mean.shape == (10,)
    assert std.shape == (10,)
    assert model.net[0].in_features == 10


def test_failure_labels_a_crossing_switch():
    # Two pedestrians cross. The hypothesis ids swap on the second frame.
    frames = [
        {
            "gt_ids": np.array([1, 2]),
            "gt_xyxy": np.array([[0.0, 0.0, 10.0, 30.0], [12.0, 0.0, 22.0, 30.0]]),
            "pr_ids": np.array([1, 2]),
            "pr_xyxy": np.array([[0.0, 0.0, 10.0, 30.0], [12.0, 0.0, 22.0, 30.0]]),
        },
        {
            "gt_ids": np.array([1, 2]),
            "gt_xyxy": np.array([[0.0, 0.0, 10.0, 30.0], [12.0, 0.0, 22.0, 30.0]]),
            "pr_ids": np.array([2, 1]),
            "pr_xyxy": np.array([[0.0, 0.0, 10.0, 30.0], [12.0, 0.0, 22.0, 30.0]]),
        },
    ]
    report = classify_switches(frames)
    assert report["count"] == 2
    assert report["counts"]["crossing"] == 2


def test_cpp_sort_matches_python_text():
    extension = ensure_extension(force=False)
    rng = np.random.default_rng(0)
    rows = []
    for frame in range(1, 12):
        for k in range(4):
            x = 10.0 + frame * 3.0 + k * 40.0
            y = 15.0 + k * 5.0
            rows.append([frame, x, y, 20.0, 40.0])
    dets = np.asarray(rows, dtype=np.float64)
    py_out = track_sort_array(dets, 11, 0.3, 1, 3)
    cpp_out = np.asarray(extension.track_sort(dets, 11, 0.3, 1, 3))
    assert format_mot(py_out) == format_mot(cpp_out)
    # A second call with a gap still matches, including empty frames.
    gapped = dets[dets[:, 0] != 6]
    py_gap = track_sort_array(gapped, 11, 0.3, 1, 3)
    cpp_gap = np.asarray(extension.track_sort(np.ascontiguousarray(gapped), 11, 0.3, 1, 3))
    assert format_mot(py_gap) == format_mot(cpp_gap)
