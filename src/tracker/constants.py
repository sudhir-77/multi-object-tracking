"""Fixed experiment constants.

The sequence split is part of the project definition. It was chosen from MOT17
scene order before any model was trained, and it is not revised from scores.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Public detector used for every scene. The three MOT17 detectors share a
# scene, so mixing them across splits would leak the same video.
DETECTOR = "FRCNN"

TRAIN_SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
]
VAL_SEQUENCES = ["MOT17-10-FRCNN"]
HELDOUT_SEQUENCES = ["MOT17-11-FRCNN", "MOT17-13-FRCNN"]

SEED = 0

# SORT (Bewley et al.): one missed frame of grace, confirm after 3 hits, IoU gate 0.3.
SORT_MAX_AGE = 1
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# Learned tracker. max_age is a fixed design choice (about one second at 30 fps),
# not selected on the held-out sequences. The match threshold is selected on val.
LEARNED_MAX_AGE = 30
LEARNED_MIN_HITS = 3
# Maximum center-distance, in units of the last box height, for a tracklet merge.
# The value used at test time is chosen on the validation sequence.
MATCH_THRESHOLDS = (0.40, 0.60, 0.80, 1.00)

# MOT17 distractors removed from the score, matching TrackEval's MOT17 config.
DISTRACTOR_CLASSES = frozenset({2, 7, 8, 12})
PEDESTRIAN_CLASS = 1

MOT17_LABELS_URL = "https://motchallenge.net/data/MOT17Labels.zip"
MOT17_LABELS_SHA256 = "0aa79322e91583369f42f17c4d79a0b145380d8732487bba59272048dc82b2b9"
# Class-1 boxes in train/MOT17-02-FRCNN/gt/gt.txt. Checked after extract.
MOT17_02_PEDESTRIAN_ROWS = 18581

FEATURE_NAMES = (
    "iou_pred",
    "iou_last",
    "dist_pred_norm",
    "dist_last_norm",
    "log_h_ratio",
    "log_w_ratio",
    "aspect_abs_diff",
    "gap_norm",
    "vel_error_norm",
    "det_conf",
)

Q_DIAG = (1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001)
R_DIAG = (1.0, 1.0, 10.0, 10.0)
P0_DIAG = (10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0)

# Costs at or above this value are treated as "do not match".
FORBID_COST = 1.0e6


def split_dict() -> dict[str, list[str]]:
    return {
        "train": list(TRAIN_SEQUENCES),
        "val": list(VAL_SEQUENCES),
        "heldout": list(HELDOUT_SEQUENCES),
    }
