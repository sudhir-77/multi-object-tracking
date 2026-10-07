"""Classify identity switches that remain after learned association."""

from __future__ import annotations

import math

import numpy as np

from tracker.geometry import center, height


def classify_switches(frames: list[dict[str, np.ndarray]]) -> dict[str, float | int | dict[str, int]]:
    """Walk CLEAR matches in time and label each identity switch.

    A switch is a ground-truth id matched at IoU >= 0.5 to a different
    hypothesis than the previous time that ground-truth id was matched.
    """
    from tracker.metrics import _distance
    from tracker.hungarian import hungarian

    last_hyp: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    counts = {"occlusion": 0, "crossing": 0, "short_gap": 0, "other": 0}
    details: list[dict[str, float | int | str]] = []
    for t, frame in enumerate(frames):
        gt_ids = [int(v) for v in frame["gt_ids"]]
        pr_ids = [int(v) for v in frame["pr_ids"]]
        dist = _distance(frame["gt_xyxy"], frame["pr_xyxy"])
        matched: dict[int, int] = {}
        if gt_ids and pr_ids:
            cost = np.array(dist, copy=True)
            finite = np.isfinite(cost)
            cost = np.where(finite, cost, 1.0e6)
            for i, j in hungarian(cost):
                if not finite[i, j]:
                    continue
                matched[gt_ids[i]] = pr_ids[j]
        for gid, hid in matched.items():
            if gid in last_hyp and last_hyp[gid] != hid:
                gap = (t + 1) - last_frame[gid]
                box = frame["gt_xyxy"][gt_ids.index(gid)]
                c = center(box)
                h = height(box)
                crossing = False
                for other_id, other_box in zip(gt_ids, frame["gt_xyxy"]):
                    if other_id == gid:
                        continue
                    oc = center(other_box)
                    if math.hypot(c[0] - oc[0], c[1] - oc[1]) <= 1.5 * max(h, 1.0):
                        crossing = True
                        break
                if crossing:
                    kind = "crossing"
                elif gap >= 5:
                    kind = "occlusion"
                elif gap > 1:
                    kind = "short_gap"
                else:
                    kind = "other"
                counts[kind] += 1
                details.append({"frame": t + 1, "gt_id": gid, "gap": gap, "kind": kind})
            last_hyp[gid] = hid
            last_frame[gid] = t + 1
    total = sum(counts.values())
    shares = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {"count": total, "counts": counts, "shares": shares, "examples": details[:12]}
