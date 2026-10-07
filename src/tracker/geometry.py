"""Box geometry used by both trackers. Boxes are xyxy in pixels."""

from __future__ import annotations

import math


def xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> list[float]:
    x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float] | tuple[float, ...]) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]


def xyxy_to_z(box: list[float] | tuple[float, ...]) -> list[float]:
    """SORT measurement: center x, center y, area, aspect ratio w/h."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    s = w * h
    r = 0.0 if h == 0.0 else w / h
    return [cx, cy, s, r]


def z_to_xyxy(state: list[float] | tuple[float, ...]) -> list[float]:
    """Invert :func:`xyxy_to_z` for a 7-vector or a 4-vector ``(cx, cy, s, r)``."""
    cx = float(state[0])
    cy = float(state[1])
    s = float(state[2])
    r = float(state[3])
    sr = s * r
    if sr <= 0.0:
        w = 0.0
    else:
        w = math.sqrt(sr)
    h = 0.0 if w == 0.0 else s / w
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def box_iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1 = ax1 if ax1 > bx1 else bx1
    iy1 = ay1 if ay1 > by1 else by1
    ix2 = ax2 if ax2 < bx2 else bx2
    iy2 = ay2 if ay2 < by2 else by2
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def center(box: list[float] | tuple[float, ...]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def height(box: list[float] | tuple[float, ...]) -> float:
    return float(box[3]) - float(box[1])


def width(box: list[float] | tuple[float, ...]) -> float:
    return float(box[2]) - float(box[0])
