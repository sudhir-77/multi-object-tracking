"""MOT challenge text format."""

from __future__ import annotations

import numpy as np


def format_mot(rows: np.ndarray | list[tuple[float, ...]]) -> str:
    """``frame,id,x,y,w,h,conf,-1,-1,-1`` sorted by frame then id."""
    array = np.asarray(rows, dtype=np.float64)
    if array.size == 0:
        return ""
    if array.ndim == 1:
        array = array.reshape(1, -1)
    order = np.lexsort((array[:, 1], array[:, 0]))
    array = array[order]
    lines = []
    for row in array:
        lines.append(
            f"{int(row[0])},{int(row[1])},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},1,-1,-1,-1"
        )
    return "\n".join(lines) + "\n"
