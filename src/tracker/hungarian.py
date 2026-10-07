"""Rectangular Hungarian algorithm (minimization).

This is the O(n^3) potential method. Rows and columns may differ. The same
control flow is implemented in ``cpp/bindings.cpp``; tests check that the two
return the same pairs.
"""

from __future__ import annotations

import numpy as np

_INF = 1.0e18


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return a minimum-cost assignment as ``(row, col)`` pairs.

    Every row is assigned when there are at least as many columns as rows,
    and every column is assigned otherwise. Pairs that were introduced only
    by padding are not returned; callers that need a gate should put a large
    finite cost on forbidden pairs and drop them afterwards.
    """
    a = np.asarray(cost, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        return []
    n, m = a.shape
    transposed = False
    if n > m:
        a = np.ascontiguousarray(a.T)
        n, m = a.shape
        transposed = True
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = float(a[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs: list[tuple[int, int]] = []
    for j in range(1, m + 1):
        i = p[j]
        if i == 0:
            continue
        if transposed:
            pairs.append((j - 1, i - 1))
        else:
            pairs.append((i - 1, j - 1))
    pairs.sort()
    return pairs
