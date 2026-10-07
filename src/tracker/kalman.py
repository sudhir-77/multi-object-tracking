"""Constant-velocity Kalman filter on SORT's 7D box state.

State is ``[cx, cy, s, r, vx, vy, vs]`` where ``s`` is box area and ``r`` is
aspect ratio. Process and measurement noise match the public SORT
implementation: ``Q[6,6]`` is scaled twice, so the diagonal entry is ``1e-4``.

The predict and update loops are written out elementwise so the C++ core can
follow the same operation order.
"""

from __future__ import annotations

import math

from tracker.constants import P0_DIAG, Q_DIAG, R_DIAG
from tracker.geometry import xyxy_to_z, z_to_xyxy


def _copy_matrix(P: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in P]


def invert_4x4(A: list[list[float]]) -> list[list[float]]:
    """Invert a 4x4 matrix with Gauss-Jordan elimination and partial pivoting."""
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for col in range(4):
        piv = col
        best = abs(M[col][col])
        for r in range(col + 1, 4):
            value = abs(M[r][col])
            if value > best:
                best = value
                piv = r
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            M[col][col] += 1e-6
        div = M[col][col]
        for j in range(8):
            M[col][j] /= div
        for r in range(4):
            if r == col:
                continue
            factor = M[r][col]
            for j in range(8):
                M[r][j] -= factor * M[col][j]
    return [row[4:] for row in M]


def predict_state(x: list[float], P: list[list[float]]) -> None:
    """In-place constant-velocity predict with ``dt = 1``."""
    if x[6] + x[2] <= 0.0:
        x[6] = 0.0
    x[0] = x[0] + x[4]
    x[1] = x[1] + x[5]
    x[2] = x[2] + x[6]
    M = _copy_matrix(P)
    for j in range(7):
        M[0][j] = P[0][j] + P[4][j]
        M[1][j] = P[1][j] + P[5][j]
        M[2][j] = P[2][j] + P[6][j]
    for i in range(7):
        P[i][0] = M[i][0] + M[i][4]
        P[i][1] = M[i][1] + M[i][5]
        P[i][2] = M[i][2] + M[i][6]
        P[i][3] = M[i][3]
        P[i][4] = M[i][4]
        P[i][5] = M[i][5]
        P[i][6] = M[i][6]
        P[i][0] += Q_DIAG[0] if i == 0 else 0.0
        P[i][1] += Q_DIAG[1] if i == 1 else 0.0
        P[i][2] += Q_DIAG[2] if i == 2 else 0.0
        P[i][3] += Q_DIAG[3] if i == 3 else 0.0
        P[i][4] += Q_DIAG[4] if i == 4 else 0.0
        P[i][5] += Q_DIAG[5] if i == 5 else 0.0
        P[i][6] += Q_DIAG[6] if i == 6 else 0.0


def update_state(x: list[float], P: list[list[float]], z: list[float]) -> None:
    """In-place Kalman update. ``z`` is ``[cx, cy, s, r]``."""
    y = [z[i] - x[i] for i in range(4)]
    S = [[P[i][j] for j in range(4)] for i in range(4)]
    S[0][0] += R_DIAG[0]
    S[1][1] += R_DIAG[1]
    S[2][2] += R_DIAG[2]
    S[3][3] += R_DIAG[3]
    Sinv = invert_4x4(S)
    K = [[0.0] * 4 for _ in range(7)]
    for i in range(7):
        for j in range(4):
            acc = 0.0
            for k in range(4):
                acc += P[i][k] * Sinv[k][j]
            K[i][j] = acc
    for i in range(7):
        acc = 0.0
        for j in range(4):
            acc += K[i][j] * y[j]
        x[i] = x[i] + acc
    Pnew = [[0.0] * 7 for _ in range(7)]
    for i in range(7):
        for j in range(7):
            acc = P[i][j]
            for k in range(4):
                acc -= K[i][k] * P[k][j]
            Pnew[i][j] = acc
    for i in range(7):
        for j in range(7):
            P[i][j] = Pnew[i][j]


class KalmanBox:
    """One object's filter, initialised from an xyxy detection."""

    def __init__(self, xyxy: list[float] | tuple[float, ...]) -> None:
        z = xyxy_to_z(xyxy)
        self.x = [z[0], z[1], z[2], z[3], 0.0, 0.0, 0.0]
        self.P = [[0.0] * 7 for _ in range(7)]
        for i, value in enumerate(P0_DIAG):
            self.P[i][i] = value

    def predict(self) -> None:
        predict_state(self.x, self.P)

    def update(self, xyxy: list[float] | tuple[float, ...]) -> None:
        update_state(self.x, self.P, xyxy_to_z(xyxy))

    def xyxy(self) -> list[float]:
        box = z_to_xyxy(self.x)
        if any(math.isnan(v) for v in box):
            return [math.nan, math.nan, math.nan, math.nan]
        return box
