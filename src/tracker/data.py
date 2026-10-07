"""Download MOT17 labels and load public detections plus ground truth."""

from __future__ import annotations

import configparser
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import requests

from tracker.constants import (
    HELDOUT_SEQUENCES,
    MOT17_02_PEDESTRIAN_ROWS,
    MOT17_LABELS_SHA256,
    MOT17_LABELS_URL,
    PEDESTRIAN_CLASS,
    ROOT,
    TRAIN_SEQUENCES,
    VAL_SEQUENCES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "mot-portfolio/1.0"}) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)


def pedestrian_rows(gt_path: Path) -> int:
    count = 0
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 8 and int(float(parts[7])) == PEDESTRIAN_CLASS:
                count += 1
    return count


def ensure_dataset(root: Path | None = None) -> Path:
    """Cache MOT17Labels.zip under data/ and extract it to data/MOT17."""
    root = ROOT if root is None else root
    data = root / "data"
    archive = data / "MOT17Labels.zip"
    extracted = data / "MOT17"
    marker = extracted / "train" / "MOT17-02-FRCNN" / "gt" / "gt.txt"
    if not archive.exists() or _sha256(archive) != MOT17_LABELS_SHA256:
        _download(MOT17_LABELS_URL, archive)
    if _sha256(archive) != MOT17_LABELS_SHA256:
        raise RuntimeError("MOT17Labels.zip checksum does not match the pinned SHA-256")
    if not marker.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
    counted = pedestrian_rows(marker)
    if counted != MOT17_02_PEDESTRIAN_ROWS:
        raise RuntimeError(
            f"MOT17-02-FRCNN pedestrian rows are {counted}, expected {MOT17_02_PEDESTRIAN_ROWS}"
        )
    return extracted


def _load_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append([float(part) for part in line.split(",")])
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    width = max(len(row) for row in rows)
    array = np.zeros((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        array[i, : len(row)] = row
    return array


def load_sequence(name: str, max_frames: int | None = None, root: Path | None = None) -> dict:
    """Load one training sequence.

    Detections are ``frame, x, y, w, h, conf``. Ground truth rows are
    ``frame, id, x, y, w, h, conf, class``.
    """
    root = ROOT if root is None else root
    seq_dir = root / "data" / "MOT17" / "train" / name
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    seq_length = int(parser["Sequence"]["seqLength"])
    fps = int(parser["Sequence"]["frameRate"])
    if max_frames is not None:
        seq_length = min(seq_length, int(max_frames))
    det = _load_table(seq_dir / "det" / "det.txt")
    gt = _load_table(seq_dir / "gt" / "gt.txt")
    if det.size:
        keep = det[:, 0] <= seq_length
        det = det[keep]
        dets = np.column_stack([det[:, 0], det[:, 2], det[:, 3], det[:, 4], det[:, 5], det[:, 6]])
    else:
        dets = np.zeros((0, 6), dtype=np.float64)
    if gt.size:
        keep = gt[:, 0] <= seq_length
        gt = gt[keep]
        gt_rows = np.column_stack(
            [gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3], gt[:, 4], gt[:, 5], gt[:, 6], gt[:, 7]]
        )
    else:
        gt_rows = np.zeros((0, 8), dtype=np.float64)
    return {
        "name": name,
        "seq_length": seq_length,
        "fps": fps,
        "dets": dets.astype(np.float64),
        "gt": gt_rows.astype(np.float64),
    }


def canonical_names() -> dict[str, list[str]]:
    return {
        "train": list(TRAIN_SEQUENCES),
        "val": list(VAL_SEQUENCES),
        "heldout": list(HELDOUT_SEQUENCES),
    }
