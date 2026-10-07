"""End-to-end training, evaluation, and C++ parity."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from tracker.constants import (
    FEATURE_NAMES,
    HELDOUT_SEQUENCES,
    LEARNED_MAX_AGE,
    MATCH_THRESHOLDS,
    ROOT,
    SEED,
    SORT_IOU_THRESHOLD,
    SORT_MAX_AGE,
    SORT_MIN_HITS,
    TRAIN_SEQUENCES,
    VAL_SEQUENCES,
    split_dict,
)
from tracker.cpp_build import ensure_extension
from tracker.data import ensure_dataset, load_sequence
from tracker.failure import classify_switches
from tracker.learned import frames_from_array, track_learned_frames
from tracker.metrics import prepare_sequence, score_sequences
from tracker.model import AssociationNet
from tracker.plots import plot_failure_modes, plot_metric_bars, plot_training_loss
from tracker.report import write_outputs
from tracker.serialize import format_mot
from tracker.sort import track_sort_array
from tracker.train import train_association


def _require(names: list[str], allowed: list[str], what: str) -> None:
    extra = sorted(set(names) - set(allowed))
    if extra:
        raise RuntimeError(f"{what} used sequences outside its split: {extra}")


class _PeakMonitor:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        proc = psutil.Process()
        while not self._stop.wait(0.2):
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, rss)

    def start(self) -> None:
        self.peak = psutil.Process().memory_info().rss
        self._thread.start()

    def stop_gb(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self.peak / (1024.0**3)


def _configure_runtime(device: str) -> torch.device:
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _public_metrics(scores: dict[str, float]) -> dict[str, float]:
    return {key: float(scores[key]) for key in scores if key != "accumulator"}


def _run_sort(sequence: dict) -> np.ndarray:
    dets = sequence["dets"][:, :5] if len(sequence["dets"]) else np.zeros((0, 5))
    return track_sort_array(
        dets,
        int(sequence["seq_length"]),
        iou_threshold=SORT_IOU_THRESHOLD,
        max_age=SORT_MAX_AGE,
        min_hits=SORT_MIN_HITS,
    )


def _run_learned(sequence: dict, model: AssociationNet, mean: np.ndarray, std: np.ndarray, threshold: float, device: torch.device) -> np.ndarray:
    frames, confs = frames_from_array(sequence["dets"], int(sequence["seq_length"]))
    rows = track_learned_frames(
        frames,
        confs,
        model,
        mean,
        std,
        threshold=threshold,
        max_age=LEARNED_MAX_AGE,
        device=device,
    )
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _score_tracker(sequences: list[dict], rows_by_name: dict[str, np.ndarray]) -> tuple[dict[str, float], list[list]]:
    prepared = [prepare_sequence(seq["gt"], rows_by_name[seq["name"]], int(seq["seq_length"])) for seq in sequences]
    return score_sequences(prepared), prepared


def _select_threshold(
    sequences: list[dict],
    sort_rows: dict[str, np.ndarray],
    model: AssociationNet,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: tuple[float, ...],
    device: torch.device,
) -> tuple[float, dict[str, float], list[dict]]:
    _require([seq["name"] for seq in sequences], VAL_SEQUENCES, "threshold selection")
    sort_scores, _ = _score_tracker(sequences, sort_rows)
    records = []
    best_threshold = thresholds[0]
    best_key = None
    best_scores = sort_scores
    for threshold in thresholds:
        learned_rows = {
            seq["name"]: _run_learned(seq, model, mean, std, threshold, device) for seq in sequences
        }
        scores, _ = _score_tracker(sequences, learned_rows)
        feasible = scores["mota"] >= sort_scores["mota"] - 1e-12 and scores["id_switches"] <= 0.85 * sort_scores["id_switches"] + 1e-9
        record = {"threshold": threshold, "feasible": bool(feasible), **_public_metrics(scores)}
        records.append(record)
        key = (1 if feasible else 0, scores["idf1"])
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_scores = scores
        print(
            f"val threshold {threshold:.2f}: IDF1 {scores['idf1']:.4f} MOTA {scores['mota']:.4f} "
            f"IDSW {scores['id_switches']:.0f} feasible {feasible}",
            flush=True,
        )
    return best_threshold, _public_metrics(best_scores), records


def _write_tracks(folder: Path, rows_by_name: dict[str, np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_by_name.items():
        (folder / f"{name}.txt").write_text(format_mot(rows), encoding="utf-8")


def _benchmark(sequences: list[dict], repeats: int) -> dict[str, float | bool]:
    extension = ensure_extension(force=False)
    py_rows = []
    cpp_rows = []
    total_frames = 0
    py_seconds = 0.0
    cpp_seconds = 0.0
    for sequence in sequences:
        dets = sequence["dets"][:, :5] if len(sequence["dets"]) else np.zeros((0, 5), dtype=np.float64)
        dets = np.ascontiguousarray(dets, dtype=np.float64)
        length = int(sequence["seq_length"])
        total_frames += length
        kwargs = dict(iou_threshold=SORT_IOU_THRESHOLD, max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS)
        track_sort_array(dets, length, **kwargs)
        extension.track_sort(dets, length, SORT_IOU_THRESHOLD, SORT_MAX_AGE, SORT_MIN_HITS)
        py_samples = []
        cpp_samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            py_out = track_sort_array(dets, length, **kwargs)
            py_samples.append(time.perf_counter() - t0)
            t1 = time.perf_counter()
            cpp_out = np.asarray(extension.track_sort(dets, length, SORT_IOU_THRESHOLD, SORT_MAX_AGE, SORT_MIN_HITS))
            cpp_samples.append(time.perf_counter() - t1)
        py_samples.sort()
        cpp_samples.sort()
        py_seconds += py_samples[len(py_samples) // 2]
        cpp_seconds += cpp_samples[len(cpp_samples) // 2]
        py_rows.append(format_mot(py_out))
        cpp_rows.append(format_mot(cpp_out))
    identical = py_rows == cpp_rows
    py_ms = py_seconds / max(total_frames, 1) * 1000.0
    cpp_ms = cpp_seconds / max(total_frames, 1) * 1000.0
    speedup = py_ms / cpp_ms if cpp_ms > 0 else 0.0
    return {
        "identical": identical,
        "py_ms_per_frame": py_ms,
        "cpp_ms_per_frame": cpp_ms,
        "speedup": speedup,
    }


def _coverage() -> float:
    env = os.environ.copy()
    env["MOT_SKIP_COVERAGE"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    cov_file = ROOT / "coverage.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=src",
        "--cov-report",
        f"json:{cov_file}",
        "-q",
        "--ignore=build",
    ]
    subprocess.check_call(cmd, cwd=ROOT, env=env)
    payload = json.loads(cov_file.read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"])


def run_pipeline(quick: bool = False, out_dir: Path | None = None, device: str = "cpu", skip_coverage: bool = False) -> dict:
    started = time.perf_counter()
    monitor = _PeakMonitor()
    monitor.start()
    dev = _configure_runtime(device)
    out = ROOT / "results" if out_dir is None else out_dir
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    mode = "quick" if quick else "full"
    print(f"mode {mode}, device {dev}", flush=True)
    ensure_dataset()
    split = split_dict()
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    max_frames = 25 if quick else None
    epochs = 2 if quick else 40
    thresholds = (0.30,) if quick else MATCH_THRESHOLDS
    repeats = 1 if quick else 5
    train_names = TRAIN_SEQUENCES[:1] if quick else list(TRAIN_SEQUENCES)
    val_names = list(VAL_SEQUENCES)
    held_names = list(HELDOUT_SEQUENCES)
    _require(train_names, TRAIN_SEQUENCES, "training")
    _require(val_names, VAL_SEQUENCES, "validation")
    _require(held_names, HELDOUT_SEQUENCES, "held-out evaluation")

    print("building C++ extension", flush=True)
    ensure_extension(force=not quick)

    train_seqs = [load_sequence(name, max_frames=max_frames) for name in train_names]
    val_seqs = [load_sequence(name, max_frames=max_frames) for name in val_names]
    held_seqs = [load_sequence(name, max_frames=max_frames) for name in held_names]

    print("training association model", flush=True)
    model, mean, std, losses = train_association(
        train_seqs,
        epochs=epochs,
        cap_per_class=2000 if quick else 20000,
        device=dev,
    )
    torch.save(
        {"state_dict": model.state_dict(), "mean": mean, "std": std, "feature_names": list(FEATURE_NAMES)},
        out / "association.pt",
    )
    (out / "train_loss.csv").write_text(
        "epoch,loss\n" + "\n".join(f"{i},{loss}" for i, loss in enumerate(losses, start=1)) + "\n",
        encoding="utf-8",
    )
    plot_training_loss(losses, out / "figures" / "training_loss.png")

    print("scoring SORT on validation", flush=True)
    val_sort = {seq["name"]: _run_sort(seq) for seq in val_seqs}
    threshold, _val_learned, val_records = _select_threshold(val_seqs, val_sort, model, mean, std, thresholds, dev)
    val_sort_scores, _ = _score_tracker(val_seqs, val_sort)
    print(f"selected match threshold {threshold:.2f}", flush=True)

    print("evaluating held-out sequences", flush=True)
    held_sort = {seq["name"]: _run_sort(seq) for seq in held_seqs}
    held_learned = {seq["name"]: _run_learned(seq, model, mean, std, threshold, dev) for seq in held_seqs}
    _write_tracks(out / "tracks" / "sort", held_sort)
    _write_tracks(out / "tracks" / "learned", held_learned)
    sort_scores, sort_prepared = _score_tracker(held_seqs, held_sort)
    learned_scores, learned_prepared = _score_tracker(held_seqs, held_learned)

    per_rows = []
    for tracker_name, prepared, sequences, bundle in (
        ("sort", sort_prepared, held_seqs, held_sort),
        ("learned", learned_prepared, held_seqs, held_learned),
    ):
        for seq, frames in zip(sequences, prepared):
            from tracker.metrics import motmetrics_score, hota_score

            one = motmetrics_score(frames)
            one.pop("accumulator", None)
            one["hota"] = hota_score(frames)
            one["sequence"] = seq["name"]
            one["tracker"] = tracker_name
            one["rows"] = int(len(bundle[seq["name"]]))
            per_rows.append(one)
    csv_path = out / "per_sequence_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sequence", "tracker", "mota", "idf1", "hota", "id_switches", "false_positives", "false_negatives", "num_objects", "rows"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in per_rows:
            writer.writerow(row)

    failure_counts = {"occlusion": 0, "crossing": 0, "short_gap": 0, "other": 0}
    examples = []
    for frames in learned_prepared:
        report = classify_switches(frames)
        for key, value in report["counts"].items():
            failure_counts[key] = failure_counts.get(key, 0) + int(value)
        examples.extend(report["examples"])
    classified = int(sum(failure_counts.values()))
    failure_payload = {
        "motmetrics_id_switches": learned_scores["id_switches"],
        "classified_id_switches": classified,
        "counts": failure_counts,
        "shares": {key: (value / classified if classified else 0.0) for key, value in failure_counts.items()},
        "examples": examples[:12],
        "notes": (
            "Categories come from per-frame IoU matching at 0.5. "
            "occlusion: previous match was at least 5 frames earlier. "
            "crossing: another pedestrian center was within 1.5 box heights. "
            "short_gap: the previous match was 2 to 4 frames earlier. "
            "other: the identity changed on consecutive matched frames."
        ),
    }
    (out / "failure_analysis.json").write_text(json.dumps(failure_payload, indent=2) + "\n", encoding="utf-8")
    plot_failure_modes(failure_counts, out / "figures" / "failure_modes.png")
    plot_metric_bars(sort_scores, learned_scores, out / "figures" / "heldout_metrics.png")

    print("benchmarking C++ core", flush=True)
    parity = _benchmark(held_seqs, repeats)
    (out / "cpp_parity.json").write_text(
        json.dumps(
            {
                "identical": bool(parity["identical"]),
                "py_ms_per_frame": parity["py_ms_per_frame"],
                "cpp_ms_per_frame": parity["cpp_ms_per_frame"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"cpp identical {parity['identical']} speedup {parity['speedup']:.2f}x "
        f"({parity['py_ms_per_frame']:.3f} vs {parity['cpp_ms_per_frame']:.3f} ms/frame)",
        flush=True,
    )

    coverage = 0.0
    if mode == "full" and not skip_coverage and os.environ.get("MOT_SKIP_COVERAGE") != "1":
        print("measuring test coverage", flush=True)
        coverage = _coverage()
        print(f"coverage {coverage:.1f}%", flush=True)

    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine()
    payload = {
        "project": "multi-object-tracking",
        "mode": mode,
        "success_criteria": [],
        "metrics": {
            "heldout": {"learned": _public_metrics(learned_scores), "sort": _public_metrics(sort_scores)},
            "val": {
                "sort": _public_metrics(val_sort_scores),
                "selected_threshold": threshold,
                "learned_at_selected_threshold": _val_learned,
                "grid": val_records,
            },
            "cpp": parity,
            "coverage_percent": coverage,
            "failure_counts": failure_counts,
            "train_loss": losses,
            "learned_max_age": LEARNED_MAX_AGE,
            "sort_max_age": SORT_MAX_AGE,
            "sort_iou_threshold": SORT_IOU_THRESHOLD,
            "sort_min_hits": SORT_MIN_HITS,
        },
        "baseline": _public_metrics(sort_scores),
        "splits": {"MOT17": {"train": len(TRAIN_SEQUENCES), "val": len(VAL_SEQUENCES), "test": len(HELDOUT_SEQUENCES)}},
        "seeds": [SEED],
        "hardware": {
            "cpu": cpu,
            "ram_gb": round(psutil.virtual_memory().total / (1024.0**3), 2),
            "gpu": None if dev.type == "cpu" else torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "os": platform.platform(),
            "peak_rss_gb": round(monitor.stop_gb(), 3),
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    write_outputs(out, ROOT / "README.md", ROOT / "RESUME_NOTES.md", payload)
    # runtime includes the write; refresh the stored duration once more
    payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
    payload["hardware"]["peak_rss_gb"] = round(monitor.stop_gb(), 3)
    write_outputs(out, ROOT / "README.md", ROOT / "RESUME_NOTES.md", payload)
    print(f"wrote {out / 'metrics.json'} in {payload['runtime_seconds']:.1f}s", flush=True)
    return payload
