"""README table and resume notes filled from metrics.json."""

from __future__ import annotations

import json
from pathlib import Path


def _pct(value: float) -> float:
    return float(f"{100.0 * value:.2f}")


def _num(value: float, digits: int) -> float:
    return float(f"{value:.{digits}f}")


def reported_numbers(metrics: dict) -> dict[str, float]:
    learned = metrics["metrics"]["heldout"]["learned"]
    sort = metrics["metrics"]["heldout"]["sort"]
    cpp = metrics["metrics"]["cpp"]
    sort_idsw = float(sort["id_switches"])
    learned_idsw = float(learned["id_switches"])
    if sort_idsw <= 0:
        reduction = 1.0 if learned_idsw <= 0 else 0.0
    else:
        reduction = (sort_idsw - learned_idsw) / sort_idsw
    return {
        "learned_idf1": _pct(learned["idf1"]),
        "sort_idf1": _pct(sort["idf1"]),
        "learned_mota": _pct(learned["mota"]),
        "sort_mota": _pct(sort["mota"]),
        "learned_hota": _pct(learned["hota"]),
        "sort_hota": _pct(sort["hota"]),
        "learned_id_switches": float(int(round(learned_idsw))),
        "sort_id_switches": float(int(round(sort_idsw))),
        "idsw_reduction_percent": _num(100.0 * reduction, 1),
        "speedup": _num(float(cpp["speedup"]), 2),
        "py_ms_per_frame": _num(float(cpp["py_ms_per_frame"]), 3),
        "cpp_ms_per_frame": _num(float(cpp["cpp_ms_per_frame"]), 3),
        "coverage_percent": _num(float(metrics["metrics"].get("coverage_percent", 0.0)), 1),
    }


def results_table(payload: dict) -> str:
    numbers = payload["reported"]
    learned = payload["metrics"]["heldout"]["learned"]
    sort = payload["metrics"]["heldout"]["sort"]
    lines = [
        "| Metric | SORT | Learned |",
        "|---|---:|---:|",
        f"| IDF1 | {numbers['sort_idf1']:.2f} | {numbers['learned_idf1']:.2f} |",
        f"| MOTA | {numbers['sort_mota']:.2f} | {numbers['learned_mota']:.2f} |",
        f"| HOTA | {numbers['sort_hota']:.2f} | {numbers['learned_hota']:.2f} |",
        f"| ID switches | {int(numbers['sort_id_switches'])} | {int(numbers['learned_id_switches'])} |",
        f"| False positives | {int(sort['false_positives'])} | {int(learned['false_positives'])} |",
        f"| False negatives | {int(sort['false_negatives'])} | {int(learned['false_negatives'])} |",
        "",
        f"C++ core identical to Python: {payload['metrics']['cpp']['identical']}. "
        f"Python {numbers['py_ms_per_frame']:.3f} ms/frame, "
        f"C++ {numbers['cpp_ms_per_frame']:.3f} ms/frame "
        f"({numbers['speedup']:.2f}x).",
        "",
    ]
    return "\n".join(lines)


def success_criteria(payload: dict) -> list[dict]:
    learned = payload["metrics"]["heldout"]["learned"]
    sort = payload["metrics"]["heldout"]["sort"]
    cpp = payload["metrics"]["cpp"]
    sort_idsw = float(sort["id_switches"])
    learned_idsw = float(learned["id_switches"])
    if sort_idsw <= 0:
        reduction = 1.0 if learned_idsw <= 0 else 0.0
    else:
        reduction = (sort_idsw - learned_idsw) / sort_idsw
    speedup = float(cpp["speedup"])
    identical = bool(cpp["identical"])
    coverage = float(payload["metrics"].get("coverage_percent", 0.0))
    return [
        {
            "name": "IDF1 on held-out sequences",
            "value": learned["idf1"] * 100.0,
            "op": ">=",
            "threshold": sort["idf1"] * 100.0 + 3.0,
            "threshold_high": None,
            "passed": bool(learned["idf1"] >= sort["idf1"] + 0.03 - 1e-12),
        },
        {
            "name": "ID switches",
            "value": reduction,
            "op": ">=",
            "threshold": 0.15,
            "threshold_high": None,
            "passed": bool(reduction >= 0.15 - 1e-12),
        },
        {
            "name": "MOTA on held-out sequences",
            "value": learned["mota"] * 100.0,
            "op": ">=",
            "threshold": sort["mota"] * 100.0,
            "threshold_high": None,
            "passed": bool(learned["mota"] >= sort["mota"] - 1e-12),
        },
        {
            "name": "C++ core correctness and speed",
            "value": speedup,
            "op": ">=",
            "threshold": 5.0,
            "threshold_high": None,
            "passed": bool(identical and speedup >= 5.0),
        },
        {
            "name": "Test coverage of src/",
            "value": coverage,
            "op": ">=",
            "threshold": 75.0,
            "threshold_high": None,
            "passed": bool(coverage >= 75.0),
        },
    ]


def render_resume(payload: dict) -> str:
    numbers = payload["reported"]
    summary = (
        "This project tracks pedestrians on MOT17 public detections with a constant-velocity Kalman filter. "
        "A PyTorch model scores broken SORT tracklets from motion and box geometry, and a C++ port of the SORT core checks the same track output."
    )
    bullets = [
        (
            f"Built a tracking-by-detection system on MOT17 with a Kalman filter and a learned PyTorch "
            f"association model, raising IDF1 to {numbers['learned_idf1']:.2f} vs. {numbers['sort_idf1']:.2f} "
            f"for SORT on held-out sequences"
        ),
        (
            f"Cut identity switches by {numbers['idsw_reduction_percent']:.1f}% through learned association "
            f"and track management, with failure-mode analysis of the remaining errors"
        ),
        (
            f"Wrote a C++17 Kalman/Hungarian tracking core with pybind11 bindings, matching Python outputs "
            f"at {numbers['speedup']:.2f}x the speed"
        ),
        (
            f"Scored the held-out sequences at MOTA {numbers['learned_mota']:.2f} and HOTA {numbers['learned_hota']:.2f} "
            f"({int(numbers['learned_id_switches'])} identity switches) against SORT at MOTA {numbers['sort_mota']:.2f}, "
            f"HOTA {numbers['sort_hota']:.2f}, and {int(numbers['sort_id_switches'])} identity switches"
        ),
    ]
    lines = [
        "# Resume notes",
        "",
        "## Draft bullets",
        "",
    ]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.extend(
        [
            "",
            "## Skills and tools",
            "",
            "Python, C++, PyTorch, NumPy, SciPy, pybind11, CMake, MSVC, scikit-build-core, motmetrics, TrackEval, pytest.",
            "",
            "## Project summary",
            "",
            summary,
            "",
            "## GitHub description",
            "",
            "Kalman-filter multi-object tracking on MOT17 with a learned association model and a matching C++ core.",
            "",
        ]
    )
    return "\n".join(lines)


def update_readme(readme: Path, payload: dict) -> None:
    text = readme.read_text(encoding="utf-8")
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    table = results_table(payload)
    block = f"{start}\n\n{table}\n{end}"
    if start in text and end in text:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        readme.write_text(pre + block + post, encoding="utf-8")
    else:
        readme.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def write_outputs(out_dir: Path, readme: Path, resume: Path, payload: dict) -> None:
    payload = json.loads(json.dumps(payload))
    payload["reported"] = reported_numbers(payload)
    payload["success_criteria"] = success_criteria(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload.get("mode") == "full":
        update_readme(readme, payload)
        resume.write_text(render_resume(payload), encoding="utf-8")
