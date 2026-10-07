import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quick_entrypoint_writes_metrics_schema(tmp_path):
    env = os.environ.copy()
    env["MOT_SKIP_COVERAGE"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    subprocess.check_call(
        [sys.executable, "run.py", "--quick", "--out", str(tmp_path), "--skip-coverage"],
        cwd=ROOT,
        env=env,
        timeout=180,
    )
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["project"] == "multi-object-tracking"
    assert "success_criteria" in payload
    assert "baseline" in payload
    assert payload["splits"]["MOT17"]["test"] == 2
    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert len(split["heldout"]) == 2
    assert set(split["train"]).isdisjoint(split["heldout"])
    assert (tmp_path / "cpp_parity.json").exists()
    for name in split["heldout"]:
        assert (tmp_path / "tracks" / "sort" / f"{name}.txt").exists()
        assert (tmp_path / "tracks" / "learned" / f"{name}.txt").exists()
