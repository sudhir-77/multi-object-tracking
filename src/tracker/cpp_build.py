"""Build the C++ extension with scikit-build-core and the installed MSVC."""

from __future__ import annotations

import subprocess
import sys

from tracker.constants import ROOT


def ensure_extension(force: bool = False):
    if not force:
        try:
            import tracker_cpp

            return tracker_cpp
        except ImportError:
            pass
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "--force-reinstall",
        str(ROOT),
    ]
    subprocess.check_call(cmd, cwd=ROOT)
    import importlib

    import tracker_cpp

    return importlib.reload(tracker_cpp)
