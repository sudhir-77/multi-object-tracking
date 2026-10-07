# Resume notes

## Draft bullets

- Built a tracking-by-detection system on MOT17 with a Kalman filter and a learned PyTorch association model, raising IDF1 to 55.64 vs. 52.30 for SORT on held-out sequences
- Cut identity switches by 36.4% through learned association and track management, with failure-mode analysis of the remaining errors
- Wrote a C++17 Kalman/Hungarian tracking core with pybind11 bindings, matching Python outputs at 110.36x the speed
- Scored the held-out sequences at MOTA 50.31 and HOTA 48.58 (157 identity switches) against SORT at MOTA 49.89, HOTA 46.70, and 247 identity switches

## Skills and tools

Python, C++, PyTorch, NumPy, SciPy, pybind11, CMake, MSVC, scikit-build-core, motmetrics, TrackEval, pytest.

## Project summary

This project tracks pedestrians on MOT17 public detections with a constant-velocity Kalman filter. A PyTorch model scores broken SORT tracklets from motion and box geometry, and a C++ port of the SORT core checks the same track output.

## GitHub description

Kalman-filter multi-object tracking on MOT17 with a learned association model and a matching C++ core.
