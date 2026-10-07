// Kalman filter and SORT tracker. Operation order matches src/tracker/kalman.py
// and src/tracker/sort.py. MSVC is built with /fp:precise so contractions do
// not diverge from CPython floats.
#pragma float_control(precise, on)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double Q_DIAG[7] = {1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001};
constexpr double R_DIAG[4] = {1.0, 1.0, 10.0, 10.0};
constexpr double P0_DIAG[7] = {10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0};
constexpr double FORBID_COST = 1.0e6;
constexpr double HUNGARIAN_INF = 1.0e18;

struct Filter {
    double x[7];
    double P[7][7];
};

double box_iou(const double* a, const double* b) {
    const double ix1 = a[0] > b[0] ? a[0] : b[0];
    const double iy1 = a[1] > b[1] ? a[1] : b[1];
    const double ix2 = a[2] < b[2] ? a[2] : b[2];
    const double iy2 = a[3] < b[3] ? a[3] : b[3];
    const double iw = ix2 - ix1;
    const double ih = iy2 - iy1;
    if (iw <= 0.0 || ih <= 0.0) {
        return 0.0;
    }
    const double inter = iw * ih;
    const double area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const double area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const double uni = area_a + area_b - inter;
    if (uni <= 0.0) {
        return 0.0;
    }
    return inter / uni;
}

void xyxy_to_z(const double* box, double* z) {
    const double w = box[2] - box[0];
    const double h = box[3] - box[1];
    z[0] = box[0] + w * 0.5;
    z[1] = box[1] + h * 0.5;
    z[2] = w * h;
    z[3] = (h == 0.0) ? 0.0 : w / h;
}

void z_to_xyxy(const double* state, double* box) {
    const double s = state[2];
    const double r = state[3];
    const double sr = s * r;
    const double w = (sr <= 0.0) ? 0.0 : std::sqrt(sr);
    const double h = (w == 0.0) ? 0.0 : s / w;
    const double cx = state[0];
    const double cy = state[1];
    box[0] = cx - w * 0.5;
    box[1] = cy - h * 0.5;
    box[2] = cx + w * 0.5;
    box[3] = cy + h * 0.5;
}

void invert_4x4(const double A[4][4], double inv[4][4]) {
    double M[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            M[i][j] = A[i][j];
            M[i][4 + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        double best = std::fabs(M[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            const double value = std::fabs(M[r][col]);
            if (value > best) {
                best = value;
                piv = r;
            }
        }
        if (piv != col) {
            for (int j = 0; j < 8; ++j) {
                const double tmp = M[col][j];
                M[col][j] = M[piv][j];
                M[piv][j] = tmp;
            }
        }
        if (std::fabs(M[col][col]) < 1e-12) {
            M[col][col] += 1e-6;
        }
        const double div = M[col][col];
        for (int j = 0; j < 8; ++j) {
            M[col][j] /= div;
        }
        for (int r = 0; r < 4; ++r) {
            if (r == col) {
                continue;
            }
            const double factor = M[r][col];
            for (int j = 0; j < 8; ++j) {
                M[r][j] -= factor * M[col][j];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            inv[i][j] = M[i][4 + j];
        }
    }
}

void predict_state(Filter& f) {
    if (f.x[6] + f.x[2] <= 0.0) {
        f.x[6] = 0.0;
    }
    f.x[0] = f.x[0] + f.x[4];
    f.x[1] = f.x[1] + f.x[5];
    f.x[2] = f.x[2] + f.x[6];
    double M[7][7];
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            M[i][j] = f.P[i][j];
        }
    }
    for (int j = 0; j < 7; ++j) {
        M[0][j] = f.P[0][j] + f.P[4][j];
        M[1][j] = f.P[1][j] + f.P[5][j];
        M[2][j] = f.P[2][j] + f.P[6][j];
    }
    for (int i = 0; i < 7; ++i) {
        f.P[i][0] = M[i][0] + M[i][4];
        f.P[i][1] = M[i][1] + M[i][5];
        f.P[i][2] = M[i][2] + M[i][6];
        f.P[i][3] = M[i][3];
        f.P[i][4] = M[i][4];
        f.P[i][5] = M[i][5];
        f.P[i][6] = M[i][6];
        f.P[i][0] += (i == 0) ? Q_DIAG[0] : 0.0;
        f.P[i][1] += (i == 1) ? Q_DIAG[1] : 0.0;
        f.P[i][2] += (i == 2) ? Q_DIAG[2] : 0.0;
        f.P[i][3] += (i == 3) ? Q_DIAG[3] : 0.0;
        f.P[i][4] += (i == 4) ? Q_DIAG[4] : 0.0;
        f.P[i][5] += (i == 5) ? Q_DIAG[5] : 0.0;
        f.P[i][6] += (i == 6) ? Q_DIAG[6] : 0.0;
    }
}

void update_state(Filter& f, const double* xyxy) {
    double z[4];
    xyxy_to_z(xyxy, z);
    double y[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = z[i] - f.x[i];
    }
    double S[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            S[i][j] = f.P[i][j];
        }
    }
    S[0][0] += R_DIAG[0];
    S[1][1] += R_DIAG[1];
    S[2][2] += R_DIAG[2];
    S[3][3] += R_DIAG[3];
    double Sinv[4][4];
    invert_4x4(S, Sinv);
    double K[7][4];
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 4; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 4; ++k) {
                acc += f.P[i][k] * Sinv[k][j];
            }
            K[i][j] = acc;
        }
    }
    for (int i = 0; i < 7; ++i) {
        double acc = 0.0;
        for (int j = 0; j < 4; ++j) {
            acc += K[i][j] * y[j];
        }
        f.x[i] = f.x[i] + acc;
    }
    double Pnew[7][7];
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            double acc = f.P[i][j];
            for (int k = 0; k < 4; ++k) {
                acc -= K[i][k] * f.P[k][j];
            }
            Pnew[i][j] = acc;
        }
    }
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            f.P[i][j] = Pnew[i][j];
        }
    }
}

void init_filter(Filter& f, const double* xyxy) {
    double z[4];
    xyxy_to_z(xyxy, z);
    for (int i = 0; i < 4; ++i) {
        f.x[i] = z[i];
    }
    f.x[4] = 0.0;
    f.x[5] = 0.0;
    f.x[6] = 0.0;
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            f.P[i][j] = 0.0;
        }
        f.P[i][i] = P0_DIAG[i];
    }
}

struct Track {
    Filter kf;
    int id = 0;
    int hits = 0;
    int hit_streak = 0;
    int age = 0;
    int time_since_update = 0;
};

std::vector<std::pair<int, int>> hungarian(const std::vector<std::vector<double>>& cost_in) {
    std::vector<std::pair<int, int>> pairs;
    if (cost_in.empty() || cost_in[0].empty()) {
        return pairs;
    }
    int n = static_cast<int>(cost_in.size());
    int m = static_cast<int>(cost_in[0].size());
    bool transposed = false;
    std::vector<std::vector<double>> a = cost_in;
    if (n > m) {
        transposed = true;
        a.assign(m, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                a[j][i] = cost_in[i][j];
            }
        }
        n = static_cast<int>(a.size());
        m = static_cast<int>(a[0].size());
    }
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0), minv(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    std::vector<char> used(m + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), HUNGARIAN_INF);
        std::fill(used.begin(), used.end(), 0);
        while (true) {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = HUNGARIAN_INF;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) {
                    continue;
                }
                const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) {
                break;
            }
        }
        while (true) {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) {
                break;
            }
        }
    }
    for (int j = 1; j <= m; ++j) {
        const int i = p[j];
        if (i == 0) {
            continue;
        }
        if (transposed) {
            pairs.emplace_back(j - 1, i - 1);
        } else {
            pairs.emplace_back(i - 1, j - 1);
        }
    }
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

}  // namespace

py::array_t<int> assign(py::array_t<double, py::array::c_style | py::array::forcecast> cost) {
    auto buf = cost.unchecked<2>();
    const int n = static_cast<int>(buf.shape(0));
    const int m = static_cast<int>(buf.shape(1));
    std::vector<std::vector<double>> matrix(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = buf(i, j);
        }
    }
    const auto pairs = hungarian(matrix);
    py::array_t<int> out({static_cast<py::ssize_t>(pairs.size()), static_cast<py::ssize_t>(2)});
    auto view = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(pairs.size()); ++i) {
        view(i, 0) = pairs[i].first;
        view(i, 1) = pairs[i].second;
    }
    return out;
}

py::array_t<double> track_sort(
    py::array_t<double, py::array::c_style | py::array::forcecast> dets,
    int seq_length,
    double iou_threshold,
    int max_age,
    int min_hits) {
    auto buf = dets.unchecked<2>();
    const int n = static_cast<int>(buf.shape(0));
    std::vector<std::vector<std::array<double, 4>>> frames(static_cast<size_t>(seq_length));
    if (n > 0 && buf.shape(1) >= 5) {
        for (int i = 0; i < n; ++i) {
            const int frame = static_cast<int>(buf(i, 0));
            if (frame < 1 || frame > seq_length) {
                continue;
            }
            const double x = buf(i, 1);
            const double y = buf(i, 2);
            const double w = buf(i, 3);
            const double h = buf(i, 4);
            frames[static_cast<size_t>(frame - 1)].push_back({x, y, x + w, y + h});
        }
    }
    std::vector<Track> tracks;
    int next_id = 1;
    std::vector<std::array<double, 6>> rows;
    for (int frame_index = 0; frame_index < seq_length; ++frame_index) {
        const int frame_count = frame_index + 1;
        for (auto& tr : tracks) {
            predict_state(tr.kf);
            tr.age += 1;
            if (tr.time_since_update > 0) {
                tr.hit_streak = 0;
            }
            tr.time_since_update += 1;
        }
        std::vector<Track> alive;
        alive.reserve(tracks.size());
        for (auto& tr : tracks) {
            double box[4];
            z_to_xyxy(tr.kf.x, box);
            if (std::isnan(box[0]) || std::isnan(box[1]) || std::isnan(box[2]) || std::isnan(box[3])) {
                continue;
            }
            alive.push_back(tr);
        }
        tracks.swap(alive);
        const auto& dets_f = frames[static_cast<size_t>(frame_index)];
        const int n_d = static_cast<int>(dets_f.size());
        const int n_t = static_cast<int>(tracks.size());
        std::vector<std::pair<int, int>> matched;
        std::vector<int> unmatched_d;
        if (n_d == 0 || n_t == 0) {
            unmatched_d.resize(static_cast<size_t>(n_d));
            for (int i = 0; i < n_d; ++i) {
                unmatched_d[static_cast<size_t>(i)] = i;
            }
        } else {
            std::vector<std::array<double, 4>> preds(static_cast<size_t>(n_t));
            for (int j = 0; j < n_t; ++j) {
                z_to_xyxy(tracks[static_cast<size_t>(j)].kf.x, preds[static_cast<size_t>(j)].data());
            }
            std::vector<std::vector<double>> cost(static_cast<size_t>(n_d), std::vector<double>(static_cast<size_t>(n_t), 0.0));
            for (int i = 0; i < n_d; ++i) {
                for (int j = 0; j < n_t; ++j) {
                    const double iou = box_iou(dets_f[static_cast<size_t>(i)].data(), preds[static_cast<size_t>(j)].data());
                    if (iou < iou_threshold) {
                        cost[static_cast<size_t>(i)][static_cast<size_t>(j)] = FORBID_COST;
                    } else {
                        cost[static_cast<size_t>(i)][static_cast<size_t>(j)] = 1.0 - iou;
                    }
                }
            }
            const auto pairs = hungarian(cost);
            std::vector<char> matched_d(static_cast<size_t>(n_d), 0);
            for (const auto& pair : pairs) {
                if (cost[static_cast<size_t>(pair.first)][static_cast<size_t>(pair.second)] >= FORBID_COST * 0.5) {
                    continue;
                }
                matched.push_back(pair);
                matched_d[static_cast<size_t>(pair.first)] = 1;
            }
            for (int i = 0; i < n_d; ++i) {
                if (!matched_d[static_cast<size_t>(i)]) {
                    unmatched_d.push_back(i);
                }
            }
        }
        for (const auto& pair : matched) {
            update_state(tracks[static_cast<size_t>(pair.second)].kf, dets_f[static_cast<size_t>(pair.first)].data());
            tracks[static_cast<size_t>(pair.second)].time_since_update = 0;
            tracks[static_cast<size_t>(pair.second)].hits += 1;
            tracks[static_cast<size_t>(pair.second)].hit_streak += 1;
        }
        for (int i : unmatched_d) {
            Track born;
            init_filter(born.kf, dets_f[static_cast<size_t>(i)].data());
            born.id = next_id++;
            born.hits = 0;
            born.hit_streak = 0;
            born.age = 0;
            born.time_since_update = 0;
            tracks.push_back(born);
        }
        std::vector<Track> kept;
        kept.reserve(tracks.size());
        for (auto& tr : tracks) {
            if (tr.time_since_update < 1 && (tr.hit_streak >= min_hits || frame_count <= min_hits)) {
                double box[4];
                z_to_xyxy(tr.kf.x, box);
                rows.push_back({static_cast<double>(frame_count), static_cast<double>(tr.id), box[0], box[1], box[2] - box[0], box[3] - box[1]});
            }
            if (tr.time_since_update <= max_age) {
                kept.push_back(tr);
            }
        }
        tracks.swap(kept);
    }
    py::array_t<double> out({static_cast<py::ssize_t>(rows.size()), static_cast<py::ssize_t>(6)});
    auto view = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(rows.size()); ++i) {
        for (py::ssize_t j = 0; j < 6; ++j) {
            view(i, j) = rows[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
    return out;
}

PYBIND11_MODULE(tracker_cpp, m) {
    m.doc() = "C++17 Kalman filter and Hungarian SORT core";
    m.def("assign", &assign, "Minimum-cost assignment. Returns int array (K, 2).");
    m.def("track_sort", &track_sort, "SORT tracker. dets is (N, 5) frame,x,y,w,h.");
}
