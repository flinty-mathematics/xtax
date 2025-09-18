#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"

// Hard bound on the magnitude of any entry of A or X. Moves that would push an
// entry beyond this are rejected. This keeps the search numerically sane and
// guarantees the int64 score arithmetic below cannot overflow in practice.
constexpr int64_t MAGNITUDE_LIMIT = 1ll << 48;

// Bound on |s| for an Add (shear) move. With entries bounded by MAGNITUDE_LIMIT
// (2^48) this keeps every intermediate product (s*entry, s*s*entry) comfortably
// inside the int64 range, so we can range-check results after the fact.
constexpr int64_t SHEAR_CAP = 64;

// Largest off-diagonal entry in a +/-1 pivot row that the deflation stage will
// clear. The clearing shears have |s| <= this value, and with matrix entries
// bounded by MAGNITUDE_LIMIT (2^48) the product |s| * entry stays below 2^62, so
// the int64 arithmetic in try_add cannot overflow. A unit pivot worth locking in
// has tiny off-diagonal mass anyway, so this generous cap is rarely the limit.
constexpr int64_t DEFLATE_OFFDIAG_CAP = 1 << 13;

struct Matrix {
    size_t n = 0;
    std::vector<int64_t> data;

    Matrix() = default;
    explicit Matrix(size_t n_) : n(n_), data(n_ * n_, 0) {}

    void fill_identity() {
        std::fill(data.begin(), data.end(), 0);
        for (size_t i = 0; i < n; ++i) data[i * n + i] = 1;
    }
    inline int64_t& at(int i, int j) { return data[(size_t)i * n + j]; }
    inline const int64_t& at(int i, int j) const { return data[(size_t)i * n + j]; }

    void print() const {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::cout << at((int)i, (int)j);
                if (j + 1 < n) std::cout << ',';
            }
            std::cout << '\n';
        }
    }

    bool is_diagonal() const {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j && at((int)i, (int)j) != 0) return false;
            }
        }
        return true;
    }

    // Number of nonzero off-diagonal pairs {i,j}, i < j. Zero iff diagonal.
    int64_t count_offdiag_nonzero() const {
        int64_t c = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                if (at((int)i, (int)j) != 0) ++c;
        return c;
    }

    // Sparsity score: 2 * sum|A_ij| - sum|A_ii|. Lower is more diagonal.
    // Computed once up front, then maintained incrementally by the worker.
    int64_t score() const {
        int64_t full_sum = 0;
        for (const auto e : data) full_sum += std::llabs(e);
        int64_t diag_sum = 0;
        for (size_t i = 0; i < n; ++i) diag_sum += std::llabs(data[i * n + i]);
        return 2 * full_sum - diag_sum;
    }
};

enum class TType { Add, Swap, Neg };
struct Trans { TType type; int i, j; int s; };

// Congruence move applied to the accumulating transform, which we store
// transposed (Xt = X^T). The move acts on X by column operations, so on Xt it
// is the corresponding row operation, which is contiguous and cache-friendly.
void apply_to_Xt(Matrix& Xt, const Trans& t) {
    const size_t n = Xt.n;
    int64_t* base = Xt.data.data();

    if (t.type == TType::Add) {
        int64_t* row_j = base + (size_t)t.j * n;
        const int64_t* row_i = base + (size_t)t.i * n;
        const int64_t s = (int64_t)t.s;
        for (size_t k = 0; k < n; ++k) row_j[k] += s * row_i[k];
    }
    else if (t.type == TType::Swap) {
        int64_t* row_i = base + (size_t)t.i * n;
        int64_t* row_j = base + (size_t)t.j * n;
        for (size_t k = 0; k < n; ++k) std::swap(row_i[k], row_j[k]);
    }
    else {
        int64_t* row_i = base + (size_t)t.i * n;
        for (size_t k = 0; k < n; ++k) row_i[k] = -row_i[k];
    }
}

// Congruence move applied to the symmetric matrix A as A -> P^T A P.
void apply_to_A(Matrix& mtx, const Trans& t) {
    const size_t n = mtx.n;

    switch (t.type) {
        case TType::Neg: {
            int i = t.i;
            for (size_t k = 0; k < n; ++k) {
                mtx.at(i, (int)k) = -mtx.at(i, (int)k);
                mtx.at((int)k, i) = -mtx.at((int)k, i);
            }
            break;
        }
        case TType::Swap: {
            int a = t.i, b = t.j;
            for (size_t k = 0; k < n; ++k) {
                if (k != (size_t)a && k != (size_t)b) {
                    std::swap(mtx.at(a, (int)k), mtx.at(b, (int)k));
                    std::swap(mtx.at((int)k, a), mtx.at((int)k, b));
                }
            }
            std::swap(mtx.at(a, a), mtx.at(b, b));
            std::swap(mtx.at(a, b), mtx.at(b, a));
            break;
        }
        case TType::Add: {
            int i = t.i, j = t.j, s = t.s;
            for (size_t k = 0; k < n; ++k) {
                if (k != (size_t)i && k != (size_t)j) {
                    int64_t v = mtx.at(j, (int)k) + (int64_t)s * mtx.at(i, (int)k);
                    mtx.at(j, (int)k) = v;
                    mtx.at((int)k, j) = v;
                }
            }
            int64_t a_ji = mtx.at(j, i) + (int64_t)s * mtx.at(i, i);
            mtx.at(j, i) = a_ji;
            mtx.at(i, j) = a_ji;

            // new_jj = old_jj + 2*s*old_ji + s*s*old_ii
            // expressed with a_ji (new_ji): new_jj = old_jj + 2*s*new_ji - s*s*old_ii
            mtx.at(j, j) += (int64_t)2 * s * mtx.at(j, i) - (int64_t)s * s * mtx.at(i, i);
            break;
        }
    }
}

// Round a/b to the nearest integer (ties away from zero). Requires b != 0.
static int64_t rounded_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a - q * b;
    if (2 * std::llabs(r) >= std::llabs(b)) {
        q += ((a < 0) == (b < 0)) ? 1 : -1;
    }
    return q;
}

// Evaluate an Add(i,j,s) move without mutating A or Xt.
//
// An Add only changes column j of X (row j of the transposed Xt) and row/column
// j (plus the diagonal jj) of the symmetric A, so everything here is O(n).
// Returns false (the move is infeasible) if any affected entry of X or A would
// exceed MAGNITUDE_LIMIT. On success new_a_row holds the proposed row j of A
// (with new_a_row[j] = new A_jj), new_xt_row holds the proposed row j of Xt,
// delta is the score change, and d_nonzero is the change in the number of
// nonzero off-diagonal pairs.
static bool try_add(const Matrix& A, const Matrix& Xt, int i, int j, int64_t s,
                    std::vector<int64_t>& new_a_row, std::vector<int64_t>& new_xt_row,
                    int64_t& delta, int64_t& d_nonzero) {
    const int n = (int)A.n;

    // Row j of Xt (i.e. column j of X) is the only part of X that changes.
    const int64_t* xt_row_j = Xt.data.data() + (size_t)j * n;
    const int64_t* xt_row_i = Xt.data.data() + (size_t)i * n;
    for (int r = 0; r < n; ++r) {
        const int64_t v = xt_row_j[r] + s * xt_row_i[r];
        if (v > MAGNITUDE_LIMIT || v < -MAGNITUDE_LIMIT) return false;
        new_xt_row[r] = v;
    }

    const int64_t a_ii = A.at(i, i);
    const int64_t a_ji = A.at(j, i);
    const int64_t* a_row_j = A.data.data() + (size_t)j * n;
    const int64_t* a_row_i = A.data.data() + (size_t)i * n;

    // sum over k != j of (|new A_jk| - |old A_jk|), plus off-diagonal zero count.
    int64_t off_delta = 0;
    int64_t dnz = 0;
    for (int k = 0; k < n; ++k) {
        if (k == j) continue;
        const int64_t old = a_row_j[k];
        const int64_t nv = old + s * a_row_i[k];
        if (nv > MAGNITUDE_LIMIT || nv < -MAGNITUDE_LIMIT) return false;
        new_a_row[k] = nv;
        off_delta += std::llabs(nv) - std::llabs(old);
        if (old == 0) { if (nv != 0) ++dnz; }
        else if (nv == 0) --dnz;
    }

    const int64_t old_jj = a_row_j[j];
    const int64_t new_jj = old_jj + 2 * s * a_ji + s * s * a_ii;
    if (new_jj > MAGNITUDE_LIMIT || new_jj < -MAGNITUDE_LIMIT) return false;
    new_a_row[j] = new_jj;

    // Each off-diagonal change is counted twice in 2*sum|A| (symmetric pair). The
    // diagonal change contributes once to 2*sum|A| and once to -sum|A_ii|.
    delta = 4 * off_delta + (std::llabs(new_jj) - std::llabs(old_jj));
    d_nonzero = dnz;
    return true;
}

// Commit the precomputed result of try_add into A (row/col j) and Xt (row j),
// and keep row_mass (per-row off-diagonal absolute mass) up to date.
static void commit_add(Matrix& A, Matrix& Xt, int j,
                       const std::vector<int64_t>& new_a_row,
                       const std::vector<int64_t>& new_xt_row,
                       std::vector<int64_t>& row_mass) {
    const int n = (int)A.n;
    int64_t* a_row_j = A.data.data() + (size_t)j * n;
    int64_t mass_j = 0;
    for (int k = 0; k < n; ++k) {
        if (k == j) continue;
        const int64_t old = a_row_j[k];
        const int64_t nv = new_a_row[k];
        a_row_j[k] = nv;             // contiguous row write
        A.at(k, j) = nv;             // strided symmetric mirror
        const int64_t na = std::llabs(nv);
        row_mass[k] += na - std::llabs(old);
        mass_j += na;
    }
    a_row_j[j] = new_a_row[j];
    row_mass[j] = mass_j;
    int64_t* xt_row_j = Xt.data.data() + (size_t)j * n;
    for (int r = 0; r < n; ++r) xt_row_j[r] = new_xt_row[r];
}

// row_mass[r] = sum over k != r of |A_rk|. Recomputed when a worker (re)seeds.
static void compute_row_mass(const Matrix& A, std::vector<int64_t>& row_mass) {
    const int n = (int)A.n;
    for (int r = 0; r < n; ++r) {
        const int64_t* row = A.data.data() + (size_t)r * n;
        int64_t m = 0;
        for (int k = 0; k < n; ++k) if (k != r) m += std::llabs(row[k]);
        row_mass[r] = m;
    }
}

// A holds the working matrix. Xt is the accumulating transform stored
// transposed (Xt = X^T). off_nonzero tracks nonzero off-diagonal pairs so the
// diagonal test is O(1).
struct Congruence { Matrix A; Matrix Xt; int64_t score; int64_t off_nonzero; };

Matrix transpose(const Matrix& M) {
    Matrix R(M.n);
    for (size_t i = 0; i < M.n; ++i)
        for (size_t j = 0; j < M.n; ++j)
            R.at((int)i, (int)j) = M.at((int)j, (int)i);
    return R;
}

void write_matrix_csv(const Matrix& M, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    int n = static_cast<int>(M.n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << M.at(i, j);
            if (j + 1 < n) out << ",";
        }
        out << "\n";
    }
}

Matrix read_matrix_csv(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) throw std::runtime_error("Cannot open " + filename);

    std::vector<int64_t> raw;
    std::string line;
    size_t rows = 0, cols = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        ++rows;
        std::stringstream ss(line);
        std::string token;
        size_t inner = 0;
        while (std::getline(ss, token, ',')) {
            ++inner;
            raw.push_back(static_cast<int64_t>(std::stoll(token)));
        }
        if (cols == 0) cols = inner;
        if (inner != cols) throw std::runtime_error("Bad CSV row length in " + filename);
    }
    if (rows == 0) throw std::runtime_error("Empty matrix file: " + filename);
    if (rows != cols) throw std::runtime_error("Matrix must be square in " + filename);

    Matrix A(rows);
    A.data = std::move(raw);
    return A;
}

// A lattice basis: m vectors (rows) living in dimension d (columns). For a
// genuine basis m <= d, and the Gram matrix A = L L^T is then m x m.
struct Lattice {
    int m = 0;                  // number of basis vectors (rows)
    int d = 0;                  // ambient dimension (columns)
    std::vector<int64_t> data;  // row-major, size m*d
    int64_t* row(int i) { return data.data() + (size_t)i * d; }
    const int64_t* row(int i) const { return data.data() + (size_t)i * d; }
};

// Read a lattice basis CSV (rows are vectors). Unlike read_matrix_csv this does
// not require the matrix to be square, so m x d bases with m != d are accepted.
Lattice read_lattice_csv(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) throw std::runtime_error("Cannot open " + filename);

    std::vector<int64_t> raw;
    std::string line;
    size_t rows = 0, cols = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        ++rows;
        std::stringstream ss(line);
        std::string token;
        size_t inner = 0;
        while (std::getline(ss, token, ',')) {
            ++inner;
            raw.push_back(static_cast<int64_t>(std::stoll(token)));
        }
        if (cols == 0) cols = inner;
        if (inner != cols) throw std::runtime_error("Bad CSV row length in " + filename);
    }
    if (rows == 0) throw std::runtime_error("Empty lattice file: " + filename);

    Lattice L;
    L.m = (int)rows;
    L.d = (int)cols;
    L.data = std::move(raw);
    return L;
}

void write_lattice_csv(const Lattice& L, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open output file: " + filename);
    for (int i = 0; i < L.m; ++i) {
        const int64_t* ri = L.row(i);
        for (int k = 0; k < L.d; ++k) {
            out << ri[k];
            if (k + 1 < L.d) out << ',';
        }
        out << '\n';
    }
}

struct Params {
    int workers = 1;
    int stuck_threshold = 20000;   // moves without improvement before reheating
    double t_init = 0.0;           // initial SA temperature (<= 0 auto-calibrates)
    double t_min = 1e-3;           // floor temperature
    double cooling = 0.999;        // geometric factor applied per cooling step
    int moves_per_cool = 200;      // moves between cooling steps
    double reheat = 1.0;           // fraction of t_init restored when stuck
    double greedy_fraction = 0.5;  // probability an Add uses the reducing quotient
    double add_weight = 0.8;       // relative weight of Add moves (only score-changing move)
    double swap_weight = 0.1;      // relative weight of Swap moves
    double neg_weight = 0.1;       // relative weight of Neg moves
    double target_fraction = 0.5;  // probability an Add targets a hot row instead of uniform
    int target_samples = 8;        // tournament size for hot-row / large-pivot selection
    double reseed_factor = 1.25;   // reseed from global best when stuck and this far behind
    double max_seconds = 0.0;      // wall-clock stop (<= 0 runs until a diagonal is found)
    double save_interval = 2.0;    // minimum seconds between best_*.csv disk writes
    bool deflate = false;          // strict deflation outer loop (requires a unimodular matrix)
    bool deflate_blocks = false;   // relaxed deflation: peel off orthogonal summands (any Gram)
    double deflate_slice = 0.5;    // annealing seconds per slice before checking for pivots
    bool verbose = false;          // also show the inner annealer's progress inside deflate modes
    bool quiet = false;            // internal: silence per-worker console/disk output (deflation slices)
};

// One independent simulated-annealing worker. Workers explore on their own
// schedule and only share the global best, so they are not forced into lockstep.
void anneal_worker(const Params& params, uint32_t seed,
                   Congruence& global_best,
                   std::atomic<int64_t>& best_score_atomic,
                   std::mutex& best_mtx,
                   std::mutex& print_mtx,
                   std::atomic<bool>& done_flag,
                   std::atomic<uint64_t>& total_moves,
                   std::chrono::steady_clock::time_point t0,
                   double& last_save_elapsed,
                   int thread_id,
                   const std::vector<int>& active) {
    Congruence cur;
    {
        std::lock_guard<std::mutex> lk(best_mtx);
        cur = global_best;
    }
    const int n = (int)cur.A.n;

    std::mt19937 rng(seed);
    // Move selection samples only the active indices (the full index set when no
    // deflation is in effect). idx(rng) returns an actual matrix index, so all the
    // call sites below are unchanged whether or not the problem has been deflated.
    std::uniform_int_distribution<int> aidx(0, (int)active.size() - 1);
    auto idx = [&](std::mt19937& r) -> int { return active[(size_t)aidx(r)]; };
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::vector<int64_t> new_a_row(n), new_xt_row(n);
    std::vector<int64_t> row_mass(n);
    compute_row_mass(cur.A, row_mass);

    // Try Add(i,j,s) at temperature Tcur. Returns -1 if infeasible or rejected,
    // 0 if accepted without lowering the score, 1 if accepted and the score
    // strictly decreased. On accept it updates A, Xt, score, off_nonzero and
    // row_mass. Shared by ordinary shears and jumps.
    auto attempt_add = [&](int i, int j, int64_t s, double Tcur) -> int {
        int64_t delta = 0, dnz = 0;
        if (!try_add(cur.A, cur.Xt, i, j, s, new_a_row, new_xt_row, delta, dnz)) return -1;
        if (!(delta <= 0 || unit(rng) < std::exp(-(double)delta / Tcur))) return -1;
        commit_add(cur.A, cur.Xt, j, new_a_row, new_xt_row, row_mass);
        cur.score += delta;
        cur.off_nonzero += dnz;
        return delta < 0 ? 1 : 0;
    };

    // Row weight used to pick the target of a shear: bias toward the rows
    // carrying the most off-diagonal mass.
    auto row_weight = [&](int r) -> int64_t { return row_mass[r]; };

    // Console output is shown for a normal run, or in --verbose even when this is
    // an inner slice of a deflation loop. Disk writes stay
    // tied to !quiet so wrapped slices never clobber the coordinator's files.
    const bool show = !params.quiet || params.verbose;

    // Pick a target row by a hot-row tournament, then a pivot as the largest
    // off-diagonal entry of that row (another tournament). Stays stochastic.
    auto pick_pair_targeted = [&](int& pivot, int& target) {
        int jbest = idx(rng);
        for (int c = 1; c < params.target_samples; ++c) {
            const int cand = idx(rng);
            if (row_weight(cand) > row_weight(jbest)) jbest = cand;
        }
        target = jbest;
        const int64_t* a_row_j = cur.A.data.data() + (size_t)target * n;
        int ibest = -1;
        int64_t vbest = -1;
        for (int c = 0; c < params.target_samples; ++c) {
            const int cand = idx(rng);
            if (cand == target) continue;
            const int64_t v = std::llabs(a_row_j[cand]);
            if (v > vbest) { vbest = v; ibest = cand; }
        }
        if (ibest < 0) { do { ibest = idx(rng); } while (ibest == target); }
        pivot = ibest;
    };

    double t_init = params.t_init;
    if (t_init <= 0.0) {
        // Scale the starting temperature to a typical single entry (the score
        // sums ~ n*n entries) so the initial acceptance of small uphill moves is
        // reasonable.
        const double avg_entry = (double)cur.score / (double)((int64_t)n * n);
        t_init = std::max(1.0, 2.0 * avg_entry);
    }
    double T = t_init;

    const double wsum = params.add_weight + params.swap_weight + params.neg_weight;
    const double p_add = params.add_weight / wsum;
    const double p_swap_hi = (params.add_weight + params.swap_weight) / wsum;

    int64_t local_best = cur.score;
    int moves_since_improvement = 0;
    int cool_counter = 0;
    uint64_t moves = 0;

    while (!done_flag.load(std::memory_order_relaxed)) {
        ++moves;
        // ----- propose and apply a move -----
        const double u = unit(rng);
        bool score_decreased = false;

        if (u < p_add) {
            // ----- shear (the only score-changing move) -----
            int i, j;
            if (unit(rng) < params.target_fraction) {
                pick_pair_targeted(i, j);
            } else {
                i = idx(rng);
                do { j = idx(rng); } while (j == i);
            }
            const bool greedy = unit(rng) < params.greedy_fraction;
            const int64_t a_ii = cur.A.at(i, i);
            int64_t s_val;
            if (greedy && a_ii != 0) {
                s_val = -rounded_div(cur.A.at(j, i), a_ii);
                if (s_val > SHEAR_CAP) s_val = SHEAR_CAP;
                else if (s_val < -SHEAR_CAP) s_val = -SHEAR_CAP;
                if (s_val == 0) s_val = (rng() & 1u) ? 1 : -1;
            } else {
                s_val = (rng() & 1u) ? 1 : -1;
            }
            if (attempt_add(i, j, s_val, T) == 1) score_decreased = true;
        } else if (u < p_swap_hi) {
            // Swap and Neg only permute / sign-flip absolute values, so the score
            // and off-diagonal nonzero count are unchanged and X cannot blow up.
            Trans t{ TType::Swap, idx(rng), 0, 0 };
            do { t.j = idx(rng); } while (t.j == t.i);
            apply_to_Xt(cur.Xt, t);
            apply_to_A(cur.A, t);
            std::swap(row_mass[t.i], row_mass[t.j]);
        } else {
            Trans t{ TType::Neg, idx(rng), -1, 0 };
            apply_to_Xt(cur.Xt, t);
            apply_to_A(cur.A, t);
        }

        // Unified improvement bookkeeping for all move types.
        if (cur.score < local_best) {
            local_best = cur.score;
            moves_since_improvement = 0;
        } else {
            ++moves_since_improvement;
        }

        // ----- publish a new global best -----
        if (score_decreased && cur.score < best_score_atomic.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lk(best_mtx);
            if (cur.score < best_score_atomic.load(std::memory_order_relaxed)) {
                global_best = cur;
                best_score_atomic.store(cur.score, std::memory_order_relaxed);
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                const bool diagonal = (cur.off_nonzero == 0);
                if (show) {
                    // Throttle disk writes (a large matrix is expensive to serialise),
                    // but always save on the final diagonal. Saves only happen for a
                    // top-level run (!quiet). A --verbose inner slice prints but
                    // leaves the coordinator's files untouched.
                    const bool save = !params.quiet &&
                                      (diagonal || (elapsed - last_save_elapsed >= params.save_interval));
                    std::lock_guard<std::mutex> plk(print_mtx);
                    std::cout << "[t=" << elapsed << "s] new best score=" << cur.score
                              << " (thread " << thread_id << ")\n";
                    if (save) {
                        write_matrix_csv(transpose(cur.Xt), "best_X.csv");
                        write_matrix_csv(cur.A, "best_A.csv");
                        last_save_elapsed = elapsed;
                    }
                }
                if (diagonal) {
                    if (show) {
                        std::lock_guard<std::mutex> plk(print_mtx);
                        std::cout << "[t=" << elapsed << "s] matrix is diagonal, stopping (thread "
                                  << thread_id << ")\n";
                    }
                    done_flag.store(true, std::memory_order_relaxed);
                    break;
                }
            }
        }

        // ----- stuck: reheat, and reseed from the global best if far behind -----
        if (moves_since_improvement > params.stuck_threshold) {
            T = params.reheat * t_init;
            const int64_t gb = best_score_atomic.load(std::memory_order_relaxed);
            if ((double)cur.score > (double)gb * params.reseed_factor) {
                {
                    std::lock_guard<std::mutex> lk(best_mtx);
                    cur = global_best;
                }
                compute_row_mass(cur.A, row_mass);
            }
            local_best = cur.score;
            moves_since_improvement = 0;
            cool_counter = 0;
            continue;
        }

        // ----- cool -----
        if (++cool_counter >= params.moves_per_cool) {
            cool_counter = 0;
            T *= params.cooling;
            if (T < params.t_min) T = params.t_min;
        }
    }

    total_moves.fetch_add(moves, std::memory_order_relaxed);
}

// Gram matrix of a lattice whose rows are the m basis vectors in dimension d:
// A = L L^T (m x m), A(i,j) = <row_i, row_j>. Symmetric, and positive definite
// for a full-rank integer basis.
Matrix gram_of(const Lattice& L) {
    const int m = L.m, d = L.d;
    Matrix A((size_t)m);
    for (int i = 0; i < m; ++i) {
        const int64_t* ri = L.row(i);
        for (int j = i; j < m; ++j) {
            const int64_t* rj = L.row(j);
            int64_t s = 0;
            for (int k = 0; k < d; ++k) s += ri[k] * rj[k];
            A.at(i, j) = s;
            A.at(j, i) = s;
        }
    }
    return A;
}

// New basis L2 = X^T L (m x d). With the annealer producing D = X^T A X and
// A = L L^T, we have L2 L2^T = X^T A X = D, so L2 is a basis of the same
// lattice whose Gram matrix is exactly D.
Lattice xt_times(const Matrix& X, const Lattice& L) {
    const int m = L.m, d = L.d;
    Lattice R;
    R.m = m;
    R.d = d;
    R.data.assign((size_t)m * d, 0);
    for (int i = 0; i < m; ++i) {
        int64_t* ri = R.row(i);
        for (int p = 0; p < m; ++p) {
            const int64_t xpi = X.at(p, i);   // (X^T)_{i,p} = X_{p,i}
            if (xpi == 0) continue;
            const int64_t* lp = L.row(p);
            for (int k = 0; k < d; ++k) ri[k] += xpi * lp[k];
        }
    }
    return R;
}

// Run the worker pool on A (starting from the transform X_start) until a
// diagonal is found or params.max_seconds elapses, and return the best
// congruence found. Its A field holds D = X^T A X and Xt holds X^T.
Congruence run_annealer(const Matrix& A, const Matrix& X_start, const Params& params,
                        const std::vector<int>* active_in = nullptr) {
    Congruence global_best{ A, transpose(X_start), A.score(), A.count_offdiag_nonzero() };
    const int n = (int)A.n;
    if (n < 2) return global_best;

    // Indices the workers may move on. Defaults to the whole matrix. The deflation
    // coordinator passes a shrinking subset as pivots get locked in.
    std::vector<int> active;
    if (active_in) {
        active = *active_in;
    } else {
        active.resize((size_t)n);
        for (int i = 0; i < n; ++i) active[i] = i;
    }
    if (active.size() < 2) return global_best;

    std::mutex best_mtx;
    std::mutex print_mtx;
    std::atomic<bool> done_flag{ false };
    std::atomic<int64_t> best_score_atomic{ global_best.score };
    std::atomic<uint64_t> total_moves{ 0 };
    double last_save_elapsed = -1e18; // force a save on the first improvement

    const bool show = !params.quiet || params.verbose;
    const auto t0 = std::chrono::steady_clock::now();
    if (show) {
        std::cout << "[t=0s] start score=" << global_best.score << " n=" << n
                  << " workers=" << params.workers << "\n";
    }

    std::vector<std::thread> threads;
    std::random_device rd;
    for (int t = 0; t < params.workers; ++t) {
        uint32_t seed = static_cast<uint32_t>(rd()) ^ (static_cast<uint32_t>(t) * 0x9e3779b9u);
        threads.emplace_back([&, seed, t]() {
            anneal_worker(params, seed, global_best, best_score_atomic,
                          best_mtx, print_mtx, done_flag, total_moves,
                          t0, last_save_elapsed, t, active);
        });
    }

    // Optional wall-clock stop. Polls so it wakes promptly when a diagonal is found.
    std::thread timer;
    if (params.max_seconds > 0.0) {
        timer = std::thread([&]() {
            while (!done_flag.load(std::memory_order_relaxed)) {
                const double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                if (el >= params.max_seconds) {
                    done_flag.store(true, std::memory_order_relaxed);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }

    for (auto& th : threads) th.join();
    if (timer.joinable()) timer.join();

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const uint64_t moves = total_moves.load(std::memory_order_relaxed);
    if (show) {
        std::cout << "done moves=" << moves << " seconds=" << elapsed
                  << " moves_per_sec=" << (elapsed > 0.0 ? (double)moves / elapsed : 0.0) << "\n";
    }
    return global_best;
}

// a^e mod p by fast exponentiation. Requires 0 < p < 2^31 so every product of two
// residues fits in int64.
static int64_t powmod_small(int64_t a, int64_t e, int64_t p) {
    int64_t r = 1 % p;
    a %= p; if (a < 0) a += p;
    while (e > 0) {
        if (e & 1) r = (r * a) % p;
        a = (a * a) % p;
        e >>= 1;
    }
    return r;
}

// Determinant of A modulo a prime p (p < 2^31) by Gaussian elimination over Z/p.
// All intermediate products are below p^2 < 2^62, so plain int64 suffices.
static int64_t det_mod_prime(const Matrix& A, int64_t p) {
    const int n = (int)A.n;
    std::vector<int64_t> m((size_t)n * n);
    for (size_t k = 0; k < m.size(); ++k) {
        int64_t v = A.data[k] % p;
        m[k] = v < 0 ? v + p : v;
    }
    auto at = [&](int i, int j) -> int64_t& { return m[(size_t)i * n + j]; };
    int64_t det = 1 % p;
    for (int col = 0; col < n; ++col) {
        int piv = -1;
        for (int r = col; r < n; ++r) if (at(r, col) != 0) { piv = r; break; }
        if (piv < 0) return 0;                 // singular -> det = 0
        if (piv != col) {
            for (int j = 0; j < n; ++j) std::swap(at(piv, j), at(col, j));
            det = (p - det) % p;               // a row swap flips the sign
        }
        const int64_t pv = at(col, col);
        det = (det * pv) % p;
        const int64_t inv = powmod_small(pv, p - 2, p);   // Fermat inverse
        for (int r = col + 1; r < n; ++r) {
            const int64_t f = (at(r, col) * inv) % p;
            if (f == 0) continue;
            for (int j = col; j < n; ++j) {
                int64_t v = at(r, j) - (f * at(col, j)) % p;
                v %= p; if (v < 0) v += p;
                at(r, j) = v;
            }
        }
    }
    return det % p;
}

// Probabilistic exact test that det(A) == +/-1 (i.e. A is unimodular). It computes
// det(A) modulo a couple of large primes. A unimodular matrix always passes
// (+/-1 is +/-1 mod every p), and a non-unimodular one passes only if its
// determinant is congruent to +1 (or to -1) modulo every prime, which for
// distinct ~10^9 primes has negligible probability ~ (1/p)^k.
static bool is_unimodular(const Matrix& A) {
    static const int64_t primes[] = { 1000000007LL, 998244353LL };
    bool all_pos = true, all_neg = true;
    for (int64_t p : primes) {
        const int64_t d = det_mod_prime(A, p);
        if (d != 1) all_pos = false;
        if (d != p - 1) all_neg = false;
    }
    return all_pos || all_neg;
}

// Clear the off-diagonal entries of row/column i (whose pivot M_ii is +/-1) with
// exact integer shears, updating the working matrix M and the transposed transform
// Xt. Each shear s = -M_ji / M_ii is exact because the pivot is +/-1, and we reuse
// try_add/commit_add so every step is magnitude-checked. The shears do not interact
// through column i (an Add on row j leaves M_{j',i} untouched for j' != j), so the
// quotients can be read straight off the current column. Returns false and leaves
// M, Xt unchanged if any shear would push an entry past MAGNITUDE_LIMIT. The caller
// guarantees |M_ii| == 1 and a small pivot row (see DEFLATE_OFFDIAG_CAP).
static bool deflate_index(Matrix& M, Matrix& Xt, int i, const std::vector<int>& active,
                          std::vector<int64_t>& new_a_row, std::vector<int64_t>& new_xt_row,
                          std::vector<int64_t>& scratch_mass) {
    const int64_t a_ii = M.at(i, i);
    const Matrix M0 = M;     // snapshot for rollback on an out-of-range shear
    const Matrix Xt0 = Xt;
    int64_t delta = 0, dnz = 0;
    for (int j : active) {
        if (j == i) continue;
        const int64_t a_ji = M.at(j, i);
        if (a_ji == 0) continue;
        const int64_t s = (a_ii == 1) ? -a_ji : a_ji;   // s = -a_ji / a_ii, exact for a_ii = +/-1
        if (!try_add(M, Xt, i, j, s, new_a_row, new_xt_row, delta, dnz)) {
            M = M0;
            Xt = Xt0;
            return false;
        }
        commit_add(M, Xt, j, new_a_row, new_xt_row, scratch_mass);
    }
    return true;
}

// Deflation solver. Repeatedly anneal the active sub-problem for a short slice,
// then lock in pivots and drop them from the active set, so the working problem
// shrinks monotonically (a frozen coordinate is never disturbed again). Two
// locking rules:
//
//   strict (relaxed == false): lock a pivot M_ii == +/-1 by clearing its whole
//     row/column with exact integer shears. This is the classical splitting that
//     drives a unimodular form to a +/-1 diagonal. The caller must pass a
//     unimodular A (every full pivot of such a form is forced to +/-1).
//
//   relaxed (relaxed == true): lock any coordinate that has become orthogonal to
//     all other active vectors (its off-diagonal row is already zero), splitting
//     off a <c> summand for any norm c. This needs no shears (so it never
//     overflows) and works on a general, non-unimodular Gram matrix. It can only
//     peel off orthogonal summands, so it helps a reducible lattice and simply
//     never fires on an irreducible one. It cannot manufacture +/-1 pivots.
//
// Progress is reported and the best result is written to disk periodically
// (throttled by save_interval) so a long run is observable and leaves artifacts.
// Always starts from X = I.
Congruence solve_with_deflation(const Matrix& A, const Params& params, bool relaxed) {
    const int n = (int)A.n;
    Matrix M = A;
    Matrix Xt(A.n);
    Xt.fill_identity();                          // Xt = X^T, X = I, so M = X^T A X
    std::vector<int> active((size_t)n);
    for (int i = 0; i < n; ++i) active[i] = i;

    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    if (!params.quiet) {
        std::cout << "[deflate] start mode=" << (relaxed ? "blocks" : "strict")
                  << " score=" << M.score() << " n=" << n
                  << " slice=" << params.deflate_slice << "s workers=" << params.workers << "\n";
    }

    Params sp = params;
    sp.quiet = true;                             // silence per-slice worker output / disk writes
    sp.deflate = false;
    sp.deflate_blocks = false;

    std::vector<int64_t> new_a_row((size_t)n), new_xt_row((size_t)n), scratch_mass((size_t)n, 0);
    const double report_every = std::max(params.save_interval, params.deflate_slice);
    double last_report = -1e18;

    while (active.size() >= 2) {
        const double el = elapsed();
        if (params.max_seconds > 0.0) {
            if (el >= params.max_seconds) break;
            sp.max_seconds = std::min(params.deflate_slice, params.max_seconds - el);
        } else {
            sp.max_seconds = params.deflate_slice;
        }

        // Anneal the current working matrix over the active set. Passing M as the
        // working matrix and X = transpose(Xt) as the starting transform makes
        // run_annealer return the composed (M, Xt) directly.
        const Matrix X = transpose(Xt);
        const Congruence gb = run_annealer(M, X, sp, &active);
        M = gb.A;
        Xt = gb.Xt;

        // Lock in pivots. Deflating one index changes the others, so each
        // candidate is re-tested against the current M, and only indices still in
        // 'active' are touched.
        int locked = 0;
        const std::vector<int> snapshot = active;
        for (int i : snapshot) {
            const int64_t a_ii = M.at(i, i);
            if (relaxed) {
                if (a_ii == 0) continue;
                bool orthogonal = true;          // off-diagonal row already zero?
                for (int j : active) {
                    if (j == i) continue;
                    if (M.at(j, i) != 0) { orthogonal = false; break; }
                }
                if (!orthogonal) continue;
                active.erase(std::find(active.begin(), active.end(), i));
                ++locked;
            } else {
                if (std::llabs(a_ii) != 1) continue;
                int64_t max_off = 0;             // bounds |s| = |M_ji| for a +/-1 pivot
                for (int j : active) {
                    if (j == i) continue;
                    const int64_t v = std::llabs(M.at(j, i));
                    if (v > max_off) max_off = v;
                }
                if (max_off > DEFLATE_OFFDIAG_CAP) continue;
                if (deflate_index(M, Xt, i, active, new_a_row, new_xt_row, scratch_mass)) {
                    active.erase(std::find(active.begin(), active.end(), i));
                    ++locked;
                }
            }
        }

        const bool diagonal = (M.count_offdiag_nonzero() == 0);
        const double now = elapsed();
        if (!params.quiet && (locked > 0 || diagonal || now - last_report >= report_every)) {
            std::cout << "[deflate] t=" << now << "s score=" << M.score()
                      << " active=" << active.size()
                      << " offdiag=" << M.count_offdiag_nonzero();
            if (locked > 0) std::cout << " (locked " << locked << ")";
            std::cout << "\n";
            write_matrix_csv(transpose(Xt), "best_X.csv");
            write_matrix_csv(M, "best_A.csv");
            last_report = now;
        }

        if (diagonal) break;
    }

    if (!params.quiet) {
        std::cout << "[deflate] done active=" << active.size() << " score=" << M.score()
                  << " seconds=" << elapsed() << "\n";
    }
    return Congruence{ M, Xt, M.score(), M.count_offdiag_nonzero() };
}

int main(int argc, char** argv) {
    CLI::App app{ "xtax cpu congruence annealer" };

    std::string a_csv;
    std::string l_csv;
    std::string x_csv;
    Params params;
    params.workers = std::max(1, (int)std::thread::hardware_concurrency());

    app.add_option("-A", a_csv, "CSV file for A (n x n integers)");
    app.add_option("-L,--lattice", l_csv, "CSV file for a lattice basis (rows are vectors). Anneals the Gram A = L*L^T");
    app.add_option("-X", x_csv, "Path to initial X matrix (A mode). If none, start from identity");
    app.add_option("-w,--workers", params.workers, "Number of worker threads");
    app.add_option("--stuck-threshold", params.stuck_threshold, "Moves without improvement before reheating");
    app.add_option("--t-init", params.t_init, "Initial SA temperature (<= 0 auto-calibrates)");
    app.add_option("--t-min", params.t_min, "Minimum SA temperature");
    app.add_option("--cooling", params.cooling, "Geometric cooling factor per cooling step (0 < alpha < 1)");
    app.add_option("--moves-per-cool", params.moves_per_cool, "Moves between cooling steps");
    app.add_option("--reheat", params.reheat, "Fraction of initial temperature restored when stuck");
    app.add_option("--greedy-fraction", params.greedy_fraction, "Probability an Add move uses the reducing quotient shear");
    app.add_option("--add-weight", params.add_weight, "Relative weight of Add (shear) moves");
    app.add_option("--swap-weight", params.swap_weight, "Relative weight of Swap moves");
    app.add_option("--neg-weight", params.neg_weight, "Relative weight of Neg moves");
    app.add_option("--target-fraction", params.target_fraction, "Probability an Add targets a hot row (0 = uniform random)");
    app.add_option("--target-samples", params.target_samples, "Tournament size for hot-row / large-pivot selection");
    app.add_option("--reseed-factor", params.reseed_factor, "Reseed from global best when stuck and score exceeds best by this factor");
    app.add_option("--max-seconds", params.max_seconds, "Wall-clock stop in seconds (<= 0 runs until a diagonal is found)");
    app.add_option("--save-interval", params.save_interval, "Minimum seconds between best_*.csv disk writes");
    app.add_flag("--deflate", params.deflate, "Strict deflation: lock +/-1 pivots and shrink the active problem (requires a unimodular matrix, starts from identity)");
    app.add_flag("--deflate-blocks", params.deflate_blocks, "Relaxed deflation: peel off orthogonal summands (works on any Gram, starts from identity)");
    app.add_option("--deflate-slice", params.deflate_slice, "Deflation: annealing seconds per slice before checking for pivots");
    app.add_flag("--verbose", params.verbose, "Also show the inner annealer's progress inside --deflate / --deflate-blocks (does not write per-slice CSVs)");
    CLI11_PARSE(app, argc, argv);

    const bool have_A = !a_csv.empty();
    const bool have_L = !l_csv.empty();
    if (have_A == have_L) {
        std::cerr << "Error: provide exactly one of -A or -L\n";
        return 1;
    }
    if (params.deflate && params.deflate_blocks) {
        std::cerr << "Error: use at most one of --deflate or --deflate-blocks\n";
        return 1;
    }

    if (have_A) {
        // ----- A mode: diagonalize a given symmetric matrix -----
        Matrix A;
        Matrix X;
        try {
            A = read_matrix_csv(a_csv);
            if (!x_csv.empty()) {
                X = read_matrix_csv(x_csv);
                if (X.n != A.n) throw std::runtime_error("X dimensions do not match A");
            } else {
                X = Matrix(A.n);
                X.fill_identity();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }

        const int n = (int)A.n;
        if (params.deflate) {
            std::cout << "[deflate] verifying the matrix is unimodular (det = +/-1)...\n";
            if (!is_unimodular(A)) {
                std::cerr << "Error: --deflate requires a unimodular matrix (det = +/-1); this one is not.\n"
                          << "       Use --deflate-blocks for a relaxed (orthogonal-summand) deflation,\n"
                          << "       or drop --deflate to run the plain annealer.\n";
                return 1;
            }
        }
        const bool any_deflate = params.deflate || params.deflate_blocks;
        Congruence global_best = any_deflate
            ? solve_with_deflation(A, params, params.deflate_blocks)
            : run_annealer(A, X, params);
        std::cout << "Final best score: " << global_best.score << "\n";
        const Matrix best_X = transpose(global_best.Xt);
        if (n <= 20) {
            std::cout << "A:\n";
            global_best.A.print();
            std::cout << "----\nX:\n";
            best_X.print();
        }
        write_matrix_csv(best_X, "best_X.csv");
        write_matrix_csv(global_best.A, "best_A.csv");
        return 0;
    }

    // ----- L mode: build the Gram matrix A = L*L^T and anneal it -----
    Lattice curL;
    try {
        curL = read_lattice_csv(l_csv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    const int m = curL.m, d = curL.d;
    if (m < 2) {
        std::cerr << "Error: lattice must have at least 2 basis vectors\n";
        return 1;
    }
    if (m > d) {
        std::cerr << "Warning: lattice has more vectors (" << m << ") than dimensions ("
                  << d << "); the basis is linearly dependent and the Gram matrix is singular\n";
    }

    Matrix A = gram_of(curL);
    std::cout << "[lattice] m=" << m << " d=" << d << " score=" << A.score() << "\n";

    // Anneal the Gram matrix exactly as in A mode.
    Matrix I((size_t)m);
    I.fill_identity();
    if (params.deflate) {
        std::cout << "[deflate] verifying the Gram matrix is unimodular (det = +/-1)...\n";
        if (!is_unimodular(A)) {
            std::cerr << "Error: --deflate requires a unimodular matrix (det = +/-1); this Gram is not\n"
                      << "       (a lattice Gram is unimodular only when the lattice is unimodular).\n"
                      << "       Use --deflate-blocks for a relaxed (orthogonal-summand) deflation,\n"
                      << "       or drop --deflate to run the plain annealer.\n";
            return 1;
        }
    }

    const bool any_deflate = params.deflate || params.deflate_blocks;
    Congruence gb = any_deflate
        ? solve_with_deflation(A, params, params.deflate_blocks)
        : run_annealer(A, I, params);
    const Matrix X = transpose(gb.Xt);
    std::cout << "Final best score: " << gb.score << "\n";

    // Final basis of the same lattice: L_final = X^T * L, Gram = gb.A.
    Lattice Lfinal = xt_times(X, curL);
    const Matrix g2 = gram_of(Lfinal);
    if (g2.n != gb.A.n || g2.data != gb.A.data) {
        std::cout << "[warn] Gram(X^T L) != annealed Gram (transform mismatch, possible int64 overflow)\n";
    }
    write_lattice_csv(Lfinal, "final_L.csv");
    write_matrix_csv(gb.A, "best_A.csv");
    write_matrix_csv(X, "best_X.csv");
    if (m <= 20 && d <= 20) {
        std::cout << "Final L:\n";
        for (int i = 0; i < m; ++i) {
            const int64_t* ri = Lfinal.row(i);
            for (int k = 0; k < d; ++k) { std::cout << ri[k]; if (k + 1 < d) std::cout << ','; }
            std::cout << '\n';
        }
        std::cout << "----\nGram:\n";
        gb.A.print();
    }
    return 0;
}
