// xdual: simultaneous primal/dual congruence annealer.
//
// This is the xtax congruence annealer with a different objective. xtax drives a
// single working Gram P = X^T G X toward a diagonal by minimizing an L1 sparsity
// score. xdual instead anneals the lattice and its dual at the same time. Under
// the same unimodular column move on X it keeps:
//
//   P = X^T G X     (primal working Gram, exact integer, starts at G)
//   Q = P^{-1}      (true dual lattice Gram, double precision, starts at G^{-1})
//
// and minimizes a squared-Frobenius off-diagonal score
//
//   F(X) = ||offdiag(P)||_F^2 + c * ||offdiag(Q)||_F^2,
//
// so a low score means both the basis and its dual basis are close to
// orthogonal. The move P -> E^T P E (E unimodular) sends P^{-1} -> E^{-1} P^{-1}
// E^{-T}, which is the same O(n) shear machinery applied to Q with the
// pivot/target swapped and the sign of s flipped, so the dual is maintained with
// no per-move inversion.
//
// The dual is floating-point on purpose: it is only a search-guidance penalty,
// not part of the exact output (the basis X^T L is recovered from the exact
// integer P/X). Q is computed once by a numeric inverse and re-inverted from the
// exact P periodically, and additionally whenever a sampled residual of P*Q - I
// exceeds a tolerance, to bound floating-point drift.
//
// The shared machinery lives in the common headers: mat_io.hpp (Matrix, Matrixd,
// Lattice, CSV IO, invert_to), stop_signal.hpp (Ctrl-C / stop flags), and
// congruence_anneal.hpp (the templated simulated-annealing engine). This file
// provides the primal/dual objective policy and the command-line plumbing.
// Deflation is intentionally not offered here (it is xtax-specific).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "mat_io.hpp"
#include "stop_signal.hpp"
#include "congruence_anneal.hpp"

// row_mass[r] = sum over k != r of |P_rk|. Used to bias shear targets toward the
// rows carrying the most primal off-diagonal mass.
static void compute_row_mass(const Matrix& P, std::vector<int64_t>& row_mass) {
    const int n = (int)P.n;
    for (int r = 0; r < n; ++r) {
        const int64_t* row = P.data.data() + (size_t)r * n;
        int64_t m = 0;
        for (int k = 0; k < n; ++k) if (k != r) m += std::llabs(row[k]);
        row_mass[r] = m;
    }
}

// Evaluate a single congruence shear Add(pivot, target, s) on a symmetric matrix
// M without mutating it: row/column 'target' of M would gain s times row/column
// 'pivot'. Returns false (infeasible) if any affected entry would exceed
// MAGNITUDE_LIMIT. On success new_row holds the proposed row 'target' of M (with
// new_row[target] the new diagonal), d_off_frob2 is the change in the squared
// Frobenius off-diagonal norm, and d_nonzero is the change in the number of
// nonzero off-diagonal pairs. Both off-diagonal mirror entries (target,k) and
// (k,target) change identically, so the squared-norm delta carries a factor 2.
static bool eval_add(const Matrix& M, int pivot, int target, int64_t s,
                     std::vector<int64_t>& new_row,
                     long double& d_off_frob2, int64_t& d_nonzero) {
    const int n = (int)M.n;
    const int64_t* row_t = M.data.data() + (size_t)target * n;
    const int64_t* row_p = M.data.data() + (size_t)pivot * n;

    long double d = 0.0L;
    int64_t dnz = 0;
    for (int k = 0; k < n; ++k) {
        if (k == target) continue;
        const int64_t old = row_t[k];
        const int64_t nv = old + s * row_p[k];
        if (nv > MAGNITUDE_LIMIT || nv < -MAGNITUDE_LIMIT) return false;
        new_row[k] = nv;
        d += (long double)nv * (long double)nv - (long double)old * (long double)old;
        if (old == 0) { if (nv != 0) ++dnz; }
        else if (nv == 0) --dnz;
    }

    const int64_t a_pp = M.at(pivot, pivot);
    const int64_t a_tp = M.at(target, pivot);
    const int64_t old_tt = row_t[target];
    const int64_t new_tt = old_tt + 2 * s * a_tp + s * s * a_pp;
    if (new_tt > MAGNITUDE_LIMIT || new_tt < -MAGNITUDE_LIMIT) return false;
    new_row[target] = new_tt;

    d_off_frob2 = 2.0L * d;   // symmetric pair (target,k) and (k,target)
    d_nonzero = dnz;
    return true;
}

// Commit the precomputed row from eval_add into M (row/col 'target'). When
// row_mass is non-null it is kept up to date (used only for the primal P).
static void commit_add(Matrix& M, int target, const std::vector<int64_t>& new_row,
                       std::vector<int64_t>* row_mass) {
    const int n = (int)M.n;
    int64_t* row_t = M.data.data() + (size_t)target * n;
    int64_t mass_t = 0;
    for (int k = 0; k < n; ++k) {
        if (k == target) continue;
        const int64_t old = row_t[k];
        const int64_t nv = new_row[k];
        row_t[k] = nv;             // contiguous row write
        M.at(k, target) = nv;      // strided symmetric mirror
        const int64_t na = std::llabs(nv);
        if (row_mass) {
            (*row_mass)[k] += na - std::llabs(old);
            mass_t += na;
        }
    }
    row_t[target] = new_row[target];
    if (row_mass) (*row_mass)[target] = mass_t;
}

// Floating-point analogue of eval_add for the dual Q = P^{-1}. No magnitude
// limit applies (the dual is not the exact output), so this only computes the
// proposed row 'target' and the squared-Frobenius off-diagonal delta.
static void eval_add_d(const Matrixd& M, int pivot, int target, int64_t s,
                       std::vector<double>& new_row, long double& d_off_frob2) {
    const int n = (int)M.n;
    const double* row_t = M.data.data() + (size_t)target * n;
    const double* row_p = M.data.data() + (size_t)pivot * n;
    const double sf = (double)s;

    long double d = 0.0L;
    for (int k = 0; k < n; ++k) {
        if (k == target) continue;
        const double old = row_t[k];
        const double nv = old + sf * row_p[k];
        new_row[k] = nv;
        d += (long double)nv * (long double)nv - (long double)old * (long double)old;
    }
    const double a_pp = M.at(pivot, pivot);
    const double a_tp = M.at(target, pivot);
    const double old_tt = row_t[target];
    new_row[target] = old_tt + 2.0 * sf * a_tp + sf * sf * a_pp;

    d_off_frob2 = 2.0L * d;   // symmetric pair (target,k) and (k,target)
}

static void commit_add_d(Matrixd& M, int target, const std::vector<double>& new_row) {
    const int n = (int)M.n;
    double* row_t = M.data.data() + (size_t)target * n;
    for (int k = 0; k < n; ++k) {
        if (k == target) continue;
        row_t[k] = new_row[k];        // contiguous row write
        M.at(k, target) = new_row[k]; // strided symmetric mirror
    }
    row_t[target] = new_row[target];
}

// Combined score F = ||offdiag(P)||_F^2 + c * ||offdiag(Q)||_F^2.
static long double combined_score(const Matrix& P, const Matrixd& Q, long double c) {
    return P.offdiag_frob2() + c * Q.offdiag_frob2();
}

// The primal/dual objective policy for the shared annealing engine. It owns the
// primal working Gram P = X^T G X (exact integer), the dual Q = P^{-1} (double),
// the accumulated transform stored transposed (Xt = X^T), the running combined
// score, the primal off-diagonal nonzero count, and the per-row primal mass used
// to bias shear targets. See congruence_anneal.hpp for the engine contract.
struct DualObjective {
    using score_t = long double;

    Matrix P;
    Matrixd Q;
    Matrix Xt;
    long double score_ = 0.0L;
    int64_t p_off_nonzero_ = 0;
    long double c = 1.0L;          // dual weight (already normalized)
    std::vector<int64_t> row_mass;
    std::vector<int64_t> new_p_row;
    std::vector<double> new_q_row;
    std::vector<int64_t> new_xt_row;
    int dual_refresh = 200000;     // unconditional re-inversion period (0 = never)
    int dual_check = 25000;        // sampled residual check period (0 = never)
    double dual_tol = 1e-6;        // residual tolerance that triggers a refresh
    // Adaptive dual weight: ramp c linearly from 0 to its full value over the
    // first lambda_ramp seconds, so the early search optimizes the primal
    // freely and the dual pressure phases in. 0 disables the ramp (default).
    double lambda_ramp = 0.0;
    long double c_full = 1.0L;     // the fully ramped dual weight
    std::chrono::steady_clock::time_point ramp_t0 = std::chrono::steady_clock::now();
    bool ramp_done = true;

    DualObjective() = default;
    DualObjective(const Matrix& P0, const Matrixd& Q0, const Matrix& X_start,
                  long double c_norm)
        : P(P0), Q(Q0), Xt(transpose(X_start)),
          score_(combined_score(P0, Q0, c_norm)),
          p_off_nonzero_(P0.count_offdiag_nonzero()), c(c_norm),
          row_mass(P0.n), new_p_row(P0.n), new_q_row(P0.n), new_xt_row(P0.n) {}

    int n() const { return (int)P.n; }
    score_t score() const { return score_; }
    int64_t offdiag_nonzero() const { return p_off_nonzero_; }
    // A diagonal primal forces a diagonal dual too (the inverse of a diagonal
    // matrix is diagonal), so p_off == 0 is the joint solved condition.
    bool solved() const { return p_off_nonzero_ == 0; }
    int64_t row_weight(int r) const { return row_mass[(size_t)r]; }
    int64_t pivot_abs(int t, int cnd) const { return std::llabs(P.at(t, cnd)); }

    double suggest_t_init() const {
        // Scale the starting temperature to a typical squared off-diagonal entry
        // (the score sums ~ n*n such entries) so the initial acceptance of small
        // uphill moves is reasonable.
        const int nn = (int)P.n;
        const double avg = (double)(score_ / (long double)((int64_t)nn * nn));
        return std::max(1.0, 2.0 * avg);
    }

    // The primal P sees the shear directly (row/col j gains s times row/col i).
    // The dual Q = P^{-1} transforms by the inverse-transpose elementary, which
    // is the same shear with pivot/target swapped and s negated.
    bool evaluate(int i, int j, int64_t s, score_t& d_score, int64_t& d_nonzero) {
        long double d_off_p = 0.0L, d_off_q = 0.0L;
        int64_t dnz_p = 0;
        if (!eval_add(P, i, j, s, new_p_row, d_off_p, dnz_p)) return false;
        eval_add_d(Q, j, i, -s, new_q_row, d_off_q);
        d_score = d_off_p + c * d_off_q;
        d_nonzero = dnz_p;
        return true;
    }

    bool commit(int i, int j, int64_t s, score_t d_score, int64_t d_nonzero) {
        // Build the transform row (row j of Xt gains s times row i), magnitude-
        // checking each entry. An overflow here is astronomically rare. If it
        // fires, skip the move and leave the whole state untouched.
        const int nn = (int)P.n;
        const int64_t* xt_row_i = Xt.data.data() + (size_t)i * nn;
        const int64_t* xt_row_j = Xt.data.data() + (size_t)j * nn;
        for (int r = 0; r < nn; ++r) {
            const int64_t v = xt_row_j[r] + s * xt_row_i[r];
            if (v > MAGNITUDE_LIMIT || v < -MAGNITUDE_LIMIT) return false;
            new_xt_row[r] = v;
        }
        commit_add(P, j, new_p_row, &row_mass);
        commit_add_d(Q, i, new_q_row);
        int64_t* xt_j = Xt.data.data() + (size_t)j * nn;
        for (int r = 0; r < nn; ++r) xt_j[r] = new_xt_row[r];
        score_ += d_score;
        p_off_nonzero_ += d_nonzero;
        return true;
    }

    // Exact best integer shear for Add(pivot=i, target=j). Both score parts are
    // exactly quadratic in s (only off-diagonal entries enter, and they change
    // linearly): with sums over the affected off-diagonal entries,
    //   f(s) = 2 [ (a2 + c*c2) s^2 + 2 (b2 - c*d2) s ],
    //   a2 = sum_{k != j} P_ik^2,  b2 = sum_{k != j} P_jk P_ik   (primal),
    //   c2 = sum_{k != i} Q_jk^2,  d2 = sum_{k != i} Q_ik Q_jk   (dual, shear -s).
    // The real minimizer is s* = -(b2 - c*d2) / (a2 + c*c2). The best integer is
    // whichever of floor(s*), ceil(s*) has smaller f, clamped to SHEAR_CAP.
    // Returns 0 when no nonzero shear strictly helps.
    int64_t best_shear(int i, int j) const {
        const int nn = (int)P.n;
        const int64_t* p_row_i = P.data.data() + (size_t)i * nn;
        const int64_t* p_row_j = P.data.data() + (size_t)j * nn;
        long double a2 = 0.0L, b2 = 0.0L;
        for (int k = 0; k < nn; ++k) {
            if (k == j) continue;
            const long double pik = (long double)p_row_i[k];
            a2 += pik * pik;
            b2 += (long double)p_row_j[k] * pik;
        }
        const double* q_row_i = Q.data.data() + (size_t)i * nn;
        const double* q_row_j = Q.data.data() + (size_t)j * nn;
        long double c2 = 0.0L, d2 = 0.0L;
        for (int k = 0; k < nn; ++k) {
            if (k == i) continue;
            const long double qjk = (long double)q_row_j[k];
            c2 += qjk * qjk;
            d2 += (long double)q_row_i[k] * qjk;
        }
        const long double quad = a2 + c * c2;
        if (!(quad > 0.0L)) return 0;
        const long double lin = b2 - c * d2;
        const long double s_star = -lin / quad;
        auto f = [&](long double s) { return 2.0L * (quad * s * s + 2.0L * lin * s); };
        int64_t lo = (int64_t)std::floor((double)s_star);
        int64_t hi = lo + 1;
        lo = std::clamp(lo, -SHEAR_CAP, SHEAR_CAP);
        hi = std::clamp(hi, -SHEAR_CAP, SHEAR_CAP);
        int64_t best_s = 0;
        long double best_f = 0.0L;   // f(0) = 0, require a strict improvement
        for (int64_t s : { lo, hi }) {
            if (s == 0) continue;
            const long double v = f((long double)s);
            if (v < best_f) { best_f = v; best_s = s; }
        }
        return best_s;
    }

    void refresh_cache() { compute_row_mass(P, row_mass); }

    // Bound floating-point drift in the dual. Two triggers: an unconditional
    // re-inversion every dual_refresh moves, and a cheap sampled residual check
    // every dual_check moves that re-inverts early when ||P Q e_k - e_k||_inf
    // exceeds dual_tol for a sampled column k. P stays full rank (it is a
    // unimodular congruence of a full-rank Gram), so the inversion never fails.
    void periodic_maintenance(uint64_t moves) {
        // Dual weight ramp: rescale c from the shared wall clock. All workers
        // ramp on the same schedule, so their scores stay comparable. Each
        // change of c re-bases the running score.
        if (!ramp_done && (moves & 0x1FFF) == 0) {
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - ramp_t0).count();
            const double frac = el / lambda_ramp;
            if (frac >= 1.0) {
                c = c_full;
                ramp_done = true;
            } else {
                c = c_full * (long double)frac;
            }
            score_ = combined_score(P, Q, c);
        }

        bool refresh = dual_refresh > 0 && (moves % (uint64_t)dual_refresh) == 0;
        if (!refresh && dual_check > 0 && (moves % (uint64_t)dual_check) == 0) {
            const int nn = (int)P.n;
            const int k = (int)(moves % (uint64_t)nn);   // rotating sample column
            long double worst = 0.0L;
            for (int r = 0; r < nn; ++r) {
                const int64_t* p_row = P.data.data() + (size_t)r * nn;
                long double acc = 0.0L;
                for (int t = 0; t < nn; ++t)
                    acc += (long double)p_row[t] * (long double)Q.at(t, k);
                if (r == k) acc -= 1.0L;
                const long double a = acc < 0.0L ? -acc : acc;
                if (a > worst) worst = a;
            }
            refresh = (double)worst > dual_tol;
        }
        if (refresh && invert_to(P, Q)) score_ = combined_score(P, Q, c);
    }

    score_t recompute_score() {
        score_ = combined_score(P, Q, c);
        return score_;
    }

    void reorder_for_publish(bool) {}

    void publish_files() const {
        write_matrix_csv(transpose(Xt), "best_X.csv");
        write_matrix_csv(P, "best_P.csv");
        write_matrixd_csv(Q, "best_Q.csv");
    }

    std::string best_line() const {
        std::ostringstream ss;
        ss << "score=" << (double)score_
           << " primal=" << P.count_offdiag_nonzero()
           << " dual=" << std::sqrt((double)Q.offdiag_frob2());
        return ss.str();
    }
};

struct Params {
    canneal::EngineParams engine;
    double lambda = 1.0;           // dual weight (1.0 = equal initial primal/dual weight)
    int dual_refresh = 200000;     // moves between unconditional dual re-inversions (0 = never)
    int dual_check = 25000;        // moves between sampled dual residual checks (0 = never)
    double dual_tol = 1e-6;        // sampled residual that triggers an early re-inversion
    double lambda_ramp = 0.0;      // seconds to ramp the dual weight from 0 to full (0 = off)
};

// Build the initial primal/dual state (inverting the Gram once) and run the
// shared engine on it until the primal is diagonal or the budget elapses.
// rows, when non-null, restricts the engine's moves to those indices.
static DualObjective run_xdual(const Matrix& P_init, const Matrix& X_start,
                               const Params& params,
                               const std::vector<int>* rows) {
    Matrixd Q_init;
    std::cout << "[xdual] inverting the Gram (double-precision dual P^-1)...\n";
    if (!invert_to(P_init, Q_init)) {
        throw std::runtime_error(
            "could not invert the Gram (matrix singular / basis not full rank).");
    }

    const long double s_p0 = P_init.offdiag_frob2();
    const long double s_q0 = Q_init.offdiag_frob2();
    // Normalize so lambda = 1 gives the dual the same initial weight as the
    // primal, regardless of the very different scales of P and P^{-1}.
    long double c = (long double)params.lambda;
    if (s_p0 > 0.0L && s_q0 > 0.0L && std::isfinite((double)s_q0)) c *= s_p0 / s_q0;

    DualObjective start(P_init, Q_init, X_start, c);
    start.dual_refresh = params.dual_refresh;
    start.dual_check = params.dual_check;
    start.dual_tol = params.dual_tol;
    if (params.lambda_ramp > 0.0) {
        start.lambda_ramp = params.lambda_ramp;
        start.c_full = c;
        start.c = 0.0L;
        start.ramp_t0 = std::chrono::steady_clock::now();
        start.ramp_done = false;
        start.recompute_score();
    }

    std::cout << "[xdual] lambda=" << params.lambda
              << " c=" << (double)c
              << " score=" << (double)start.score()
              << " primal=" << start.offdiag_nonzero()
              << " dual=" << std::sqrt((double)s_q0) << "\n";

    DualObjective best = canneal::run_annealer(start, params.engine, rows);
    if (params.lambda_ramp > 0.0) {
        // The global best may have been recorded while the dual weight was
        // still ramping. Re-base its score at the full weight so the final
        // report is comparable across configurations.
        best.c = c;
        best.ramp_done = true;
        best.recompute_score();
    }
    return best;
}

int main(int argc, char** argv) {
    install_signal_handlers();

    CLI::App app{ "xdual simultaneous primal/dual congruence annealer" };

    std::string a_csv;
    std::string l_csv;
    std::string x_csv;
    std::string gram_rows_spec;  // --gram-rows: restrict moves to these rows
    uint64_t modulus = 0;   // 0 = off; 1 = saturation; > 1 = balanced mod
    Params params;

    const int physical = canneal::physical_core_count();
    const int logical = std::max(1, (int)std::thread::hardware_concurrency());
    params.engine.threads = physical;
    bool use_hyperthreads = false;
    bool no_pin = false;
    CLI::Option* threads_opt = nullptr;

    app.add_option("-A", a_csv, "CSV file for a symmetric matrix A (n x n integers), used as the primal working Gram");
    app.add_option("-L,--lattice", l_csv, "CSV file for a lattice basis (rows are vectors). Anneals the Gram G = L*L^T and its dual");
    app.add_option("-X", x_csv, "Path to initial X matrix (A mode). If none, start from identity");
    app.add_option("--modulus", modulus,
                   "Reduce the loaded input entries (A or L) modulo this value once at load, "
                   "to balanced residues in (-m/2, m/2]. A modulus of 1 selects saturation "
                   "instead: nonzero entries become 1, 0 stays 0")
        ->check(CLI::Range((uint64_t)1, (uint64_t)INT64_MAX));
    app.add_option("--gram-rows", gram_rows_spec,
                   "Restrict the annealer's moves to these 0-based row indices of the "
                   "working matrix: a comma-separated list of indices and inclusive "
                   "lo..hi ranges, e.g. \"0,3..6,9\". Rows outside the set are never "
                   "used as pivot or target. Default: all rows");
    app.add_option("--lambda", params.lambda, "Dual weight in F = offdiag(P)^2 + c*offdiag(Q)^2, where 1.0 gives equal initial primal/dual weight");
    threads_opt = app.add_option("-t,--threads", params.engine.threads, "Number of worker threads (default: physical core count)");
    app.add_flag("--use-hyperthreads", use_hyperthreads, "Default the worker count to all logical processors instead of physical cores (ignored if --threads is given)");
    app.add_flag("--no-pin", no_pin, "Do not pin worker threads to physical cores (Windows, on by default)");
    app.add_option("--seed", params.engine.seed, "Base RNG seed for reproducibility (0 = seed from random_device)");
    app.add_option("--stuck-threshold", params.engine.stuck_threshold, "Moves without improvement before reheating");
    app.add_option("--t-init", params.engine.t_init, "Initial SA temperature (<= 0 auto-calibrates)");
    app.add_option("--t-min", params.engine.t_min, "Minimum SA temperature");
    app.add_option("--cooling", params.engine.cooling, "Geometric cooling factor per cooling step (0 < alpha < 1)");
    app.add_option("--moves-per-cool", params.engine.moves_per_cool, "Moves between cooling steps");
    app.add_option("--reheat", params.engine.reheat, "Fraction of initial temperature restored when stuck");
    app.add_option("--greedy-fraction", params.engine.greedy_fraction, "Probability an Add move uses the exact best-integer shear");
    app.add_option("--target-fraction", params.engine.target_fraction, "Probability an Add targets a hot row (0 = uniform random)");
    app.add_option("--target-samples", params.engine.target_samples, "Tournament size for hot-row / large-pivot selection");
    app.add_option("--reseed-factor", params.engine.reseed_factor, "Reseed from global best when stuck and score exceeds best by this factor");
    app.add_option("--sweep-fraction", params.engine.sweep_fraction, "Probability of a greedy reduction sweep when a worker stalls (0 = off)");
    app.add_flag("--tempering,!--no-tempering", params.engine.tempering, "Parallel-tempering temperature ladder with replica exchange, on by default with 2+ threads. --no-tempering restores per-worker cooling with reheat/reseed");
    app.add_option("--exchange-interval", params.engine.exchange_interval, "Moves between replica-exchange sweeps (tempering mode)");
    app.add_flag("--adaptive-cooling", params.engine.adaptive_cooling, "Nudge the cooling rate toward the target acceptance ratio");
    app.add_option("--worker-diversity", params.engine.worker_diversity, "Spread of per-worker greedy/target offsets (0 = identical workers)");
    app.add_option("--max-seconds", params.engine.max_seconds, "Wall-clock stop in seconds (<= 0 runs until interrupted or fully diagonal)");
    app.add_option("--save-interval", params.engine.save_interval, "Minimum seconds between best_*.csv disk writes");
    app.add_option("--dual-refresh", params.dual_refresh, "Moves between unconditional exact re-inversions of the dual (0 = never)");
    app.add_option("--dual-check", params.dual_check, "Moves between sampled dual residual checks that can trigger an early re-inversion (0 = never)");
    app.add_option("--dual-tol", params.dual_tol, "Sampled residual of P*Q - I that triggers an early dual re-inversion");
    app.add_option("--lambda-ramp", params.lambda_ramp, "Ramp the dual weight linearly from 0 to its full value over this many seconds (0 = full weight from the start)");
    CLI11_PARSE(app, argc, argv);

    if (threads_opt->count() == 0 && use_hyperthreads) params.engine.threads = logical;
    if (params.engine.threads < 1) params.engine.threads = 1;
    params.engine.pin_threads = !no_pin;

    const bool have_A = !a_csv.empty();
    const bool have_L = !l_csv.empty();
    if (have_A == have_L) {
        std::cerr << "Error: provide exactly one of -A or -L\n";
        return 1;
    }

    if (have_A) {
        // ----- A mode: anneal a given symmetric matrix and its dual -----
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

        if (modulus > 0) {
            reduce_entries_mod(A.data, modulus);
            std::cout << "[modulus] " << modulus_note(modulus) << "\n";
        }

        const int n = (int)A.n;
        std::vector<int> rows;
        if (!gram_rows_spec.empty()) {
            try {
                rows = parse_index_spec(gram_rows_spec, n, "--gram-rows");
                if (rows.size() < 2)
                    throw std::runtime_error(
                        "--gram-rows: need at least 2 rows (a move uses a "
                        "pivot and a target)");
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
            }
            std::cout << "[gram-rows] restricting moves to " << rows.size()
                      << " of " << n << " rows\n";
        }
        // With a supplied initial transform the working Gram starts at
        // X^T A X, so the search continues where the previous run stopped and
        // the output invariant X^T A X == best_P holds against the input A.
        const Matrix A_work = x_csv.empty() ? A : congruence_of(A, X);
        DualObjective best;
        try {
            best = run_xdual(A_work, X, params, rows.empty() ? nullptr : &rows);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
        std::cout << "Final best score: " << (double)best.score() << "\n";
        const Matrix best_X = transpose(best.Xt);
        if (n <= 20) {
            std::cout << "P:\n";
            best.P.print();
            std::cout << "----\nQ:\n";
            best.Q.print();
            std::cout << "----\nX:\n";
            best_X.print();
        }
        if (g_interrupted.load(std::memory_order_relaxed))
            std::cout << "[interrupted] writing best_X.csv, best_P.csv and best_Q.csv before exit\n";
        write_matrix_csv(best_X, "best_X.csv");
        write_matrix_csv(best.P, "best_P.csv");
        write_matrixd_csv(best.Q, "best_Q.csv");
        return 0;
    }

    // ----- L mode: build the Gram G = L*L^T and anneal it with its dual -----
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

    if (modulus > 0) {
        reduce_entries_mod(curL.data, modulus);
        std::cout << "[modulus] " << modulus_note(modulus) << "\n";
        for (int i = 0; i < m; ++i) {
            const int64_t* ri = curL.row(i);
            bool zero = true;
            for (int k = 0; k < d; ++k) if (ri[k] != 0) { zero = false; break; }
            if (zero) {
                std::cerr << "Warning: row " << i << " became zero under the "
                          << "modulus; the Gram matrix is singular and the dual "
                          << "inversion will fail\n";
            }
        }
    }

    Matrix G = gram_of(curL);
    std::cout << "[lattice] m=" << m << " d=" << d << "\n";

    std::vector<int> rows;
    if (!gram_rows_spec.empty()) {
        try {
            rows = parse_index_spec(gram_rows_spec, m, "--gram-rows");
            if (rows.size() < 2)
                throw std::runtime_error(
                    "--gram-rows: need at least 2 rows (a move uses a pivot "
                    "and a target)");
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
        std::cout << "[gram-rows] restricting moves to " << rows.size()
                  << " of " << m << " rows\n";
    }

    Matrix I((size_t)m);
    I.fill_identity();
    DualObjective gb;
    try {
        gb = run_xdual(G, I, params, rows.empty() ? nullptr : &rows);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    const Matrix X = transpose(gb.Xt);
    std::cout << "Final best score: " << (double)gb.score() << "\n";

    // Final basis of the same lattice: L_final = X^T * L, Gram = gb.P.
    Lattice Lfinal = xt_times(X, curL);
    const Matrix g2 = gram_of(Lfinal);
    if (g2.n != gb.P.n || g2.data != gb.P.data) {
        std::cout << "[warn] Gram(X^T L) != annealed P (transform mismatch, possible int64 overflow)\n";
    }
    if (g_interrupted.load(std::memory_order_relaxed))
        std::cout << "[interrupted] writing final_L.csv, best_P.csv and best_X.csv before exit\n";
    write_lattice_csv(Lfinal, "final_L.csv");
    write_matrix_csv(gb.P, "best_P.csv");
    write_matrixd_csv(gb.Q, "best_Q.csv");
    write_matrix_csv(X, "best_X.csv");
    if (m <= 20 && d <= 20) {
        std::cout << "Final L:\n";
        for (int i = 0; i < m; ++i) {
            const int64_t* ri = Lfinal.row(i);
            for (int k = 0; k < d; ++k) { std::cout << ri[k]; if (k + 1 < d) std::cout << ','; }
            std::cout << '\n';
        }
        std::cout << "----\nPrimal Gram P:\n";
        gb.P.print();
    }
    return 0;
}
