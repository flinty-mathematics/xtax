// congruence_anneal.hpp: one templated simulated-annealing engine over
// unimodular congruences X^T A X, shared by xtax (L1 sparsity objective) and
// xdual (primal/dual Frobenius objective).
//
// The engine owns everything that is objective-independent: the worker threads
// and their affinity, the temperature schedule (per-worker geometric cooling
// with reheat/reseed, or a parallel-tempering ladder with replica exchange), the
// move loop (targeted (i,j) selection, exact-shear greedy vs random +/-1
// exploration, the Metropolis test), the greedy reduction-sweep plateau breaker,
// the global-best publish with throttled disk writes, and stop / time-budget
// handling.
//
// Everything objective-specific lives in an Objective policy passed as the
// template parameter Obj. One Obj instance holds a full working state (the
// matrices, the transform, the running score and caches). The engine keeps one
// Obj per worker plus the shared global best, and drives them through the
// members documented in the "Objective policy contract" below. Templating keeps
// the hot path monomorphized: no virtual calls.
//
// Objective policy contract (all methods are called single-threaded on a
// worker's own Obj unless noted):
//
//   using score_t;                         // int64_t (xtax) or long double (xdual)
//   int n() const;                         // matrix dimension
//   score_t score() const;                 // current running score
//   int64_t offdiag_nonzero() const;       // primal off-diagonal nonzero pairs
//   bool solved() const;                   // primal diagonal reached
//   int64_t row_weight(int r) const;       // targeting bias (off-diagonal mass)
//   int64_t pivot_abs(int t, int c) const; // |A[t][c]|, for the pivot tournament
//   double suggest_t_init() const;         // auto initial temperature
//   // Evaluate Add(pivot=i, target=j, s) without mutating the state. Fill the
//   // internal scratch, return feasibility (magnitude ok), set d_score and
//   // d_nonzero (change in off-diagonal nonzero pairs).
//   bool evaluate(int i, int j, int64_t s, score_t& d_score, int64_t& d_nonzero);
//   // Commit the move just evaluated (same i,j,s). Returns false and leaves the
//   // state unchanged if a transform entry would overflow. Updates the running
//   // score and nonzero count with the passed deltas.
//   bool commit(int i, int j, int64_t s, score_t d_score, int64_t d_nonzero);
//   // Optimal integer shear s for (i,j), clamped to [-SHEAR_CAP, SHEAR_CAP].
//   // Returns 0 when no shear strictly helps (the engine then explores +/-1).
//   int64_t best_shear(int i, int j) const;
//   void refresh_cache();                  // rebuild caches after a reseed
//   void periodic_maintenance(uint64_t moves); // xdual dual refresh, else no-op
//   score_t recompute_score();             // exact score from scratch (drift fix)
//   void reorder_for_publish(bool enabled);// xtax rcm/centroid, else no-op
//   void publish_files() const;            // write best_*.csv
//   std::string best_line() const;         // detail for the "new best" log line

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "stop_signal.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace canneal {

#if defined(_WIN32)
// Affinity masks for the physical cores in processor group 0 (one mask per core,
// covering that core's logical processors). The annealer is FP / memory-bandwidth
// bound, so pinning each worker to its own physical core avoids two workers
// contending for one core's shared execution units via hyperthreading. Cores in
// other processor groups (systems with > 64 logical processors) are skipped.
// pinning is then simply left off, which is safe. Returns an empty vector if the
// topology cannot be queried.
inline std::vector<DWORD_PTR> physical_core_masks() {
    std::vector<DWORD_PTR> masks;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return masks;
    std::vector<char> buf((size_t)len);
    auto* first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &len))
        return masks;
    char* ptr = buf.data();
    char* end = buf.data() + len;
    while (ptr < end) {
        auto* cur = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
        if (cur->Relationship == RelationProcessorCore &&
            cur->Processor.GroupCount >= 1 &&
            cur->Processor.GroupMask[0].Group == 0) {
            masks.push_back((DWORD_PTR)cur->Processor.GroupMask[0].Mask);
        }
        ptr += cur->Size;
    }
    return masks;
}
#endif

// Number of physical cores in group 0 (Windows) or the logical processor count
// elsewhere. Used as the default worker count.
inline int physical_core_count() {
#if defined(_WIN32)
    const int lp = std::max(1, (int)std::thread::hardware_concurrency());
    const auto masks = physical_core_masks();
    return masks.empty() ? lp : (int)masks.size();
#else
    return std::max(1, (int)std::thread::hardware_concurrency());
#endif
}

// Engine parameters shared by both tools. Tool-specific knobs (dual weight,
// deflation, etc.) live in each tool's own Params and are not part of this.
struct EngineParams {
    int threads = 1;
    double t_init = 0.0;           // initial temperature (<= 0 auto-calibrates)
    double t_min = 1e-3;           // floor temperature (and ladder bottom)
    double cooling = 0.999;        // geometric cooling factor per cooling step
    int moves_per_cool = 200;      // moves between cooling steps
    double reheat = 1.0;           // fraction of t_init restored when stuck
    int stuck_threshold = 20000;   // moves without improvement before reheating
    double reseed_factor = 1.25;   // reseed from best when stuck and this far behind
    double greedy_fraction = 0.5;  // probability a shear uses the exact-best s
    double target_fraction = 0.5;  // probability a shear targets a hot row
    int target_samples = 8;        // tournament size for hot-row / large-pivot pick
    double max_seconds = 0.0;      // wall-clock stop (<= 0 runs until solved)
    double save_interval = 2.0;    // minimum seconds between best_*.csv writes

    // Parallel tempering is the default multi-worker mode: it beat the
    // cooling + reheat + reseed scheme on every small-input benchmark (see
    // README). It needs at least 2 workers for a ladder. run_annealer falls
    // back to the cooling schedule automatically for a single worker.
    bool tempering = true;         // parallel-tempering ladder + replica exchange
    int exchange_interval = 2000;  // moves between replica-exchange sweeps (tempering)
    bool adaptive_cooling = false; // nudge the cooling rate toward target acceptance
    double target_accept = 0.44;   // acceptance-ratio target for adaptive cooling
    double worker_diversity = 0.4; // per-worker greedy/target offset spread (0 = off)
    double sweep_fraction = 0.0;   // probability of a reduction-sweep on a stall

    bool pin_threads = true;       // pin each worker to a physical core (Windows)
    bool use_hyperthreads = false; // default worker count to logical processors
    uint64_t seed = 0;             // base RNG seed (workers derive from it)
    bool quiet = false;            // silence per-worker console / disk output
    bool verbose = false;          // show inner progress even when quiet
};

// Shared coordination state for one run. Templated on the Objective so the
// global best carries the full working state.
template <typename Obj>
struct Shared {
    std::mutex best_mtx;
    std::mutex print_mtx;
    Obj global_best;
    std::atomic<bool> done_flag{ false };
    std::atomic<double> best_score_relaxed{ 0.0 };  // cheap lock-free reject hint
    std::atomic<uint64_t> total_moves{ 0 };
    double last_save_elapsed = -1e18;                // force a save on first improve

    // Parallel-tempering ladder: temps[] are the fixed rung temperatures (sorted
    // ascending). rung_of_worker[k] is the rung worker k currently occupies, and
    // worker_score[k] is its last-published score. The coordinator swaps rungs
    // between adjacent workers by the replica-exchange rule.
    std::vector<double> temps;
    std::vector<std::atomic<int>> rung_of_worker;
    std::vector<std::atomic<double>> worker_score;
};

// One independent simulated-annealing worker.
template <typename Obj>
void anneal_worker(const EngineParams& params, uint32_t seed,
                   Shared<Obj>& sh, int thread_id, const std::vector<int>& active,
                   std::chrono::steady_clock::time_point t0) {
    using score_t = typename Obj::score_t;

    Obj cur;
    {
        std::lock_guard<std::mutex> lk(sh.best_mtx);
        cur = sh.global_best;
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> aidx(0, (int)active.size() - 1);
    auto idx = [&](std::mt19937& r) -> int { return active[(size_t)aidx(r)]; };
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    cur.refresh_cache();

    const bool show = !params.quiet || params.verbose;

    // Per-worker diversity: spread the greedy / target fractions around the
    // configured values so the pool covers a range of exploit/explore balances.
    double greedy_fraction = params.greedy_fraction;
    double target_fraction = params.target_fraction;
    if (params.worker_diversity > 0.0 && params.threads > 1) {
        const double off = params.worker_diversity *
            ((double)thread_id / (double)(params.threads - 1) - 0.5);
        greedy_fraction = std::clamp(greedy_fraction + off, 0.0, 1.0);
        target_fraction = std::clamp(target_fraction + off, 0.0, 1.0);
    }

    // Pick a target row by a hot-row tournament, then a pivot as the largest
    // off-diagonal entry of that row (another tournament). Stays stochastic.
    auto pick_pair_targeted = [&](int& pivot, int& target) {
        int jbest = idx(rng);
        for (int c = 1; c < params.target_samples; ++c) {
            const int cand = idx(rng);
            if (cur.row_weight(cand) > cur.row_weight(jbest)) jbest = cand;
        }
        target = jbest;
        int ibest = -1;
        int64_t vbest = -1;
        for (int c = 0; c < params.target_samples; ++c) {
            const int cand = idx(rng);
            if (cand == target) continue;
            const int64_t v = cur.pivot_abs(target, cand);
            if (v > vbest) { vbest = v; ibest = cand; }
        }
        if (ibest < 0) { do { ibest = idx(rng); } while (ibest == target); }
        pivot = ibest;
    };

    // Attempt Add(i,j,s) at temperature Tcur. Returns -1 rejected/infeasible,
    // 0 accepted without lowering the score, 1 accepted and score decreased.
    auto attempt_add = [&](int i, int j, int64_t s, double Tcur) -> int {
        score_t delta = score_t(0);
        int64_t dnz = 0;
        if (!cur.evaluate(i, j, s, delta, dnz)) return -1;
        if (!((double)delta <= 0.0 || unit(rng) < std::exp(-(double)delta / Tcur)))
            return -1;
        if (!cur.commit(i, j, s, delta, dnz)) return -1;
        return (double)delta < 0.0 ? 1 : 0;
    };

    // A greedy reduction sweep from a pivot row: apply the exact best shear of
    // every other active row against the pivot. Each such shear is non-increasing
    // in the objective, so this is a cheap descent that collapses a lot of mass at
    // once and breaks plateaus. Returns true if the score strictly decreased.
    auto reduction_sweep = [&](int pivot) -> bool {
        const score_t before = cur.score();
        for (int j : active) {
            if (j == pivot) continue;
            if (sh.done_flag.load(std::memory_order_relaxed)) break;
            const int64_t s = cur.best_shear(pivot, j);
            if (s == 0) continue;
            score_t delta = score_t(0);
            int64_t dnz = 0;
            if (!cur.evaluate(pivot, j, s, delta, dnz)) continue;
            if ((double)delta > 0.0) continue;   // never uphill in a sweep
            cur.commit(pivot, j, s, delta, dnz);
        }
        return cur.score() < before;
    };

    // Temperature setup. In tempering mode the worker reads its (moving) rung
    // temperature each step. Otherwise it runs its own geometric cooling.
    double t_init = params.t_init > 0.0 ? params.t_init : cur.suggest_t_init();
    double T = t_init;

    score_t local_best = cur.score();
    int moves_since_improvement = 0;
    int cool_counter = 0;
    uint64_t moves = 0;

    // Adaptive-cooling acceptance tracking over the current cooling window.
    int win_proposed = 0, win_accepted = 0;
    double cooling = params.cooling;

    while (!sh.done_flag.load(std::memory_order_relaxed)) {
        ++moves;
        if (params.tempering)
            T = sh.temps[(size_t)sh.rung_of_worker[(size_t)thread_id]
                             .load(std::memory_order_relaxed)];

        int i, j;
        if (unit(rng) < target_fraction) {
            pick_pair_targeted(i, j);
        } else {
            i = idx(rng);
            do { j = idx(rng); } while (j == i);
        }

        int64_t s_val;
        if (unit(rng) < greedy_fraction) {
            s_val = cur.best_shear(i, j);
            if (s_val == 0) s_val = (rng() & 1u) ? 1 : -1;   // fall back to explore
        } else {
            s_val = (rng() & 1u) ? 1 : -1;
        }

        const int res = attempt_add(i, j, s_val, T);
        ++win_proposed;
        if (res >= 0) ++win_accepted;

        if (cur.score() < local_best) {
            local_best = cur.score();
            moves_since_improvement = 0;
        } else {
            ++moves_since_improvement;
        }

        // ----- publish a new global best -----
        if (res == 1 &&
            (double)cur.score() < sh.best_score_relaxed.load(std::memory_order_relaxed)) {
            // The running score can drift over many incremental updates, so recompute
            // it exactly before comparing and publishing.
            cur.recompute_score();
            if ((double)cur.score() < sh.best_score_relaxed.load(std::memory_order_relaxed)) {
                // Reorder (score-preserving) and deep-copy off the critical section,
                // then swap into the shared best under the lock (O(1) held time).
                Obj scratch = cur;
                if (!params.quiet) scratch.reorder_for_publish(true);
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                std::lock_guard<std::mutex> lk(sh.best_mtx);
                if ((double)cur.score() < sh.best_score_relaxed.load(std::memory_order_relaxed)) {
                    std::swap(sh.global_best, scratch);
                    sh.best_score_relaxed.store((double)sh.global_best.score(),
                                                std::memory_order_relaxed);
                    const bool solved = sh.global_best.solved();
                    if (show) {
                        const bool save = !params.quiet &&
                            (solved || elapsed - sh.last_save_elapsed >= params.save_interval);
                        std::lock_guard<std::mutex> plk(sh.print_mtx);
                        std::cout << "[t=" << elapsed << "s] new best "
                                  << sh.global_best.best_line()
                                  << " (thread " << thread_id << ")\n";
                        if (save) {
                            sh.global_best.publish_files();
                            sh.last_save_elapsed = elapsed;
                        }
                    }
                    if (solved) {
                        if (show) {
                            std::lock_guard<std::mutex> plk(sh.print_mtx);
                            std::cout << "[t=" << elapsed
                                      << "s] matrix is diagonal, stopping (thread "
                                      << thread_id << ")\n";
                        }
                        sh.done_flag.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }
        }

        // Publish this worker's score for the replica-exchange coordinator.
        if (params.tempering && (moves & 0xFF) == 0)
            sh.worker_score[(size_t)thread_id].store((double)cur.score(),
                                                     std::memory_order_relaxed);

        // Objective-specific upkeep (e.g. xdual's periodic dual re-inversion to
        // bound floating-point drift). A no-op for a purely integer objective.
        cur.periodic_maintenance(moves);

        if (params.tempering) {
            // In tempering mode temperatures are managed by the ladder, so a
            // stall only triggers the greedy reduction-sweep plateau breaker.
            if (moves_since_improvement > params.stuck_threshold) {
                if (params.sweep_fraction > 0.0 && unit(rng) < params.sweep_fraction)
                    reduction_sweep(idx(rng));
                local_best = cur.score();
                moves_since_improvement = 0;
            }
        } else {
            // ----- stuck: optional sweep, reheat, reseed if far behind -----
            if (moves_since_improvement > params.stuck_threshold) {
                if (params.sweep_fraction > 0.0 && unit(rng) < params.sweep_fraction) {
                    if (reduction_sweep(idx(rng))) {
                        local_best = cur.score();
                        moves_since_improvement = 0;
                    }
                }
                T = params.reheat * t_init;
                const double gb = sh.best_score_relaxed.load(std::memory_order_relaxed);
                if ((double)cur.score() > gb * params.reseed_factor) {
                    {
                        std::lock_guard<std::mutex> lk(sh.best_mtx);
                        cur = sh.global_best;
                    }
                    cur.refresh_cache();
                }
                local_best = cur.score();
                moves_since_improvement = 0;
                cool_counter = 0;
                continue;
            }

            // ----- cool -----
            if (++cool_counter >= params.moves_per_cool) {
                cool_counter = 0;
                if (params.adaptive_cooling && win_proposed > 0) {
                    const double acc = (double)win_accepted / (double)win_proposed;
                    // Too many accepts: cool faster (smaller factor). Too few:
                    // cool slower. Nudge gently and clamp to a sane band.
                    if (acc > params.target_accept) cooling *= 0.999;
                    else cooling /= 0.999;
                    cooling = std::clamp(cooling, 0.90, 0.99999);
                }
                win_proposed = 0;
                win_accepted = 0;
                T *= cooling;
                if (T < params.t_min) T = params.t_min;
            }
        }
    }

    sh.total_moves.fetch_add(moves, std::memory_order_relaxed);
}

// Run the worker pool starting from `start` (its A/Xt already set up by the
// caller). Returns the best Obj found. `active` restricts which indices workers
// may move on (the whole matrix by default, deflation passes a subset).
template <typename Obj>
Obj run_annealer(Obj start, const EngineParams& params_in,
                 const std::vector<int>* active_in = nullptr) {
    // A tempering ladder needs at least two rungs. A single worker runs the
    // plain cooling schedule instead.
    EngineParams params = params_in;
    if (params.threads < 2) params.tempering = false;

    const int n = start.n();

    Shared<Obj> sh;
    sh.global_best = start;
    sh.best_score_relaxed.store((double)start.score(), std::memory_order_relaxed);

    if (n < 2) return sh.global_best;

    std::vector<int> active;
    if (active_in) {
        active = *active_in;
    } else {
        active.resize((size_t)n);
        for (int i = 0; i < n; ++i) active[i] = i;
    }
    if (active.size() < 2) return sh.global_best;

    const bool show = !params.quiet || params.verbose;
    const auto t0 = std::chrono::steady_clock::now();
    if (show) {
        std::cout << "[t=0s] start " << sh.global_best.best_line() << " n=" << n
                  << " threads=" << params.threads
                  << (params.tempering ? " (parallel tempering)" : "") << "\n";
    }

    // Build the tempering ladder: a geometric spread from t_min to t_init.
    if (params.tempering) {
        const int w = params.threads;
        sh.temps.resize((size_t)w);
        const double top = params.t_init > 0.0 ? params.t_init
                                               : sh.global_best.suggest_t_init();
        const double bot = std::min(params.t_min, top);
        if (w == 1) {
            sh.temps[0] = top;
        } else {
            const double ratio = std::pow(top / std::max(bot, 1e-12),
                                          1.0 / (double)(w - 1));
            double t = bot;
            for (int k = 0; k < w; ++k) { sh.temps[(size_t)k] = t; t *= ratio; }
        }
        sh.rung_of_worker = std::vector<std::atomic<int>>((size_t)w);
        sh.worker_score = std::vector<std::atomic<double>>((size_t)w);
        for (int k = 0; k < w; ++k) {
            sh.rung_of_worker[(size_t)k].store(k, std::memory_order_relaxed);
            sh.worker_score[(size_t)k].store((double)start.score(),
                                             std::memory_order_relaxed);
        }
    }

#if defined(_WIN32)
    std::vector<DWORD_PTR> core_masks =
        params.pin_threads ? physical_core_masks() : std::vector<DWORD_PTR>{};
#endif

    std::vector<std::thread> threads;
    std::mt19937 seed_rng(params.seed ? (uint32_t)params.seed
                                      : (uint32_t)std::random_device{}());
    for (int t = 0; t < params.threads; ++t) {
        uint32_t seed = seed_rng() ^ (static_cast<uint32_t>(t) * 0x9e3779b9u);
        threads.emplace_back([&, seed, t]() {
#if defined(_WIN32)
            if (!core_masks.empty())
                SetThreadAffinityMask(GetCurrentThread(),
                                      core_masks[(size_t)t % core_masks.size()]);
#endif
            anneal_worker<Obj>(params, seed, sh, t, active, t0);
        });
    }

    // Monitor / coordinator: enforces the wall-clock limit and Ctrl-C, and (in
    // tempering mode) periodically proposes replica exchanges between adjacent
    // rungs. Runs on its own thread so it also wakes promptly once a worker sets
    // done_flag on finding a diagonal.
    std::thread monitor([&]() {
        std::mt19937 xrng(0xC0FFEEu ^ (uint32_t)params.seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        auto last_exchange = std::chrono::steady_clock::now();
        while (!sh.done_flag.load(std::memory_order_relaxed)) {
            if (g_interrupted.load(std::memory_order_relaxed) ||
                g_stop.load(std::memory_order_relaxed)) {
                sh.done_flag.store(true, std::memory_order_relaxed);
                break;
            }
            if (params.max_seconds > 0.0) {
                const double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                if (el >= params.max_seconds) {
                    sh.done_flag.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            if (params.tempering) {
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double, std::milli>(now - last_exchange)
                        .count() >= 5.0) {
                    last_exchange = now;
                    // owner[r] = worker currently on rung r.
                    const int w = params.threads;
                    std::vector<int> owner((size_t)w);
                    for (int k = 0; k < w; ++k)
                        owner[(size_t)sh.rung_of_worker[(size_t)k]
                                  .load(std::memory_order_relaxed)] = k;
                    // Sweep adjacent rungs, proposing config exchanges (swap of
                    // rung ownership). Accept by min(1, exp((b_r - b_{r+1})(S_r - S_{r+1}))).
                    for (int r = 0; r + 1 < w; ++r) {
                        const int wa = owner[(size_t)r];
                        const int wb = owner[(size_t)r + 1];
                        const double Ta = sh.temps[(size_t)r];
                        const double Tb = sh.temps[(size_t)r + 1];
                        const double Sa = sh.worker_score[(size_t)wa]
                                              .load(std::memory_order_relaxed);
                        const double Sb = sh.worker_score[(size_t)wb]
                                              .load(std::memory_order_relaxed);
                        const double arg = (1.0 / Ta - 1.0 / Tb) * (Sa - Sb);
                        if (arg >= 0.0 || u(xrng) < std::exp(arg)) {
                            sh.rung_of_worker[(size_t)wa].store(r + 1,
                                std::memory_order_relaxed);
                            sh.rung_of_worker[(size_t)wb].store(r,
                                std::memory_order_relaxed);
                            std::swap(owner[(size_t)r], owner[(size_t)r + 1]);
                        }
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    for (auto& th : threads) th.join();
    monitor.join();

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const uint64_t moves = sh.total_moves.load(std::memory_order_relaxed);
    if (show) {
        std::cout << "done moves=" << moves << " seconds=" << elapsed
                  << " moves_per_sec=" << (elapsed > 0.0 ? (double)moves / elapsed : 0.0)
                  << "\n";
    }
    return sh.global_best;
}

}  // namespace canneal
