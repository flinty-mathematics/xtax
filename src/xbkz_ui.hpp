#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

// Live worker phase for the Win32 dashboard (and internal status atomics).
enum class WorkerPhase : int {
    starting = 0,
    init_lll = 1,
    warmup = 2,
    tour = 3,
    reseed = 4,
    stopped = 5,
    sieving = 6,
};

inline const char* worker_phase_name(WorkerPhase p) {
    switch (p) {
    case WorkerPhase::starting: return "Starting";
    case WorkerPhase::init_lll: return "Init LLL";
    case WorkerPhase::warmup: return "Warmup";
    case WorkerPhase::tour: return "Tour";
    case WorkerPhase::reseed: return "Reseed";
    case WorkerPhase::stopped: return "Stopped";
    case WorkerPhase::sieving: return "Sieving";
    }
    return "?";
}

struct WorkerStatus {
    std::atomic<int> phase{ (int)WorkerPhase::starting };
    std::atomic<int> tour{ 0 };
    std::atomic<int> target_beta{ 0 };
    std::atomic<int> cur_beta{ 0 };
    std::atomic<int> block_idx{ 0 };   // 1-based index in the current tour
    std::atomic<int> block_total{ 0 };
    std::atomic<int> block_kappa{ 0 };
    std::atomic<int> block_h{ 0 };
    std::atomic<long long> sieve_step{ 0 };  // cumulative sieve attempts this pass
    std::atomic<long long> sieve_total{ 0 }; // total sieve budget (cap) for the pass
    std::atomic<long long> enum_nodes{ 0 };
    std::atomic<int> tours_changed{ 0 };
    std::atomic<int> blocks_hit{ 0 };
    std::atomic<double> local_b0{ 0.0 };
};

// Shared best basis. b0_relaxed mirrors b0 so promote can reject without locking.
// B and U are flat row-major buffers (B is n*d, U is n*n) so promotion and
// reseeding copy them with a single contiguous memcpy rather than row by row.
struct GlobalBest {
    std::mutex mtx;
    std::vector<int64_t> B;
    std::vector<int64_t> U;
    int n = 0;
    int d = 0;
    bool u_valid = true;
    bool track_u = true;
    double b0 = 0;
    double pot = 0;
    bool has = false;
    std::atomic<bool> has_relaxed{ false };
    std::atomic<double> b0_relaxed{ 0.0 };
    std::atomic<long long> improvements{ 0 };
};

struct BkzUiConfig {
    int threads = 1;
    int block_start = 2;
    int block = 20;
    int lattice_m = 0;
    int lattice_d = 0;
    double init_short_norm2 = 0.0;
    double max_seconds = 0.0;
};

using SteadyClock = std::chrono::steady_clock;
using SteadyTimePoint = SteadyClock::time_point;

#if defined(_WIN32)

// Runs the Win32 message loop until the user closes the window, Ctrl-C trips
// stop_flag, or finished() returns true. Sets stop_flag on window close.
void xbkz_ui_run(const BkzUiConfig& cfg, SteadyTimePoint start,
                 std::vector<WorkerStatus>& workers, const GlobalBest& best,
                 std::atomic<bool>& stop_flag,
                 const std::function<bool()>& finished);

#else

inline void xbkz_ui_run(const BkzUiConfig&, SteadyTimePoint,
                        std::vector<WorkerStatus>&, const GlobalBest&,
                        std::atomic<bool>&, const std::function<bool()>&) {}

#endif
