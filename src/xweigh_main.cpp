// xweigh: construct unrestricted weighing matrices W(n,w).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "CLI11.hpp"
#include "xweigh_anneal.hpp"

namespace {

bool is_square(int value) {
    const int root = (int)std::sqrt((double)value);
    return root * root == value || (root + 1) * (root + 1) == value;
}

bool is_sum_of_two_squares(int value) {
    for (int a = 0; (int64_t)a * a <= value; ++a) {
        const int remainder = value - a * a;
        if (is_square(remainder)) return true;
    }
    return false;
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

std::string impossibility_reason(int order, int weight) {
    if (weight == order && order > 2 && order % 4 != 0) {
        return "a full weighing matrix is a Hadamard matrix, whose order must "
               "be 1, 2, or a multiple of 4";
    }
    if ((order & 1) != 0) {
        if (!is_square(weight))
            return "odd-order weighing matrices require a square weight";
        if (order > 1) {
            const int root = (int)std::sqrt((double)weight);
            if (order < weight + root + 1) {
                return "odd-order weighing matrices require n >= w + sqrt(w) + 1";
            }
        }
    }
    if (order % 4 == 2 && !is_sum_of_two_squares(weight)) {
        return "orders congruent to 2 modulo 4 require the weight to be a "
               "sum of two integer squares";
    }
    return {};
}

bool valid_params(const xweigh::Params& params, std::string& error) {
    if (params.threads < 1) error = "--threads must be positive";
    else if (params.max_seconds < 0.0) error = "--max-seconds must be nonnegative";
    else if (params.save_interval < 0.0) error = "--save-interval must be nonnegative";
    else if (params.sign_fraction < 0.0 || params.sign_fraction > 1.0)
        error = "--sign-fraction must be in [0, 1]";
    else if (params.greedy_fraction < 0.0 || params.greedy_fraction > 1.0)
        error = "--greedy-fraction must be in [0, 1]";
    else if (params.target_fraction < 0.0 || params.target_fraction > 1.0)
        error = "--target-fraction must be in [0, 1]";
    else if (params.candidate_samples < 1)
        error = "--candidate-samples must be positive";
    else if (params.target_samples < 1)
        error = "--target-samples must be positive";
    else if (params.exchange_interval < 1)
        error = "--exchange-interval must be positive";
    else if (params.t_init < 0.0) error = "--t-init must be nonnegative";
    else if (params.t_min <= 0.0) error = "--t-min must be positive";
    else if (params.cooling <= 0.0 || params.cooling >= 1.0)
        error = "--cooling must be in (0, 1)";
    else if (params.moves_per_cool < 1)
        error = "--moves-per-cool must be positive";
    else if (params.stuck_threshold < 1)
        error = "--stuck-threshold must be positive";
    else if (params.reheat <= 0.0) error = "--reheat must be positive";
    else if (params.reseed_factor < 1.0)
        error = "--reseed-factor must be at least 1";
    return error.empty();
}

}  // namespace

int main(int argc, char** argv) {
    install_signal_handlers();

    CLI::App app{ "xweigh unrestricted weighing-matrix annealer" };
    int order = 0;
    int weight = 0;
    xweigh::Params params;
    std::string start_file;
    params.threads = xweigh::physical_core_count();
    const int logical =
        std::max(1, (int)std::thread::hardware_concurrency());
    CLI::Option* threads_option = nullptr;

    app.add_option("n", order, "Matrix order n")
        ->required()
        ->check(CLI::Range(1, xweigh::MAX_ORDER));
    app.add_option("w", weight, "Matrix weight w")
        ->required()
        ->check(CLI::Range(1, xweigh::MAX_ORDER));
    app.add_option("-o,--out", params.output,
                   "Output CSV for the best matrix");
    app.add_option("--start", start_file,
                   "Start from a complete fixed-weight ternary CSV")
        ->check(CLI::ExistingFile);
    threads_option = app.add_option(
        "-t,--threads", params.threads,
        "Number of worker threads (default: physical cores)");
    app.add_flag(
        "--use-hyperthreads", params.use_hyperthreads,
        "Default to all logical processors (ignored with --threads)");
    bool no_pin = false;
    app.add_flag("--no-pin", no_pin,
                 "Do not pin workers to physical cores on Windows");
    app.add_option("--seed", params.seed,
                   "Base RNG seed (0 = random_device)");
    app.add_option("--max-seconds", params.max_seconds,
                   "Wall-clock search budget (0 = until solved or interrupted)");
    app.add_option("--save-interval", params.save_interval,
                   "Minimum seconds between best-matrix writes");

    app.add_option("--sign-fraction", params.sign_fraction,
                   "Probability of proposing a sign flip instead of a support switch");
    app.add_option("--greedy-fraction", params.greedy_fraction,
                   "Probability of selecting the best of sampled candidate moves");
    app.add_option("--candidate-samples", params.candidate_samples,
                   "Candidate count for a sampled greedy move");
    app.add_option("--target-fraction", params.target_fraction,
                   "Probability of targeting a high-residual row");
    app.add_option("--target-samples", params.target_samples,
                   "Tournament size for high-residual row selection");

    app.add_flag(
        "--tempering,!--no-tempering", params.tempering,
        "Parallel tempering for two or more workers");
    app.add_option("--exchange-interval", params.exchange_interval,
                   "Approximate moves per worker between replica exchanges");
    app.add_option("--t-init", params.t_init,
                   "Initial temperature (0 = auto-calibrate)");
    app.add_option("--t-min", params.t_min,
                   "Minimum temperature");
    app.add_option("--cooling", params.cooling,
                   "Single-worker geometric cooling factor");
    app.add_option("--moves-per-cool", params.moves_per_cool,
                   "Single-worker moves between cooling steps");
    app.add_option("--stuck-threshold", params.stuck_threshold,
                   "Single-worker moves without improvement before reheating");
    app.add_option("--reheat", params.reheat,
                   "Fraction of initial temperature restored when stuck");
    app.add_option("--reseed-factor", params.reseed_factor,
                   "Reseed from the best when this factor behind it");

    CLI11_PARSE(app, argc, argv);

    if (weight > order) {
        std::cerr << "Error: w must not exceed n\n";
        return 1;
    }
    if (threads_option->count() == 0 && params.use_hyperthreads)
        params.threads = logical;
    params.pin_threads = !no_pin;

    std::string error;
    if (!valid_params(params, error)) {
        std::cerr << "Error: " << error << "\n";
        return 1;
    }
    const std::string impossible = impossibility_reason(order, weight);
    if (!impossible.empty()) {
        std::cerr << "Error: W(" << order << ',' << weight
                  << ") cannot exist: " << impossible << "\n";
        return 1;
    }

    const uint32_t start_seed = params.seed
        ? (uint32_t)params.seed
        : (uint32_t)std::random_device{}();
    const auto initialization_start = std::chrono::steady_clock::now();

    xweigh::State start;
    const bool loaded_start = !start_file.empty();
    const bool use_sylvester_blocks =
        !loaded_start && is_power_of_two(weight) && order % weight == 0;
    try {
        if (loaded_start) {
            start = xweigh::read_state_csv(
                start_file, order, weight);
        } else if (use_sylvester_blocks) {
            start = xweigh::State::sylvester_blocks(order, weight);
        } else {
            start = xweigh::State::random_start(order, weight, start_seed);
        }
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: insufficient memory for W(" << order << ','
                  << weight << ") with " << params.threads << " workers\n";
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return 1;
    }

    const double initialization_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initialization_start).count();
    const size_t worker_bytes =
        start.estimated_bytes() + start.entries().size();
    std::cout << "[init] seconds=" << initialization_seconds
              << " worker_mib="
              << (double)worker_bytes / (1024.0 * 1024.0)
              << " total_worker_mib="
              << (double)worker_bytes * params.threads /
                     (1024.0 * 1024.0)
              << "\n";
    if (loaded_start) {
        std::cout << "[start] loaded " << start_file
                  << " score=" << start.score()
                  << " conflicts=" << start.conflicts() << "\n";
    } else if (use_sylvester_blocks) {
        std::cout << "[construct] using direct sum of Sylvester blocks\n";
    }

    xweigh::State best;
    try {
        best = start.solved()
            ? std::move(start)
            : xweigh::run_annealer(std::move(start), params);
        xweigh::write_state_csv(best, params.output);
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: insufficient memory while starting worker states\n";
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return 1;
    }

    if (!best.verify_support()) {
        std::cerr << "Error: internal support invariant failed\n";
        return 1;
    }

    bool verified = false;
    if (best.solved()) {
        verified = best.verify_weighing();
        if (!verified) {
            std::cerr << "Error: score reached zero but exact verification failed\n";
            return 1;
        }
    }

    std::cout << "Final best score: " << best.score()
              << " conflicts=" << best.conflicts() << "\n";
    if (verified)
        std::cout << "Found W(" << order << ',' << weight
                  << "); exact verification passed\n";
    else
        std::cout << "No weighing matrix found; wrote the best fixed-weight "
                     "candidate to " << params.output << "\n";
    if (g_interrupted.load(std::memory_order_relaxed))
        std::cout << "[interrupted] best candidate written before exit\n";
    if (order <= 20) {
        std::cout << "W:\n";
        best.print();
    }
    return 0;
}
