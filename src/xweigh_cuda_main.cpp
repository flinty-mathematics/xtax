// xweigh_cuda: GPU population annealer for small weighing matrices W(n,w).

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include "CLI11.hpp"
#include "xweigh_cuda.hpp"

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

bool valid_params(const xweigh_cuda::Params& params, std::string& error) {
    if (params.device < 0) error = "--device must be nonnegative";
    else if (params.replicas < 0) error = "--replicas must be nonnegative";
    else if (params.moves_per_launch < 1)
        error = "--moves-per-launch must be positive";
    else if (params.max_seconds < 0.0)
        error = "--max-seconds must be nonnegative";
    else if (params.save_interval < 0.0)
        error = "--save-interval must be nonnegative";
    else if (params.sign_fraction < 0.0 || params.sign_fraction > 1.0)
        error = "--sign-fraction must be in [0, 1]";
    else if (params.double_sign_fraction < 0.0 ||
             params.double_sign_fraction > 1.0)
        error = "--double-sign-fraction must be in [0, 1]";
    else if (params.switch_sign_fraction < 0.0 ||
             params.switch_sign_fraction > 1.0)
        error = "--switch-sign-fraction must be in [0, 1]";
    else if (params.squared_objective_fraction < 0.0 ||
             params.squared_objective_fraction > 1.0)
        error = "--squared-objective-fraction must be in [0, 1]";
    else if (params.parity_objective_fraction < 0.0 ||
             params.parity_objective_fraction > 1.0)
        error = "--parity-objective-fraction must be in [0, 1]";
    else if (params.squared_objective_fraction +
                 params.parity_objective_fraction >
             1.0)
        error = "alternate objective fractions must sum to at most 1";
    else if (params.greedy_fraction < 0.0 || params.greedy_fraction > 1.0)
        error = "--greedy-fraction must be in [0, 1]";
    else if (params.target_fraction < 0.0 || params.target_fraction > 1.0)
        error = "--target-fraction must be in [0, 1]";
    else if (params.candidate_samples < 1 ||
             params.candidate_samples > xweigh_cuda::MAX_CANDIDATE_SAMPLES) {
        error = "--candidate-samples must be in [1, " +
            std::to_string(xweigh_cuda::MAX_CANDIDATE_SAMPLES) + "]";
    } else if (params.target_samples < 1 ||
               params.target_samples > xweigh_cuda::MAX_TARGET_SAMPLES) {
        error = "--target-samples must be in [1, " +
            std::to_string(xweigh_cuda::MAX_TARGET_SAMPLES) + "]";
    } else if (params.t_init < 0.0) {
        error = "--t-init must be nonnegative";
    } else if (params.t_min <= 0.0) {
        error = "--t-min must be positive";
    } else if (params.cooling <= 0.0 || params.cooling >= 1.0) {
        error = "--cooling must be in (0, 1)";
    } else if (params.moves_per_cool < 1) {
        error = "--moves-per-cool must be positive";
    } else if (params.stuck_threshold < 1) {
        error = "--stuck-threshold must be positive";
    } else if (params.reheat <= 0.0) {
        error = "--reheat must be positive";
    } else if (params.restart_interval < 0.0) {
        error = "--restart-interval must be nonnegative";
    } else if (params.restart_fraction <= 0.0 ||
               params.restart_fraction > 1.0) {
        error = "--restart-fraction must be in (0, 1]";
    } else if (params.restart_kick_min < 0) {
        error = "--restart-kick-min must be nonnegative";
    } else if (params.restart_kick_max <
               params.restart_kick_min) {
        error = "--restart-kick-max must be at least --restart-kick-min";
    }
    return error.empty();
}

bool finish(const xweigh::State& best, int order, int weight,
            const std::string& output) {
    if (!best.verify_support()) {
        std::cerr << "Error: internal support invariant failed\n";
        return false;
    }
    if (best.solved() && !best.verify_weighing()) {
        std::cerr << "Error: score reached zero but exact verification failed\n";
        return false;
    }

    std::cout << "Final best score: " << best.score()
              << " conflicts=" << best.conflicts() << "\n";
    if (best.solved()) {
        std::cout << "Found W(" << order << ',' << weight
                  << "); exact verification passed\n";
    } else {
        std::cout << "No weighing matrix found; wrote the best fixed-weight "
                     "candidate to " << output << "\n";
    }
    if (g_interrupted.load(std::memory_order_relaxed))
        std::cout << "[interrupted] best candidate written before exit\n";
    if (order <= 20) {
        std::cout << "W:\n";
        best.print();
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    install_signal_handlers();

    CLI::App app{ "xweigh_cuda shared-memory weighing-matrix annealer" };
    int order = 0;
    int weight = 0;
    xweigh_cuda::Params params;
    std::string start_file;

    app.add_option("n", order, "Matrix order n")
        ->required()
        ->check(CLI::Range(1, xweigh::MAX_ORDER));
    app.add_option("w", weight, "Matrix weight w")
        ->required()
        ->check(CLI::Range(1, xweigh::MAX_ORDER));
    app.add_option("-o,--out", params.output,
                   "Output CSV for the best matrix");
    app.add_option("--start", start_file,
                   "Start every replica from a complete fixed-weight "
                   "ternary CSV")
        ->check(CLI::ExistingFile);
    app.add_option("--device", params.device, "CUDA device ordinal");
    app.add_option("--replicas", params.replicas,
                   "Independent GPU replicas (0 = occupancy-based)");
    app.add_option("--moves-per-launch", params.moves_per_launch,
                   "Annealing iterations per replica between host checkpoints");
    app.add_option("--seed", params.seed,
                   "Base RNG seed (0 = random_device)");
    app.add_option("--max-seconds", params.max_seconds,
                   "Wall-clock search budget (0 = until solved or interrupted)");
    app.add_option("--save-interval", params.save_interval,
                   "Minimum seconds between best-matrix writes");

    app.add_option("--sign-fraction", params.sign_fraction,
                   "Probability of proposing a sign flip");
    app.add_option(
        "--double-sign-fraction", params.double_sign_fraction,
        "Probability of first proposing an atomic double sign flip");
    app.add_option(
        "--switch-sign-fraction", params.switch_sign_fraction,
        "Probability of flipping each sign in a support switch");
    app.add_option(
        "--squared-objective-fraction",
        params.squared_objective_fraction,
        "Fraction of replicas guided by squared Gram residual");
    app.add_option(
        "--parity-objective-fraction",
        params.parity_objective_fraction,
        "Fraction of replicas minimizing odd support intersections");
    app.add_option("--greedy-fraction", params.greedy_fraction,
                   "Probability of selecting the best sampled move");
    app.add_option("--candidate-samples", params.candidate_samples,
                   "Candidate count for a sampled greedy move");
    app.add_option("--target-fraction", params.target_fraction,
                   "Probability of targeting a high-residual row");
    app.add_option("--target-samples", params.target_samples,
                   "Tournament size for high-residual row selection");

    app.add_option("--t-init", params.t_init,
                   "Initial temperature (0 = auto-calibrate)");
    app.add_option("--t-min", params.t_min, "Minimum temperature");
    app.add_option("--cooling", params.cooling,
                   "Geometric cooling factor");
    app.add_option("--moves-per-cool", params.moves_per_cool,
                   "Moves between cooling steps");
    app.add_option("--stuck-threshold", params.stuck_threshold,
                   "Moves without improvement before reheating");
    app.add_option("--reheat", params.reheat,
                   "Fraction of initial temperature restored when stuck");
    app.add_option(
        "--restart-interval", params.restart_interval,
        "Seconds without a global improvement before elite restarts "
        "(0 disables)");
    app.add_option(
        "--restart-fraction", params.restart_fraction,
        "Fraction of non-elite replicas restarted from the global best");
    app.add_option(
        "--restart-kick-min", params.restart_kick_min,
        "Minimum legal perturbation moves after an elite restart");
    app.add_option(
        "--restart-kick-max", params.restart_kick_max,
        "Maximum legal perturbation moves after an elite restart");

    CLI11_PARSE(app, argc, argv);

    if (weight > order) {
        std::cerr << "Error: w must not exceed n\n";
        return 1;
    }

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

    try {
        std::optional<xweigh::State> initial_state;
        if (!start_file.empty()) {
            initial_state = xweigh::read_state_csv(
                start_file, order, weight);
            std::cout << "[start] loaded " << start_file
                      << " score=" << initial_state->score()
                      << " conflicts=" << initial_state->conflicts()
                      << "\n";
            if (initial_state->solved()) {
                xweigh::write_state_csv(
                    *initial_state, params.output);
                return finish(
                    *initial_state, order, weight, params.output)
                    ? 0 : 1;
            }
        }

        if (!initial_state.has_value() &&
            is_power_of_two(weight) && order % weight == 0) {
            xweigh::State best =
                xweigh::State::sylvester_blocks(order, weight);
            std::cout << "[construct] using direct sum of Sylvester blocks\n";
            xweigh::write_state_csv(best, params.output);
            return finish(best, order, weight, params.output) ? 0 : 1;
        }

        if (params.seed == 0)
            params.seed = (uint64_t)std::random_device{}();
        xweigh_cuda::RunResult result =
            xweigh_cuda::run_annealer(
                order, weight, params,
                initial_state.has_value() ? &*initial_state : nullptr);
        xweigh::write_state_csv(result.best, params.output);
        return finish(result.best, order, weight, params.output) ? 0 : 1;
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: insufficient host memory for W(" << order << ','
                  << weight << ")\n";
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
    }
    return 1;
}
