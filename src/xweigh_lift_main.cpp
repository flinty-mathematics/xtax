// xweigh_lift: search for W(35,25) in 5-circulant 7 by 7 blocks.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "xweigh_lift_search.hpp"

namespace {

bool valid_params(const xweigh_lift::Params& params, std::string& error) {
    if (params.threads < 1) {
        error = "--threads must be positive";
    } else if (params.template_index < -1 ||
               params.template_index >=
                   static_cast<int>(xweigh_lift::IW7_TEMPLATES.size())) {
        error = "--template must be 0 or in [1, 44]";
    } else if (params.max_seconds < 0.0) {
        error = "--max-seconds must be nonnegative";
    } else if (params.greedy_fraction < 0.0 ||
               params.greedy_fraction > 1.0) {
        error = "--greedy-fraction must be in [0, 1]";
    } else if (params.target_fraction < 0.0 ||
               params.target_fraction > 1.0) {
        error = "--target-fraction must be in [0, 1]";
    } else if (params.pair_fraction < 0.0 ||
               params.pair_fraction > 1.0) {
        error = "--pair-fraction must be in [0, 1]";
    } else if (params.pair_samples < 1) {
        error = "--pair-samples must be positive";
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
    } else if (params.stuck_threshold >
               std::numeric_limits<uint64_t>::max() / 3) {
        error = "--stuck-threshold is too large";
    } else if (params.reheat <= 0.0) {
        error = "--reheat must be positive";
    } else if (params.exact_threshold < -1) {
        error = "--exact-threshold must be at least -1";
    }
    return error.empty();
}

bool verify_templates() {
    for (const auto& record : xweigh_lift::IW7_TEMPLATES) {
        for (int first = 0; first < xweigh_lift::TEMPLATE_ORDER; ++first) {
            for (int second = 0;
                 second < xweigh_lift::TEMPLATE_ORDER; ++second) {
                int dot = 0;
                for (int column = 0;
                     column < xweigh_lift::TEMPLATE_ORDER; ++column) {
                    dot +=
                        record.entries[static_cast<size_t>(
                            first * xweigh_lift::TEMPLATE_ORDER +
                            column)] *
                        record.entries[static_cast<size_t>(
                            second * xweigh_lift::TEMPLATE_ORDER +
                            column)];
                }
                const int target = first == second ? 25 : 0;
                if (dot != target) return false;
            }
        }
    }
    return true;
}

std::vector<int8_t> read_start_csv(const std::string& filename) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error(
            "failed to open start matrix: " + filename);
    }
    std::vector<std::vector<int8_t>> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::stringstream stream(line);
        std::vector<int8_t> row;
        std::string token;
        while (std::getline(stream, token, ',')) {
            size_t consumed = 0;
            int value = 0;
            try {
                value = std::stoi(token, &consumed);
            } catch (const std::exception&) {
                consumed = 0;
            }
            if (consumed != token.size() || value < -1 || value > 1) {
                throw std::runtime_error(
                    "invalid ternary entry in start matrix");
            }
            row.push_back(static_cast<int8_t>(value));
        }
        rows.push_back(std::move(row));
    }
    if (rows.size() != xweigh_lift::MATRIX_ORDER ||
        std::any_of(
            rows.begin(), rows.end(),
            [](const auto& row) {
                return row.size() != xweigh_lift::MATRIX_ORDER;
            })) {
        throw std::runtime_error("start matrix must be 35 by 35");
    }

    std::vector<int8_t> entries(
        static_cast<size_t>(
            xweigh_lift::MATRIX_ORDER * xweigh_lift::MATRIX_ORDER));
    for (int row = 0; row < xweigh_lift::MATRIX_ORDER; ++row) {
        for (int column = 0;
             column < xweigh_lift::MATRIX_ORDER; ++column) {
            entries[static_cast<size_t>(
                column * xweigh_lift::MATRIX_ORDER + row)] =
                rows[static_cast<size_t>(row)]
                    [static_cast<size_t>(column)];
        }
    }
    return entries;
}

}  // namespace

int main(int argc, char** argv) {
    install_signal_handlers();

    CLI::App app{
        "xweigh_lift structured W(35,25) 5-circulant lift search"
    };
    xweigh_lift::Params params;
    params.threads = std::max(
        1, static_cast<int>(std::thread::hardware_concurrency()));
    int template_number = 0;
    bool list_templates = false;
    std::string start_file;

    app.add_option(
        "-o,--out", params.output,
        "Output CSV for the best expanded 35 by 35 candidate");
    app.add_option(
        "-t,--threads", params.threads,
        "Number of independent search workers");
    app.add_option(
        "--template", template_number,
        "IW(7,25) template number (0 cycles through all 44)")
        ->check(CLI::Range(0, 44));
    app.add_flag(
        "--list-templates", list_templates,
        "List the 44 embedded IW(7,25) templates and exit");
    app.add_option(
        "--seed", params.seed,
        "Base RNG seed (0 = random_device)");
    app.add_option(
        "--start", start_file,
        "Resume from a block-circulant 35 by 35 CSV candidate");
    app.add_option(
        "--max-seconds", params.max_seconds,
        "Wall-clock search budget (0 = until solved or interrupted)");

    app.add_option(
        "--greedy-fraction", params.greedy_fraction,
        "Probability of exhaustive best replacement in a block row");
    app.add_option(
        "--target-fraction", params.target_fraction,
        "Probability of mutating the block row with largest residual");
    app.add_option(
        "--pair-fraction", params.pair_fraction,
        "Probability of a self-correlation-preserving pair move");
    app.add_option(
        "--pair-samples", params.pair_samples,
        "Column pairs searched by each greedy move");
    app.add_option(
        "--t-init", params.t_init,
        "Initial temperature (0 = auto-calibrate each restart)");
    app.add_option(
        "--t-min", params.t_min,
        "Minimum annealing temperature");
    app.add_option(
        "--cooling", params.cooling,
        "Geometric cooling factor");
    app.add_option(
        "--moves-per-cool", params.moves_per_cool,
        "Moves between cooling steps");
    app.add_option(
        "--stuck-threshold", params.stuck_threshold,
        "Moves without improvement before reheating");
    app.add_option(
        "--restart-moves", params.restart_moves,
        "Maximum moves on one template before restarting "
        "(0 disables)");
    app.add_option(
        "--reheat", params.reheat,
        "Fraction of the initial temperature restored when stuck");

    app.add_option(
        "--exact-threshold", params.exact_threshold,
        "Try exact meet-in-the-middle row completion at this score "
        "(-1 disables)");
    app.add_option(
        "--exact-max-side", params.exact_max_side,
        "Maximum combinations on either side of exact completion "
        "(0 disables)");
    app.add_option(
        "--exact-column-nodes", params.exact_column_nodes,
        "Node limit for exact frozen-column completion "
        "(0 disables)");
    app.add_option(
        "--exact-two-column-nodes", params.exact_two_column_nodes,
        "Node limit for exact two-column completion "
        "(0 disables)");

    CLI11_PARSE(app, argc, argv);

    if (list_templates) {
        for (size_t index = 0;
             index < xweigh_lift::IW7_TEMPLATES.size(); ++index) {
            std::cout << index + 1 << ' '
                      << xweigh_lift::IW7_TEMPLATES[index].name << '\n';
        }
        return 0;
    }

    params.template_index =
        template_number == 0 ? -1 : template_number - 1;
    std::string error;
    if (!valid_params(params, error)) {
        std::cerr << "Error: " << error << '\n';
        return 1;
    }
    if (!verify_templates()) {
        std::cerr << "Error: embedded IW(7,25) template verification failed\n";
        return 1;
    }

    try {
        const auto& sequences = xweigh_lift::sequence_table();
        (void)sequences;
        std::optional<xweigh_lift::State> initial_state;
        if (!start_file.empty()) {
            initial_state = xweigh_lift::State::from_expanded_entries(
                read_start_csv(start_file));
            if (params.template_index >= 0 &&
                params.template_index !=
                    initial_state->template_index()) {
                throw std::runtime_error(
                    "--template does not match --start block sums");
            }
            params.template_index = initial_state->template_index();
        }
        std::cout << "[init] templates="
                  << xweigh_lift::IW7_TEMPLATES.size()
                  << " ternary_sequences="
                  << xweigh_lift::SEQUENCE_COUNT
                  << " threads=" << params.threads << '\n';
        if (params.template_index >= 0) {
            const auto& record =
                xweigh_lift::IW7_TEMPLATES[
                    static_cast<size_t>(params.template_index)];
            std::cout << "[template] "
                      << params.template_index + 1 << ':'
                      << record.name << '\n';
        } else {
            std::cout << "[template] cycling through all 44 classes\n";
        }
        if (initial_state.has_value()) {
            std::cout << "[start] score=" << initial_state->score()
                      << " conflicts=" << initial_state->conflicts()
                      << '\n';
        }

        xweigh_lift::RunResult result =
            xweigh_lift::run_annealer(
                params,
                initial_state.has_value() ? &*initial_state : nullptr);
        if (!result.best.verify_template_sums()) {
            std::cerr << "Error: block domain invariant failed\n";
            return 1;
        }
        if (!result.best.verify_support_margins()) {
            std::cerr << "Error: block-support margin invariant failed\n";
            return 1;
        }
        if (!result.best.verify_signature_margins()) {
            std::cerr
                << "Error: autocorrelation signature invariant failed\n";
            return 1;
        }
        if (!result.best.verify_cache()) {
            std::cerr << "Error: incremental Gram cache verification failed\n";
            return 1;
        }
        const bool verified =
            result.best.solved() && result.best.verify_weighing();
        if (result.best.solved() && !verified) {
            std::cerr
                << "Error: score reached zero but exact verification failed\n";
            return 1;
        }

        xweigh_lift::write_state_csv(result.best, params.output);
        const double moves_per_second =
            result.seconds > 0.0
                ? static_cast<double>(result.moves) / result.seconds
                : 0.0;
        const double evaluations_per_second =
            result.seconds > 0.0
                ? static_cast<double>(result.candidate_evaluations) /
                      result.seconds
                : 0.0;
        std::cout << "[done] moves=" << result.moves
                  << " candidate_evaluations="
                  << result.candidate_evaluations
                  << " restarts=" << result.restarts
                  << " exact_attempts=" << result.exact_attempts
                  << " exact_successes=" << result.exact_successes
                  << " seconds=" << result.seconds
                  << " moves_per_second=" << moves_per_second
                  << " candidate_evaluations_per_second="
                  << evaluations_per_second << '\n';
        std::cout << "Final best template: "
                  << result.best.template_index() + 1 << ':'
                  << result.best.template_name() << '\n';
        std::cout << "Final best score: " << result.best.score()
                  << " conflicts=" << result.best.conflicts() << '\n';
        if (verified) {
            std::cout
                << "Found 5-circulant W(35,25); exact verification passed\n";
        } else {
            std::cout
                << "No weighing matrix found; wrote the best structured "
                   "candidate to "
                << params.output << '\n';
        }
        if (g_interrupted.load(std::memory_order_relaxed)) {
            std::cout
                << "[interrupted] best candidate written before exit\n";
        }
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: insufficient memory during structured search\n";
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
