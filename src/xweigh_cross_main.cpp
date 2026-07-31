// xweigh_cross: exact alternating row/column extension search.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "xweigh_cross_search.hpp"

namespace {

std::string indices_string(uint64_t mask) {
    std::string result;
    for (int index : xweigh_cross::mask_indices(mask)) {
        if (!result.empty()) result += ',';
        result += std::to_string(index + 1);
    }
    return result;
}

bool valid_params(const xweigh_cross::Params& params,
                  int weight, std::string& error) {
    if (weight < 1 || weight > xweigh_cross::MAX_ORDER) {
        error = "--weight must be in [1, 63]";
    } else if (params.threads < 1) {
        error = "--threads must be positive";
    } else if (params.max_seconds < 0.0) {
        error = "--max-seconds must be nonnegative";
    } else if (params.seed_limit < 1) {
        error = "--seed-limit must be positive";
    } else if (params.seed_index < -1 ||
               (params.seed_index >= 0 &&
                static_cast<size_t>(params.seed_index) >=
                    params.seed_limit)) {
        error = "--seed-index must be in the selected seed portfolio";
    } else if (params.radius_start < 0) {
        error = "--radius-start must be nonnegative";
    } else if (params.radius_max < params.radius_start) {
        error = "--radius-max must be at least --radius-start";
    } else if (!params.exhaustive && params.candidate_limit < 1) {
        error = "--branch-limit must be positive";
    }
    return error.empty();
}

std::vector<xweigh_cross::SeedPair> analyze_candidate(
    const xweigh_cross::Matrix& candidate, size_t seed_limit) {
    auto portfolio =
        xweigh_cross::build_seed_portfolio(candidate, seed_limit);
    const uint64_t largest_rows = portfolio.front().rows;
    const uint64_t largest_columns = portfolio.front().columns;

    std::cout << "order=" << candidate.order()
              << " weight=" << candidate.weight()
              << " fixed_weight="
              << (candidate.verify_fixed_weight() ? "yes" : "no")
              << " score=" << candidate.score()
              << " conflicts=" << candidate.conflicts() << '\n';
    std::cout << "max_row_clique=" << std::popcount(largest_rows)
              << " rows=" << indices_string(largest_rows) << '\n';
    std::cout << "max_column_clique="
              << std::popcount(largest_columns)
              << " columns=" << indices_string(largest_columns)
              << '\n';
    std::cout << "seed_portfolio=" << portfolio.size() << '\n';
    for (size_t index = 0; index < portfolio.size(); ++index) {
        std::cout << "seed=" << index
                  << " rows=" << std::popcount(portfolio[index].rows)
                  << " columns="
                  << std::popcount(portfolio[index].columns)
                  << " progress="
                  << std::popcount(portfolio[index].rows) +
                         std::popcount(portfolio[index].columns)
                  << '\n';
    }
    return portfolio;
}

}  // namespace

int main(int argc, char** argv) {
    install_signal_handlers();

    CLI::App app{
        "xweigh_cross exact candidate-guided Criss-Cross search"
    };
    xweigh_cross::Params params;
    params.threads = std::max(
        1, static_cast<int>(std::thread::hardware_concurrency()));
    std::string candidate_file;
    int weight = 25;

    app.add_option("candidate", candidate_file,
                   "Ternary square CSV candidate")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-w,--weight", weight,
                   "Required row and column weight");
    app.add_flag("--analyze-only", params.analyze_only,
                 "Print orthogonality seeds and exit");
    app.add_option("--seed-limit", params.seed_limit,
                   "Maximum number of row/column seed pairs");
    app.add_option("--seed-index", params.seed_index,
                   "Use one zero-based seed pair (-1 = all)");
    app.add_option("--radius-start", params.radius_start,
                   "First Hamming radius tried");
    app.add_option("--radius-max", params.radius_max,
                   "Largest Hamming radius tried");
    app.add_option("--branch-limit", params.candidate_limit,
                   "Nearest extensions retained at each frontier node");
    app.add_option("--assignment-limit", params.assignment_limit,
                   "Maximum half assignments per MITM side (0 = unlimited)");
    app.add_option("--node-limit", params.frontier_node_limit,
                   "Maximum frontier nodes (0 = unlimited)");
    app.add_option("-t,--threads", params.threads,
                   "Number of independent root workers");
    app.add_option("--max-seconds", params.max_seconds,
                   "Wall-clock budget (0 = unlimited)");
    app.add_option("--checkpoint", params.checkpoint,
                   "Atomic best-frontier checkpoint path");
    app.add_option("--resume", params.resume,
                   "Resume a frontier checkpoint")
        ->check(CLI::ExistingFile);
    app.add_option("-o,--out", params.output,
                   "Output CSV for an exact solution");
    app.add_flag("--exhaustive", params.exhaustive,
                 "Remove radius, branch, assignment, and node limits");

    CLI11_PARSE(app, argc, argv);

    std::string error;
    if (!valid_params(params, weight, error)) {
        std::cerr << "Error: " << error << '\n';
        return 1;
    }

    try {
        const xweigh_cross::Matrix candidate =
            xweigh_cross::Matrix::read_csv(candidate_file, weight);
        if (!candidate.verify_fixed_weight()) {
            std::cerr
                << "Error: candidate must have exactly " << weight
                << " nonzeros in every row and column\n";
            return 1;
        }
        params.radius_start =
            std::min(params.radius_start, candidate.order());
        params.radius_max =
            std::min(params.radius_max, candidate.order());
        const auto portfolio =
            analyze_candidate(candidate, params.seed_limit);
        if (params.analyze_only) return 0;

        if (candidate.verify_weighing()) {
            xweigh_cross::write_matrix_csv(candidate, params.output);
            std::cout << "candidate is already an exact weighing matrix; "
                      << "wrote " << params.output << '\n';
            return 0;
        }

        if (params.exhaustive) {
            params.radius_start = candidate.order();
            params.radius_max = candidate.order();
            params.candidate_limit = 0;
            params.assignment_limit = 0;
            params.frontier_node_limit = 0;
            std::cout
                << "exhaustive mode is complete for the selected seed "
                << "portfolio but may require exponential time and memory\n";
        }

        const xweigh_cross::SearchResult result =
            xweigh_cross::run_search(candidate, params, &portfolio);
        std::cout << "[summary] seconds=" << result.seconds
                  << " nodes=" << result.frontier_nodes
                  << " extensions=" << result.generated_extensions
                  << " row_extensions="
                  << result.generated_row_extensions
                  << " column_extensions="
                  << result.generated_column_extensions
                  << " left_assignments=" << result.left_assignments
                  << " right_assignments=" << result.right_assignments
                  << " truncated_generators="
                  << result.truncated_generators
                  << " best_rows=" << result.best.row_count()
                  << " best_columns=" << result.best.column_count()
                  << '\n';
        if (result.solution.has_value()) {
            if (!result.solution->verify_weighing())
                throw std::runtime_error(
                    "internal error: final matrix verification failed");
            xweigh_cross::write_matrix_csv(
                *result.solution, params.output);
            std::cout << "verified W(" << candidate.order() << ','
                      << weight << "); wrote " << params.output << '\n';
            return 0;
        }
        std::cout << "no exact matrix found within the selected limits; "
                  << "best frontier is in " << params.checkpoint << '\n';
        return 2;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
}
