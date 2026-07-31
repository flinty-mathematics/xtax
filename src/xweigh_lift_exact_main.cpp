// xweigh_lift_exact: exhaustive 5-circulant block lift search for W(35,25).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "xweigh_lift_exact.hpp"

namespace {

bool valid_params(const xweigh_lift_exact::Params& params,
                  std::string& error) {
    if (params.threads < 1) {
        error = "--threads must be positive";
    } else if (params.max_seconds < 0.0) {
        error = "--max-seconds must be nonnegative";
    }
    return error.empty();
}

std::string class_label(int index, const std::string& name) {
    if (index < 0) return name;
    return std::to_string(index + 1) + ":" + name;
}

}  // namespace

int main(int argc, char** argv) {
    install_signal_handlers();

    CLI::App app{
        "xweigh_lift_exact exhaustive W(35,25) 5-circulant lift search"
    };
    xweigh_lift_exact::Params params;
    params.threads = std::max(
        1, static_cast<int>(std::thread::hardware_concurrency()));
    int template_number = 0;
    bool list_templates = false;

    app.add_option(
        "-o,--out", params.output,
        "Output CSV for the first weighing-matrix lift found");
    app.add_option(
        "-t,--threads", params.threads,
        "Number of search worker threads");
    app.add_option(
        "--template", template_number,
        "IW(7,25) template number (0 searches all 44 classes)")
        ->check(CLI::Range(0, 44));
    app.add_option(
        "--template-file", params.template_file,
        "Search a custom template CSV instead of the built-in classes");
    app.add_flag(
        "--list-templates", list_templates,
        "List the 44 embedded IW(7,25) templates and exit");
    app.add_flag(
        "--count-only", params.count_only,
        "Report per-row lift counts without searching");
    app.add_flag(
        "--no-canonical", params.no_canonical,
        "Disable the cyclic-shift symmetry filters (validation runs)");
    app.add_flag(
        "--find-all", params.find_all,
        "Count every lift instead of stopping at the first");
    app.add_option(
        "--node-limit", params.node_limit,
        "Stop after this many row assignments (0 = unlimited)");
    app.add_option(
        "--max-seconds", params.max_seconds,
        "Wall-clock budget in seconds (0 = unlimited)");

    CLI11_PARSE(app, argc, argv);

    if (list_templates) {
        for (size_t index = 0;
             index < xweigh_lift::IW7_TEMPLATES.size(); ++index) {
            std::cout << index + 1 << ' '
                      << xweigh_lift::IW7_TEMPLATES[index].name << '\n';
        }
        return 0;
    }
    if (!params.template_file.empty() && template_number != 0) {
        std::cerr << "Error: --template and --template-file are mutually "
                     "exclusive\n";
        return 1;
    }
    params.template_index =
        template_number == 0 ? -1 : template_number - 1;
    std::string error;
    if (!valid_params(params, error)) {
        std::cerr << "Error: " << error << '\n';
        return 1;
    }

    try {
        // (class index or -1 for a custom file, template)
        std::vector<std::pair<int, xweigh_lift_exact::TemplateInput>>
            classes;
        if (!params.template_file.empty()) {
            classes.emplace_back(
                -1,
                xweigh_lift_exact::read_template_csv(params.template_file));
        } else if (params.template_index >= 0) {
            classes.emplace_back(
                params.template_index,
                xweigh_lift_exact::built_in_template(params.template_index));
        } else {
            for (int index = 0;
                 index <
                 static_cast<int>(xweigh_lift::IW7_TEMPLATES.size());
                 ++index) {
                classes.emplace_back(
                    index, xweigh_lift_exact::built_in_template(index));
            }
        }

        std::cout << "[init] classes=" << classes.size()
                  << " threads=" << params.threads << " mode="
                  << (params.count_only ? "count" : "search")
                  << " canonical="
                  << (params.no_canonical ? "off" : "on") << '\n';

        std::optional<std::chrono::steady_clock::time_point> deadline;
        if (params.max_seconds > 0.0) {
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::duration_cast<
                           std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(
                               params.max_seconds));
        }

        if (params.count_only) {
            for (const auto& [index, tmpl] : classes) {
                const xweigh_lift_exact::ClassPlan plan =
                    xweigh_lift_exact::build_class_plan(tmpl, params);
                for (int row = 0; row < tmpl.order; ++row) {
                    std::cout << "[count] class="
                              << class_label(index, tmpl.name)
                              << " row=" << row + 1 << " raw="
                              << plan.row_counts_raw[
                                     static_cast<size_t>(row)]
                              << " canonical="
                              << plan.row_counts_filtered[
                                     static_cast<size_t>(row)]
                              << '\n';
                }
                if (g_stop.load(std::memory_order_relaxed) ||
                    (deadline.has_value() &&
                     std::chrono::steady_clock::now() >= *deadline)) {
                    std::cout << "[stopped] counting interrupted\n";
                    break;
                }
            }
            return 0;
        }

        const auto start = std::chrono::steady_clock::now();
        uint64_t total_nodes = 0;
        uint64_t total_solutions = 0;
        int exhausted = 0;
        bool solved = false;
        bool stopped = false;
        for (const auto& [index, tmpl] : classes) {
            const std::string label = class_label(index, tmpl.name);
            const xweigh_lift_exact::ClassOutcome outcome =
                xweigh_lift_exact::search_class(tmpl, params, deadline,
                                                label);
            total_nodes += outcome.nodes;
            total_solutions += outcome.solutions;
            std::cout << "[class " << label << "] result="
                      << (outcome.solved
                              ? "solved"
                              : outcome.completed ? "exhausted" : "stopped")
                      << " solutions=" << outcome.solutions
                      << " nodes=" << outcome.nodes
                      << " seconds=" << outcome.seconds << " order=[";
            for (size_t position = 0; position < outcome.row_order.size();
                 ++position) {
                if (position > 0) std::cout << ' ';
                std::cout << outcome.row_order[position] + 1;
            }
            std::cout << "] depth_nodes=[";
            for (int depth = 0; depth < tmpl.order; ++depth) {
                if (depth > 0) std::cout << ' ';
                std::cout << outcome.depth_nodes[
                    static_cast<size_t>(depth)];
            }
            std::cout << "]\n" << std::flush;
            if (outcome.completed) ++exhausted;
            if (outcome.solved && !solved) {
                solved = true;
                const std::vector<int> matrix =
                    xweigh_lift_exact::expand_solution(tmpl,
                                                       outcome.solution);
                xweigh_lift_exact::write_matrix_csv(
                    matrix, xweigh_lift_exact::BLOCK_ORDER * tmpl.order,
                    params.output);
                std::cout << "[solved] class=" << label
                          << " weight=" << tmpl.weight
                          << " output=" << params.output
                          << " (exact verification passed)\n";
                if (!params.find_all) break;
            }
            if (!outcome.completed && !outcome.solved) {
                stopped = true;
                break;
            }
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        std::cout << "[done] classes=" << classes.size()
                  << " exhausted=" << exhausted
                  << " solutions=" << total_solutions
                  << " nodes=" << total_nodes
                  << " seconds=" << seconds << '\n';
        if (solved) {
            std::cout << "Found a 5-circulant block weighing matrix; "
                         "written to " << params.output << '\n';
        } else if (!stopped &&
                   exhausted == static_cast<int>(classes.size())) {
            std::cout << "Exhausted every searched class: no 5-circulant "
                         "block lift exists for them\n";
        } else {
            std::cout << "Search stopped before exhausting all classes\n";
        }
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: insufficient memory during exact search\n";
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
