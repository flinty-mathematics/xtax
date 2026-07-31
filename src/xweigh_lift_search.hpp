// xweigh_lift_search.hpp: lift IW(7,25) templates to 5-circulant blocks.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stop_signal.hpp"
#include "xweigh_lift_templates.hpp"

namespace xweigh_lift {

constexpr int BLOCK_ORDER = 5;
constexpr int TEMPLATE_ORDER = 7;
constexpr int MATRIX_ORDER = BLOCK_ORDER * TEMPLATE_ORDER;
constexpr int SEQUENCE_COUNT = 243;
constexpr int PAIR_COUNT =
    TEMPLATE_ORDER * (TEMPLATE_ORDER + 1) / 2;
constexpr int GRAM_SIZE = PAIR_COUNT * BLOCK_ORDER;
constexpr int CONSTRAINT_SIZE = TEMPLATE_ORDER * BLOCK_ORDER;

struct Params {
    int threads = 1;
    int template_index = -1;
    uint64_t seed = 0;
    double max_seconds = 0.0;
    std::string output = "best_W35_25.csv";

    double greedy_fraction = 0.25;
    double target_fraction = 0.85;
    double pair_fraction = 0.0;
    int pair_samples = 2;
    double t_init = 0.0;
    double t_min = 0.10;
    double cooling = 0.9995;
    int moves_per_cool = 500;
    uint64_t stuck_threshold = 100000;
    uint64_t restart_moves = 200000;
    double reheat = 1.0;

    int exact_threshold = 24;
    uint64_t exact_max_side = 500000;
    uint64_t exact_column_nodes = 1000000;
    uint64_t exact_two_column_nodes = 250000;
};

struct AutocorrelationSignature {
    uint8_t support = 0;
    int8_t first = 0;
    int8_t second = 0;

    bool operator==(const AutocorrelationSignature&) const = default;
};

struct Sequence {
    std::array<int8_t, BLOCK_ORDER> values{};
    int8_t sum = 0;
    AutocorrelationSignature signature{};
};

class SequenceTable {
public:
    SequenceTable() {
        for (int code = 0; code < SEQUENCE_COUNT; ++code) {
            int remaining = code;
            int sum = 0;
            int support = 0;
            for (int index = 0; index < BLOCK_ORDER; ++index) {
                const int value = remaining % 3 - 1;
                remaining /= 3;
                sequences_[static_cast<size_t>(code)]
                    .values[static_cast<size_t>(index)] =
                    static_cast<int8_t>(value);
                sum += value;
                support += value != 0 ? 1 : 0;
            }
            sequences_[static_cast<size_t>(code)].sum =
                static_cast<int8_t>(sum);
            sequences_[static_cast<size_t>(code)].signature.support =
                static_cast<uint8_t>(support);
            int first = 0;
            int second = 0;
            for (int index = 0; index < BLOCK_ORDER; ++index) {
                const auto& values =
                    sequences_[static_cast<size_t>(code)].values;
                first += values[static_cast<size_t>(index)] *
                         values[static_cast<size_t>(
                             (index + 1) % BLOCK_ORDER)];
                second += values[static_cast<size_t>(index)] *
                          values[static_cast<size_t>(
                              (index + 2) % BLOCK_ORDER)];
            }
            sequences_[static_cast<size_t>(code)].signature.first =
                static_cast<int8_t>(first);
            sequences_[static_cast<size_t>(code)].signature.second =
                static_cast<int8_t>(second);
            by_sum_support_[static_cast<size_t>(sum + BLOCK_ORDER)]
                           [static_cast<size_t>(support)]
                .push_back(static_cast<uint8_t>(code));
            const AutocorrelationSignature signature =
                sequences_[static_cast<size_t>(code)].signature;
            by_signature_[signature_index(sum, signature)]
                .push_back(static_cast<uint8_t>(code));
            auto& signatures =
                signatures_by_sum_[static_cast<size_t>(
                    sum + BLOCK_ORDER)];
            if (std::find(signatures.begin(), signatures.end(),
                          signature) == signatures.end()) {
                signatures.push_back(signature);
            }
        }

        for (int first = 0; first < SEQUENCE_COUNT; ++first) {
            for (int second = 0; second < SEQUENCE_COUNT; ++second) {
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    int value = 0;
                    for (int index = 0; index < BLOCK_ORDER; ++index) {
                        const int shifted =
                            (index + shift) % BLOCK_ORDER;
                        value +=
                            sequences_[static_cast<size_t>(first)]
                                .values[static_cast<size_t>(index)] *
                            sequences_[static_cast<size_t>(second)]
                                .values[static_cast<size_t>(shifted)];
                    }
                    correlations_[correlation_index(
                        static_cast<uint8_t>(first),
                        static_cast<uint8_t>(second), shift)] =
                        static_cast<int8_t>(value);
                }
            }
        }
    }

    const Sequence& sequence(uint8_t id) const {
        return sequences_[static_cast<size_t>(id)];
    }

    const std::vector<uint8_t>& with_sum_support(
        int sum, int support) const {
        return by_sum_support_[
            static_cast<size_t>(sum + BLOCK_ORDER)]
            [static_cast<size_t>(support)];
    }

    const std::vector<uint8_t>& with_signature(
        int sum, const AutocorrelationSignature& signature) const {
        return by_signature_[signature_index(sum, signature)];
    }

    const std::vector<AutocorrelationSignature>& signatures_for_sum(
        int sum) const {
        return signatures_by_sum_[
            static_cast<size_t>(sum + BLOCK_ORDER)];
    }

    int correlation(uint8_t first, uint8_t second, int shift) const {
        return correlations_[correlation_index(first, second, shift)];
    }

    std::optional<uint8_t> find(
        const std::array<int8_t, BLOCK_ORDER>& values) const {
        for (int id = 0; id < SEQUENCE_COUNT; ++id) {
            if (sequences_[static_cast<size_t>(id)].values == values) {
                return static_cast<uint8_t>(id);
            }
        }
        return std::nullopt;
    }

private:
    static size_t signature_index(
        int sum, const AutocorrelationSignature& signature) {
        constexpr size_t correlation_range =
            2 * BLOCK_ORDER + 1;
        return (((static_cast<size_t>(sum + BLOCK_ORDER) *
                   (BLOCK_ORDER + 1) +
                  signature.support) *
                     correlation_range +
                 static_cast<size_t>(
                     signature.first + BLOCK_ORDER)) *
                    correlation_range +
                static_cast<size_t>(
                    signature.second + BLOCK_ORDER));
    }

    static size_t correlation_index(uint8_t first, uint8_t second,
                                    int shift) {
        return (static_cast<size_t>(first) * SEQUENCE_COUNT + second) *
                   BLOCK_ORDER +
               static_cast<size_t>(shift);
    }

    std::array<Sequence, SEQUENCE_COUNT> sequences_{};
    std::array<
        std::array<std::vector<uint8_t>, BLOCK_ORDER + 1>,
        2 * BLOCK_ORDER + 1>
        by_sum_support_{};
    std::array<
        std::vector<uint8_t>,
        (2 * BLOCK_ORDER + 1) * (BLOCK_ORDER + 1) *
            (2 * BLOCK_ORDER + 1) * (2 * BLOCK_ORDER + 1)>
        by_signature_{};
    std::array<
        std::vector<AutocorrelationSignature>,
        2 * BLOCK_ORDER + 1>
        signatures_by_sum_{};
    std::array<int8_t,
               SEQUENCE_COUNT * SEQUENCE_COUNT * BLOCK_ORDER>
        correlations_{};
};

inline const SequenceTable& sequence_table() {
    static const SequenceTable table;
    return table;
}

inline int pair_index(int first, int second) {
    if (first > second) std::swap(first, second);
    const int preceding =
        first * TEMPLATE_ORDER - first * (first - 1) / 2;
    return preceding + second - first;
}

inline int gram_index(int first, int second, int shift) {
    return pair_index(first, second) * BLOCK_ORDER + shift;
}

inline int target_gram_value(int first, int second, int shift) {
    return first == second && shift == 0 ? 25 : 0;
}

struct Evaluation {
    int score_delta = 0;
    int conflict_delta = 0;
};

struct CellMove {
    int cell = 0;
    uint8_t replacement = 0;
    Evaluation evaluation{};
};

struct PairMove {
    int first_cell = 0;
    int second_cell = 0;
    uint8_t first_replacement = 0;
    uint8_t second_replacement = 0;
    Evaluation evaluation{};
};

class State {
public:
    State() = default;

    template <class Rng>
    static State random_start(int template_index, Rng& rng) {
        State state;
        state.template_index_ = template_index;
        const auto& record =
            IW7_TEMPLATES[static_cast<size_t>(template_index)];
        if (!state.initialize_signature_pattern(record, rng)) {
            throw std::runtime_error(
                "could not generate a coupled self-correlation "
                "signature pattern for IW(7,25) template " +
                std::to_string(template_index + 1));
        }
        const auto& table = sequence_table();
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                const int cell = row * TEMPLATE_ORDER + column;
                const int sum =
                    record.entries[static_cast<size_t>(cell)];
                const auto& candidates = table.with_signature(
                    sum, state.signature(row, column));
                std::uniform_int_distribution<size_t> choose(
                    0, candidates.size() - 1);
                state.choices_[static_cast<size_t>(cell)] =
                    candidates[choose(rng)];
            }
        }
        state.rebuild_cache();
        return state;
    }

    static State from_expanded_entries(
        const std::vector<int8_t>& entries) {
        if (entries.size() !=
            static_cast<size_t>(MATRIX_ORDER * MATRIX_ORDER)) {
            throw std::runtime_error(
                "start matrix must be 35 by 35");
        }

        State state;
        const auto& table = sequence_table();
        std::array<int8_t, TEMPLATE_ORDER * TEMPLATE_ORDER>
            template_entries{};
        for (int block_row = 0;
             block_row < TEMPLATE_ORDER; ++block_row) {
            for (int block_column = 0;
                 block_column < TEMPLATE_ORDER; ++block_column) {
                std::array<int8_t, BLOCK_ORDER> values{};
                for (int local_column = 0;
                     local_column < BLOCK_ORDER; ++local_column) {
                    const int row = block_row * BLOCK_ORDER;
                    const int column =
                        block_column * BLOCK_ORDER + local_column;
                    const int8_t value = entries[static_cast<size_t>(
                        column * MATRIX_ORDER + row)];
                    if (value < -1 || value > 1) {
                        throw std::runtime_error(
                            "start matrix has an entry outside "
                            "{-1,0,1}");
                    }
                    values[static_cast<size_t>(local_column)] = value;
                }
                for (int local_row = 0;
                     local_row < BLOCK_ORDER; ++local_row) {
                    for (int local_column = 0;
                         local_column < BLOCK_ORDER; ++local_column) {
                        const int row =
                            block_row * BLOCK_ORDER + local_row;
                        const int column =
                            block_column * BLOCK_ORDER + local_column;
                        const int sequence_index =
                            (local_column - local_row + BLOCK_ORDER) %
                            BLOCK_ORDER;
                        if (entries[static_cast<size_t>(
                                column * MATRIX_ORDER + row)] !=
                            values[static_cast<size_t>(
                                sequence_index)]) {
                            throw std::runtime_error(
                                "start matrix contains a "
                                "non-circulant block");
                        }
                    }
                }
                const auto id = table.find(values);
                if (!id.has_value()) {
                    throw std::runtime_error(
                        "start matrix block is not ternary");
                }
                const int cell =
                    block_row * TEMPLATE_ORDER + block_column;
                state.choices_[static_cast<size_t>(cell)] = *id;
                const auto signature = table.sequence(*id).signature;
                state.supports_[static_cast<size_t>(cell)] =
                    signature.support;
                state.first_correlations_[static_cast<size_t>(cell)] =
                    signature.first;
                state.second_correlations_[static_cast<size_t>(cell)] =
                    signature.second;
                template_entries[static_cast<size_t>(cell)] =
                    table.sequence(*id).sum;
            }
        }

        bool found_template = false;
        for (size_t index = 0; index < IW7_TEMPLATES.size(); ++index) {
            if (IW7_TEMPLATES[index].entries == template_entries) {
                state.template_index_ = static_cast<int>(index);
                found_template = true;
                break;
            }
        }
        if (!found_template) {
            throw std::runtime_error(
                "start matrix block sums do not match an embedded "
                "IW(7,25) template");
        }
        if (!state.verify_signature_margins()) {
            throw std::runtime_error(
                "start matrix does not have exact row and column "
                "self-correlation");
        }
        state.rebuild_cache();
        return state;
    }

    int template_index() const { return template_index_; }

    std::string_view template_name() const {
        return IW7_TEMPLATES[static_cast<size_t>(template_index_)].name;
    }

    int score() const { return score_; }
    int conflicts() const { return conflicts_; }
    bool solved() const { return score_ == 0; }

    uint8_t choice(int block_row, int block_column) const {
        return choices_[static_cast<size_t>(
            block_row * TEMPLATE_ORDER + block_column)];
    }

    uint8_t support(int block_row, int block_column) const {
        return supports_[static_cast<size_t>(
            block_row * TEMPLATE_ORDER + block_column)];
    }

    AutocorrelationSignature signature(
        int block_row, int block_column) const {
        const size_t cell = static_cast<size_t>(
            block_row * TEMPLATE_ORDER + block_column);
        return {
            supports_[cell],
            first_correlations_[cell],
            second_correlations_[cell]
        };
    }

    const std::vector<uint8_t>& domain(int block_row,
                                       int block_column) const {
        const auto& record =
            IW7_TEMPLATES[static_cast<size_t>(template_index_)];
        const int cell =
            block_row * TEMPLATE_ORDER + block_column;
        return sequence_table().with_signature(
            record.entries[static_cast<size_t>(cell)],
            signature(block_row, block_column));
    }

    const std::array<uint8_t, TEMPLATE_ORDER * TEMPLATE_ORDER>&
    choices() const {
        return choices_;
    }

    int residual(int first, int second, int shift) const {
        const int index = gram_index(first, second, shift);
        return gram_[static_cast<size_t>(index)] -
               target_gram_value(first, second, shift);
    }

    Evaluation evaluate_replacement(int cell,
                                    uint8_t replacement) const {
        const int block_row = cell / TEMPLATE_ORDER;
        const int block_column = cell % TEMPLATE_ORDER;
        const uint8_t old_choice = choices_[static_cast<size_t>(cell)];
        Evaluation result;
        if (old_choice == replacement) return result;

        const auto& table = sequence_table();
        for (int other_row = 0; other_row < TEMPLATE_ORDER; ++other_row) {
            const int first = std::min(block_row, other_row);
            const int second = std::max(block_row, other_row);
            const uint8_t other_choice =
                choice(other_row, block_column);
            for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                int old_contribution = 0;
                int new_contribution = 0;
                if (other_row == block_row) {
                    old_contribution = table.correlation(
                        old_choice, old_choice, shift);
                    new_contribution = table.correlation(
                        replacement, replacement, shift);
                } else if (block_row < other_row) {
                    old_contribution = table.correlation(
                        old_choice, other_choice, shift);
                    new_contribution = table.correlation(
                        replacement, other_choice, shift);
                } else {
                    old_contribution = table.correlation(
                        other_choice, old_choice, shift);
                    new_contribution = table.correlation(
                        other_choice, replacement, shift);
                }

                const int index = gram_index(first, second, shift);
                const int target =
                    target_gram_value(first, second, shift);
                const int old_value = gram_[static_cast<size_t>(index)];
                const int new_value =
                    old_value + new_contribution - old_contribution;
                const int old_residual = old_value - target;
                const int new_residual = new_value - target;
                result.score_delta +=
                    std::abs(new_residual) - std::abs(old_residual);
                result.conflict_delta +=
                    (new_residual != 0 ? 1 : 0) -
                    (old_residual != 0 ? 1 : 0);
            }
        }
        return result;
    }

    void commit_replacement(const CellMove& move) {
        const int block_row = move.cell / TEMPLATE_ORDER;
        const int block_column = move.cell % TEMPLATE_ORDER;
        const uint8_t old_choice =
            choices_[static_cast<size_t>(move.cell)];
        if (old_choice == move.replacement) return;

        const auto& table = sequence_table();
        for (int other_row = 0; other_row < TEMPLATE_ORDER; ++other_row) {
            const int first = std::min(block_row, other_row);
            const int second = std::max(block_row, other_row);
            const uint8_t other_choice =
                choice(other_row, block_column);
            for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                int old_contribution = 0;
                int new_contribution = 0;
                if (other_row == block_row) {
                    old_contribution = table.correlation(
                        old_choice, old_choice, shift);
                    new_contribution = table.correlation(
                        move.replacement, move.replacement, shift);
                } else if (block_row < other_row) {
                    old_contribution = table.correlation(
                        old_choice, other_choice, shift);
                    new_contribution = table.correlation(
                        move.replacement, other_choice, shift);
                } else {
                    old_contribution = table.correlation(
                        other_choice, old_choice, shift);
                    new_contribution = table.correlation(
                        other_choice, move.replacement, shift);
                }
                const int index = gram_index(first, second, shift);
                gram_[static_cast<size_t>(index)] =
                    static_cast<int8_t>(
                        gram_[static_cast<size_t>(index)] +
                        new_contribution - old_contribution);
            }
        }
        choices_[static_cast<size_t>(move.cell)] = move.replacement;
        score_ += move.evaluation.score_delta;
        conflicts_ += move.evaluation.conflict_delta;
    }

    Evaluation evaluate_pair_replacement(
        int first_cell, uint8_t first_replacement,
        int second_cell, uint8_t second_replacement) const {
        const int block_row = first_cell / TEMPLATE_ORDER;
        if (second_cell / TEMPLATE_ORDER != block_row ||
            first_cell == second_cell) {
            throw std::invalid_argument(
                "pair replacement must use distinct cells in one row");
        }
        const int first_column = first_cell % TEMPLATE_ORDER;
        const int second_column = second_cell % TEMPLATE_ORDER;
        const uint8_t old_first =
            choices_[static_cast<size_t>(first_cell)];
        const uint8_t old_second =
            choices_[static_cast<size_t>(second_cell)];
        Evaluation result;
        const auto& table = sequence_table();

        for (int other_row = 0; other_row < TEMPLATE_ORDER; ++other_row) {
            const int first = std::min(block_row, other_row);
            const int second = std::max(block_row, other_row);
            const uint8_t other_first =
                choice(other_row, first_column);
            const uint8_t other_second =
                choice(other_row, second_column);
            for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                int old_contribution = 0;
                int new_contribution = 0;
                if (other_row == block_row) {
                    old_contribution =
                        table.correlation(
                            old_first, old_first, shift) +
                        table.correlation(
                            old_second, old_second, shift);
                    new_contribution =
                        table.correlation(
                            first_replacement, first_replacement,
                            shift) +
                        table.correlation(
                            second_replacement, second_replacement,
                            shift);
                } else if (block_row < other_row) {
                    old_contribution =
                        table.correlation(
                            old_first, other_first, shift) +
                        table.correlation(
                            old_second, other_second, shift);
                    new_contribution =
                        table.correlation(
                            first_replacement, other_first, shift) +
                        table.correlation(
                            second_replacement, other_second, shift);
                } else {
                    old_contribution =
                        table.correlation(
                            other_first, old_first, shift) +
                        table.correlation(
                            other_second, old_second, shift);
                    new_contribution =
                        table.correlation(
                            other_first, first_replacement, shift) +
                        table.correlation(
                            other_second, second_replacement, shift);
                }

                const int index = gram_index(first, second, shift);
                const int target =
                    target_gram_value(first, second, shift);
                const int old_value =
                    gram_[static_cast<size_t>(index)];
                const int new_value =
                    old_value + new_contribution - old_contribution;
                const int old_residual = old_value - target;
                const int new_residual = new_value - target;
                result.score_delta +=
                    std::abs(new_residual) - std::abs(old_residual);
                result.conflict_delta +=
                    (new_residual != 0 ? 1 : 0) -
                    (old_residual != 0 ? 1 : 0);
            }
        }
        return result;
    }

    void commit_pair_replacement(const PairMove& move) {
        CellMove first;
        first.cell = move.first_cell;
        first.replacement = move.first_replacement;
        first.evaluation = evaluate_replacement(
            first.cell, first.replacement);
        commit_replacement(first);

        CellMove second;
        second.cell = move.second_cell;
        second.replacement = move.second_replacement;
        second.evaluation = evaluate_replacement(
            second.cell, second.replacement);
        commit_replacement(second);
    }

    void set_row_choices(
        int block_row,
        const std::array<uint8_t,
                         TEMPLATE_ORDER * TEMPLATE_ORDER>& choices) {
        for (int column = 0; column < TEMPLATE_ORDER; ++column) {
            const int cell = block_row * TEMPLATE_ORDER + column;
            choices_[static_cast<size_t>(cell)] =
                choices[static_cast<size_t>(cell)];
        }
        rebuild_cache();
    }

    void set_column_choices(
        int block_column,
        const std::array<uint8_t, TEMPLATE_ORDER>& choices) {
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            const int cell = row * TEMPLATE_ORDER + block_column;
            choices_[static_cast<size_t>(cell)] =
                choices[static_cast<size_t>(row)];
        }
        rebuild_cache();
    }

    void set_two_column_choices(
        int first_column,
        const std::array<uint8_t, TEMPLATE_ORDER>& first_choices,
        int second_column,
        const std::array<uint8_t, TEMPLATE_ORDER>& second_choices) {
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            choices_[static_cast<size_t>(
                row * TEMPLATE_ORDER + first_column)] =
                first_choices[static_cast<size_t>(row)];
            choices_[static_cast<size_t>(
                row * TEMPLATE_ORDER + second_column)] =
                second_choices[static_cast<size_t>(row)];
        }
        rebuild_cache();
    }

    int worst_row() const {
        int best_row = 0;
        int best_residual = -1;
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            int row_residual = 0;
            for (int other = 0; other < TEMPLATE_ORDER; ++other) {
                const int first = std::min(row, other);
                const int second = std::max(row, other);
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    row_residual +=
                        std::abs(residual(first, second, shift));
                }
            }
            if (row_residual > best_residual) {
                best_residual = row_residual;
                best_row = row;
            }
        }
        return best_row;
    }

    bool row_is_exact(int block_row) const {
        for (int other = 0; other < TEMPLATE_ORDER; ++other) {
            const int first = std::min(block_row, other);
            const int second = std::max(block_row, other);
            for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                if (residual(first, second, shift) != 0) return false;
            }
        }
        return true;
    }

    bool all_rows_are_self_exact() const {
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                if (residual(row, row, shift) != 0) return false;
            }
        }
        return true;
    }

    std::vector<int8_t> expanded_entries() const {
        std::vector<int8_t> result(
            static_cast<size_t>(MATRIX_ORDER * MATRIX_ORDER), 0);
        const auto& table = sequence_table();
        for (int block_row = 0; block_row < TEMPLATE_ORDER; ++block_row) {
            for (int block_column = 0;
                 block_column < TEMPLATE_ORDER; ++block_column) {
                const auto& sequence =
                    table.sequence(choice(block_row, block_column));
                for (int local_row = 0; local_row < BLOCK_ORDER;
                     ++local_row) {
                    for (int local_column = 0;
                         local_column < BLOCK_ORDER; ++local_column) {
                        const int row =
                            block_row * BLOCK_ORDER + local_row;
                        const int column =
                            block_column * BLOCK_ORDER + local_column;
                        const int sequence_index =
                            (local_column - local_row + BLOCK_ORDER) %
                            BLOCK_ORDER;
                        result[static_cast<size_t>(
                            column * MATRIX_ORDER + row)] =
                            sequence.values[
                                static_cast<size_t>(sequence_index)];
                    }
                }
            }
        }
        return result;
    }

    bool verify_template_sums() const {
        const auto& record =
            IW7_TEMPLATES[static_cast<size_t>(template_index_)];
        const auto& table = sequence_table();
        for (int cell = 0; cell < TEMPLATE_ORDER * TEMPLATE_ORDER;
             ++cell) {
            const auto& sequence =
                table.sequence(choices_[static_cast<size_t>(cell)]);
            if (sequence.sum !=
                    record.entries[static_cast<size_t>(cell)] ||
                sequence.signature != signature(
                    cell / TEMPLATE_ORDER,
                    cell % TEMPLATE_ORDER)) {
                return false;
            }
        }
        return true;
    }

    bool verify_support_margins() const {
        for (int index = 0; index < TEMPLATE_ORDER; ++index) {
            int row_sum = 0;
            int column_sum = 0;
            for (int other = 0; other < TEMPLATE_ORDER; ++other) {
                row_sum += support(index, other);
                column_sum += support(other, index);
            }
            if (row_sum != 25 || column_sum != 25) return false;
        }
        return true;
    }

    bool verify_signature_margins() const {
        for (int index = 0; index < TEMPLATE_ORDER; ++index) {
            int row_support = 0;
            int row_first = 0;
            int row_second = 0;
            int column_support = 0;
            int column_first = 0;
            int column_second = 0;
            for (int other = 0; other < TEMPLATE_ORDER; ++other) {
                const auto row_signature = signature(index, other);
                row_support += row_signature.support;
                row_first += row_signature.first;
                row_second += row_signature.second;
                const auto column_signature = signature(other, index);
                column_support += column_signature.support;
                column_first += column_signature.first;
                column_second += column_signature.second;
            }
            if (row_support != 25 || row_first != 0 ||
                row_second != 0 || column_support != 25 ||
                column_first != 0 || column_second != 0) {
                return false;
            }
        }
        return true;
    }

    bool verify_cache() const {
        State rebuilt = *this;
        rebuilt.rebuild_cache();
        return rebuilt.gram_ == gram_ &&
               rebuilt.score_ == score_ &&
               rebuilt.conflicts_ == conflicts_;
    }

    bool verify_weighing() const {
        const auto entries = expanded_entries();
        for (int first = 0; first < MATRIX_ORDER; ++first) {
            for (int second = first; second < MATRIX_ORDER; ++second) {
                int dot = 0;
                for (int column = 0; column < MATRIX_ORDER; ++column) {
                    dot += entries[static_cast<size_t>(
                               column * MATRIX_ORDER + first)] *
                           entries[static_cast<size_t>(
                               column * MATRIX_ORDER + second)];
                }
                const int target = first == second ? 25 : 0;
                if (dot != target) return false;
            }
        }
        return true;
    }

    void rebuild_cache() {
        gram_.fill(0);
        const auto& table = sequence_table();
        for (int first = 0; first < TEMPLATE_ORDER; ++first) {
            for (int second = first; second < TEMPLATE_ORDER; ++second) {
                for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                    const uint8_t first_choice =
                        choice(first, column);
                    const uint8_t second_choice =
                        choice(second, column);
                    for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                        const int index =
                            gram_index(first, second, shift);
                        gram_[static_cast<size_t>(index)] =
                            static_cast<int8_t>(
                                gram_[static_cast<size_t>(index)] +
                                table.correlation(
                                    first_choice, second_choice,
                                    shift));
                    }
                }
            }
        }

        score_ = 0;
        conflicts_ = 0;
        for (int first = 0; first < TEMPLATE_ORDER; ++first) {
            for (int second = first; second < TEMPLATE_ORDER; ++second) {
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    const int value = residual(first, second, shift);
                    score_ += std::abs(value);
                    conflicts_ += value != 0 ? 1 : 0;
                }
            }
        }
    }

private:
    template <class Rng>
    bool initialize_signature_pattern(const TemplateRecord& record,
                                      Rng& rng) {
        using SignatureRow =
            std::array<AutocorrelationSignature, TEMPLATE_ORDER>;
        struct SignatureSum {
            int support = 0;
            int first = 0;
            int second = 0;
        };

        auto encode = [](const SignatureSum& sum) {
            return static_cast<uint32_t>(sum.support) |
                   (static_cast<uint32_t>(sum.first + 35) << 6) |
                   (static_cast<uint32_t>(sum.second + 35) << 13);
        };
        auto decode = [](uint32_t key) {
            SignatureSum sum;
            sum.support = static_cast<int>(key & 63U);
            sum.first =
                static_cast<int>((key >> 6) & 127U) - 35;
            sum.second =
                static_cast<int>((key >> 13) & 127U) - 35;
            return sum;
        };
        auto in_range = [](const SignatureSum& sum) {
            return sum.support >= 0 && sum.support <= 35 &&
                   sum.first >= -35 && sum.first <= 35 &&
                   sum.second >= -35 && sum.second <= 35;
        };

        const auto& table = sequence_table();
        std::array<std::vector<SignatureRow>, TEMPLATE_ORDER>
            row_candidates;
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            SignatureRow current{};
            auto generate = [&record, &table, &row_candidates, &current,
                             row](int column, SignatureSum total,
                                  auto&& generate_self) -> void {
                if (column == TEMPLATE_ORDER) {
                    if (total.support == 25 &&
                        total.first == 0 &&
                        total.second == 0) {
                        row_candidates[static_cast<size_t>(row)]
                            .push_back(current);
                    }
                    return;
                }
                const int cell = row * TEMPLATE_ORDER + column;
                const int sum =
                    record.entries[static_cast<size_t>(cell)];
                for (const auto& signature :
                     table.signatures_for_sum(sum)) {
                    SignatureSum next{
                        total.support + signature.support,
                        total.first + signature.first,
                        total.second + signature.second
                    };
                    const int remaining =
                        TEMPLATE_ORDER - column - 1;
                    if (next.support > 25 ||
                        next.support + BLOCK_ORDER * remaining < 25 ||
                        std::abs(next.first) >
                            BLOCK_ORDER * remaining ||
                        std::abs(next.second) >
                            BLOCK_ORDER * remaining) {
                        continue;
                    }
                    current[static_cast<size_t>(column)] = signature;
                    generate_self(
                        column + 1, next, generate_self);
                }
            };
            generate(0, SignatureSum{}, generate);
            if (row_candidates[static_cast<size_t>(row)].empty())
                return false;
            std::shuffle(
                row_candidates[static_cast<size_t>(row)].begin(),
                row_candidates[static_cast<size_t>(row)].end(), rng);
        }

        std::array<int, TEMPLATE_ORDER> row_order{};
        for (int row = 0; row < TEMPLATE_ORDER; ++row)
            row_order[static_cast<size_t>(row)] = row;
        std::sort(
            row_order.begin(), row_order.end(),
            [&row_candidates](int first, int second) {
                return row_candidates[static_cast<size_t>(first)].size() <
                       row_candidates[static_cast<size_t>(second)].size();
            });

        using ReachableSums = std::array<
            std::array<std::unordered_set<uint32_t>, TEMPLATE_ORDER>,
            TEMPLATE_ORDER + 1>;
        ReachableSums reachable;
        for (int column = 0; column < TEMPLATE_ORDER; ++column) {
            reachable[TEMPLATE_ORDER][static_cast<size_t>(column)]
                .insert(encode(SignatureSum{}));
        }
        for (int depth = TEMPLATE_ORDER - 1; depth >= 0; --depth) {
            const int row = row_order[static_cast<size_t>(depth)];
            for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                const int cell = row * TEMPLATE_ORDER + column;
                const int sum =
                    record.entries[static_cast<size_t>(cell)];
                auto& destination =
                    reachable[static_cast<size_t>(depth)]
                             [static_cast<size_t>(column)];
                for (const auto& signature :
                     table.signatures_for_sum(sum)) {
                    for (uint32_t tail_key :
                         reachable[static_cast<size_t>(depth + 1)]
                                  [static_cast<size_t>(column)]) {
                        const SignatureSum tail = decode(tail_key);
                        const SignatureSum combined{
                            tail.support + signature.support,
                            tail.first + signature.first,
                            tail.second + signature.second
                        };
                        if (in_range(combined))
                            destination.insert(encode(combined));
                    }
                }
            }
        }

        std::array<SignatureSum, TEMPLATE_ORDER> remaining;
        for (SignatureSum& value : remaining)
            value = { 25, 0, 0 };
        auto assign = [this, &row_candidates, &row_order, &reachable,
                       &remaining, &encode, &in_range](
                          int depth, auto&& assign_self) -> bool {
            if (depth == TEMPLATE_ORDER) {
                return std::all_of(
                    remaining.begin(), remaining.end(),
                    [](const SignatureSum& value) {
                        return value.support == 0 &&
                               value.first == 0 &&
                               value.second == 0;
                    });
            }
            const int row = row_order[static_cast<size_t>(depth)];
            for (const SignatureRow& candidate :
                 row_candidates[static_cast<size_t>(row)]) {
                bool feasible = true;
                for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                    const auto& signature =
                        candidate[static_cast<size_t>(column)];
                    const SignatureSum next{
                        remaining[static_cast<size_t>(column)].support -
                            signature.support,
                        remaining[static_cast<size_t>(column)].first -
                            signature.first,
                        remaining[static_cast<size_t>(column)].second -
                            signature.second
                    };
                    if (!in_range(next) ||
                        reachable[static_cast<size_t>(depth + 1)]
                                 [static_cast<size_t>(column)]
                                     .find(encode(next)) ==
                            reachable[static_cast<size_t>(depth + 1)]
                                     [static_cast<size_t>(column)]
                                         .end()) {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible) continue;
                const auto saved = remaining;
                for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                    const auto& signature =
                        candidate[static_cast<size_t>(column)];
                    auto& value =
                        remaining[static_cast<size_t>(column)];
                    value.support -= signature.support;
                    value.first -= signature.first;
                    value.second -= signature.second;
                    const size_t cell = static_cast<size_t>(
                        row * TEMPLATE_ORDER + column);
                    supports_[cell] = signature.support;
                    first_correlations_[cell] = signature.first;
                    second_correlations_[cell] = signature.second;
                }
                if (assign_self(depth + 1, assign_self)) return true;
                remaining = saved;
            }
            return false;
        };
        return assign(0, assign);
    }

    template <class Rng>
    bool initialize_support_pattern(const TemplateRecord& record,
                                    Rng& rng) {
        using SupportRow = std::array<uint8_t, TEMPLATE_ORDER>;
        std::array<std::vector<SupportRow>, TEMPLATE_ORDER>
            row_candidates;
        const auto& table = sequence_table();

        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            SupportRow current{};
            auto generate = [&record, &table, &row_candidates, &current,
                             row](int column, int total,
                                  auto&& generate_self) -> void {
                if (column == TEMPLATE_ORDER) {
                    if (total == 25) {
                        row_candidates[static_cast<size_t>(row)]
                            .push_back(current);
                    }
                    return;
                }
                const int cell = row * TEMPLATE_ORDER + column;
                const int sum =
                    record.entries[static_cast<size_t>(cell)];
                for (int support = 0;
                     support <= BLOCK_ORDER; ++support) {
                    if (table.with_sum_support(sum, support).empty())
                        continue;
                    const int next_total = total + support;
                    if (next_total > 25) continue;
                    const int remaining =
                        TEMPLATE_ORDER - column - 1;
                    if (next_total + BLOCK_ORDER * remaining < 25)
                        continue;
                    current[static_cast<size_t>(column)] =
                        static_cast<uint8_t>(support);
                    generate_self(
                        column + 1, next_total, generate_self);
                }
            };
            generate(0, 0, generate);
            if (row_candidates[static_cast<size_t>(row)].empty())
                return false;
            std::shuffle(
                row_candidates[static_cast<size_t>(row)].begin(),
                row_candidates[static_cast<size_t>(row)].end(), rng);
        }

        std::array<int, TEMPLATE_ORDER> row_order{};
        for (int row = 0; row < TEMPLATE_ORDER; ++row)
            row_order[static_cast<size_t>(row)] = row;
        std::sort(
            row_order.begin(), row_order.end(),
            [&row_candidates](int first, int second) {
                return row_candidates[static_cast<size_t>(first)].size() <
                       row_candidates[static_cast<size_t>(second)].size();
            });

        using ReachableSums =
            std::array<std::array<
                           std::array<bool, MATRIX_ORDER + 1>,
                           TEMPLATE_ORDER>,
                       TEMPLATE_ORDER + 1>;
        ReachableSums reachable{};
        for (int column = 0; column < TEMPLATE_ORDER; ++column) {
            reachable[TEMPLATE_ORDER][static_cast<size_t>(column)][0] =
                true;
        }
        for (int depth = TEMPLATE_ORDER - 1; depth >= 0; --depth) {
            const int row = row_order[static_cast<size_t>(depth)];
            for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                const int cell = row * TEMPLATE_ORDER + column;
                const int sum =
                    record.entries[static_cast<size_t>(cell)];
                for (int support = 0;
                     support <= BLOCK_ORDER; ++support) {
                    if (table.with_sum_support(sum, support).empty())
                        continue;
                    for (int tail = 0;
                         tail + support <= MATRIX_ORDER; ++tail) {
                        if (reachable[static_cast<size_t>(depth + 1)]
                                     [static_cast<size_t>(column)]
                                     [static_cast<size_t>(tail)]) {
                            reachable[static_cast<size_t>(depth)]
                                     [static_cast<size_t>(column)]
                                     [static_cast<size_t>(
                                         tail + support)] = true;
                        }
                    }
                }
            }
        }

        std::array<int, TEMPLATE_ORDER> column_remaining{};
        column_remaining.fill(25);
        auto assign = [this, &row_candidates, &row_order, &reachable,
                       &column_remaining](int depth,
                                          auto&& assign_self) -> bool {
            if (depth == TEMPLATE_ORDER) {
                return std::all_of(
                    column_remaining.begin(), column_remaining.end(),
                    [](int remaining) { return remaining == 0; });
            }
            const int row = row_order[static_cast<size_t>(depth)];
            for (const SupportRow& candidate :
                 row_candidates[static_cast<size_t>(row)]) {
                bool feasible = true;
                for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                    const int next =
                        column_remaining[static_cast<size_t>(column)] -
                        candidate[static_cast<size_t>(column)];
                    if (next < 0 ||
                        !reachable[static_cast<size_t>(depth + 1)]
                                  [static_cast<size_t>(column)]
                                  [static_cast<size_t>(next)]) {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible) continue;
                for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                    const int cell = row * TEMPLATE_ORDER + column;
                    const uint8_t support =
                        candidate[static_cast<size_t>(column)];
                    supports_[static_cast<size_t>(cell)] = support;
                    column_remaining[static_cast<size_t>(column)] -=
                        support;
                }
                if (assign_self(depth + 1, assign_self)) return true;
                for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                    column_remaining[static_cast<size_t>(column)] +=
                        candidate[static_cast<size_t>(column)];
                }
            }
            return false;
        };
        return assign(0, assign);
    }

    template <class Rng>
    bool initialize_exact_row(const TemplateRecord& record, int block_row,
                              Rng& rng) {
        struct Signature {
            int norm = 0;
            int first = 0;
            int second = 0;
        };

        const auto& table = sequence_table();
        std::array<std::vector<uint8_t>, TEMPLATE_ORDER> options;
        for (int column = 0; column < TEMPLATE_ORDER; ++column) {
            const int cell = block_row * TEMPLATE_ORDER + column;
            const int sum = record.entries[static_cast<size_t>(cell)];
            options[static_cast<size_t>(column)] =
                table.with_sum_support(
                    sum, supports_[static_cast<size_t>(cell)]);
            std::shuffle(options[static_cast<size_t>(column)].begin(),
                         options[static_cast<size_t>(column)].end(), rng);
        }

        std::unordered_map<uint64_t, bool> dead;
        auto encode = [](int column, const Signature& signature) {
            const uint64_t norm =
                static_cast<uint64_t>(signature.norm);
            const uint64_t first =
                static_cast<uint64_t>(signature.first + 36);
            const uint64_t second =
                static_cast<uint64_t>(signature.second + 36);
            return static_cast<uint64_t>(column) |
                   (norm << 4) | (first << 10) | (second << 17);
        };

        auto search = [this, &options, &table, block_row, &dead,
                       &encode](int column, Signature signature,
                                auto&& search_self) -> bool {
            if (signature.norm > 25) return false;
            if (column == TEMPLATE_ORDER) {
                return signature.norm == 25 &&
                       signature.first == 0 &&
                       signature.second == 0;
            }
            const int remaining = TEMPLATE_ORDER - column;
            if (signature.norm + BLOCK_ORDER * remaining < 25)
                return false;
            const uint64_t key = encode(column, signature);
            if (dead.find(key) != dead.end()) return false;

            const int cell = block_row * TEMPLATE_ORDER + column;
            for (uint8_t candidate :
                 options[static_cast<size_t>(column)]) {
                Signature next = signature;
                next.norm +=
                    table.correlation(candidate, candidate, 0);
                next.first +=
                    table.correlation(candidate, candidate, 1);
                next.second +=
                    table.correlation(candidate, candidate, 2);
                choices_[static_cast<size_t>(cell)] = candidate;
                if (search_self(column + 1, next, search_self))
                    return true;
            }
            dead.emplace(key, true);
            return false;
        };

        return search(0, Signature{}, search);
    }

    int template_index_ = 0;
    std::array<uint8_t, TEMPLATE_ORDER * TEMPLATE_ORDER> choices_{};
    std::array<uint8_t, TEMPLATE_ORDER * TEMPLATE_ORDER> supports_{};
    std::array<int8_t, TEMPLATE_ORDER * TEMPLATE_ORDER>
        first_correlations_{};
    std::array<int8_t, TEMPLATE_ORDER * TEMPLATE_ORDER>
        second_correlations_{};
    std::array<int8_t, GRAM_SIZE> gram_{};
    int score_ = 0;
    int conflicts_ = 0;
};

struct ConstraintKey {
    std::array<int8_t, CONSTRAINT_SIZE> values{};

    bool operator==(const ConstraintKey&) const = default;
};

struct ConstraintHash {
    size_t operator()(const ConstraintKey& key) const {
        size_t hash = static_cast<size_t>(1469598103934665603ULL);
        for (int8_t value : key.values) {
            hash ^= static_cast<uint8_t>(value);
            hash *= static_cast<size_t>(1099511628211ULL);
        }
        return hash;
    }
};

inline uint64_t combination_count(
    const std::vector<int>& columns,
    const std::array<std::vector<uint8_t>, TEMPLATE_ORDER>& options,
    uint64_t limit) {
    uint64_t count = 1;
    for (int column : columns) {
        const uint64_t factor =
            options[static_cast<size_t>(column)].size();
        if (factor == 0 || count > limit / factor) return limit + 1;
        count *= factor;
    }
    return count;
}

inline bool try_exact_column_completion(State& state,
                                        uint64_t max_nodes) {
    if (max_nodes == 0) return false;

    using Polynomial = std::array<int8_t, BLOCK_ORDER>;
    struct ColumnProblem {
        int column = 0;
        std::array<Polynomial, PAIR_COUNT> required{};
        std::array<std::vector<uint8_t>, TEMPLATE_ORDER> domains;
        uint64_t domain_product = 0;
    };

    const auto& table = sequence_table();
    std::vector<ColumnProblem> problems;
    problems.reserve(TEMPLATE_ORDER);
    for (int free_column = 0;
         free_column < TEMPLATE_ORDER; ++free_column) {
        ColumnProblem problem;
        problem.column = free_column;
        for (int first = 0; first < TEMPLATE_ORDER; ++first) {
            for (int second = first;
                 second < TEMPLATE_ORDER; ++second) {
                Polynomial& required = problem.required[
                    static_cast<size_t>(pair_index(first, second))];
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    int value =
                        target_gram_value(first, second, shift);
                    for (int column = 0;
                         column < TEMPLATE_ORDER; ++column) {
                        if (column == free_column) continue;
                        value -= table.correlation(
                            state.choice(first, column),
                            state.choice(second, column), shift);
                    }
                    required[static_cast<size_t>(shift)] =
                        static_cast<int8_t>(value);
                }
            }
        }

        bool feasible = true;
        uint64_t product = 1;
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            const Polynomial& required = problem.required[
                static_cast<size_t>(pair_index(row, row))];
            auto& domain =
                problem.domains[static_cast<size_t>(row)];
            for (uint8_t candidate :
                 state.domain(row, free_column)) {
                bool matches = true;
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    if (table.correlation(
                            candidate, candidate, shift) !=
                        required[static_cast<size_t>(shift)]) {
                        matches = false;
                        break;
                    }
                }
                if (matches) domain.push_back(candidate);
            }
            if (domain.empty()) {
                feasible = false;
                break;
            }
            if (product >
                std::numeric_limits<uint64_t>::max() / domain.size()) {
                product = std::numeric_limits<uint64_t>::max();
            } else {
                product *= domain.size();
            }
        }
        if (feasible) {
            problem.domain_product = product;
            problems.push_back(std::move(problem));
        }
    }
    std::sort(
        problems.begin(), problems.end(),
        [](const ColumnProblem& first, const ColumnProblem& second) {
            return first.domain_product < second.domain_product;
        });

    for (const ColumnProblem& problem : problems) {
        std::array<uint8_t, TEMPLATE_ORDER> assignment{};
        std::array<bool, TEMPLATE_ORDER> assigned{};
        uint64_t nodes = 0;
        bool stopped = false;

        auto compatible = [&problem, &table, &assignment, &assigned](
                              int row, uint8_t candidate) {
            for (int other = 0; other < TEMPLATE_ORDER; ++other) {
                if (other == row ||
                    !assigned[static_cast<size_t>(other)]) {
                    continue;
                }
                const int first = std::min(row, other);
                const int second = std::max(row, other);
                const Polynomial& required = problem.required[
                    static_cast<size_t>(pair_index(first, second))];
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    const int value =
                        row < other
                            ? table.correlation(
                                  candidate,
                                  assignment[static_cast<size_t>(other)],
                                  shift)
                            : table.correlation(
                                  assignment[static_cast<size_t>(other)],
                                  candidate, shift);
                    if (value != required[static_cast<size_t>(shift)])
                        return false;
                }
            }
            return true;
        };

        auto search = [&problem, &assignment, &assigned, &nodes,
                       max_nodes, &stopped, &compatible](
                          int depth, auto&& search_self) -> bool {
            if (depth == TEMPLATE_ORDER) return true;
            if (nodes >= max_nodes ||
                g_stop.load(std::memory_order_relaxed)) {
                stopped = true;
                return false;
            }

            int selected_row = -1;
            size_t selected_count =
                std::numeric_limits<size_t>::max();
            for (int row = 0; row < TEMPLATE_ORDER; ++row) {
                if (assigned[static_cast<size_t>(row)]) continue;
                size_t count = 0;
                for (uint8_t candidate :
                     problem.domains[static_cast<size_t>(row)]) {
                    count += compatible(row, candidate) ? 1U : 0U;
                }
                if (count == 0) return false;
                if (count < selected_count) {
                    selected_count = count;
                    selected_row = row;
                }
            }

            assigned[static_cast<size_t>(selected_row)] = true;
            for (uint8_t candidate :
                 problem.domains[static_cast<size_t>(selected_row)]) {
                if (!compatible(selected_row, candidate)) continue;
                ++nodes;
                assignment[static_cast<size_t>(selected_row)] =
                    candidate;
                if (search_self(depth + 1, search_self)) return true;
                if (stopped) break;
            }
            assigned[static_cast<size_t>(selected_row)] = false;
            return false;
        };

        if (!search(0, search)) continue;
        State candidate = state;
        candidate.set_column_choices(problem.column, assignment);
        if (!candidate.solved() || !candidate.verify_template_sums() ||
            !candidate.verify_signature_margins()) {
            throw std::runtime_error(
                "exact column completion verification failed");
        }
        state = std::move(candidate);
        return true;
    }
    return false;
}

inline bool try_exact_two_column_completion(State& state,
                                            uint64_t max_nodes) {
    if (max_nodes == 0) return false;

    using Polynomial = std::array<int8_t, BLOCK_ORDER>;
    struct WordPair {
        uint8_t first = 0;
        uint8_t second = 0;
    };
    struct ColumnPairProblem {
        int first_column = 0;
        int second_column = 0;
        std::array<Polynomial, PAIR_COUNT> required{};
        std::array<std::vector<WordPair>, TEMPLATE_ORDER> domains;
        uint64_t domain_product = 0;
    };

    const auto& table = sequence_table();
    std::vector<ColumnPairProblem> problems;
    problems.reserve(PAIR_COUNT - TEMPLATE_ORDER);
    for (int first_column = 0;
         first_column < TEMPLATE_ORDER; ++first_column) {
        for (int second_column = first_column + 1;
             second_column < TEMPLATE_ORDER; ++second_column) {
            ColumnPairProblem problem;
            problem.first_column = first_column;
            problem.second_column = second_column;
            for (int first = 0; first < TEMPLATE_ORDER; ++first) {
                for (int second = first;
                     second < TEMPLATE_ORDER; ++second) {
                    Polynomial& required = problem.required[
                        static_cast<size_t>(
                            pair_index(first, second))];
                    for (int shift = 0;
                         shift < BLOCK_ORDER; ++shift) {
                        int value =
                            target_gram_value(first, second, shift);
                        for (int column = 0;
                             column < TEMPLATE_ORDER; ++column) {
                            if (column == first_column ||
                                column == second_column) {
                                continue;
                            }
                            value -= table.correlation(
                                state.choice(first, column),
                                state.choice(second, column), shift);
                        }
                        required[static_cast<size_t>(shift)] =
                            static_cast<int8_t>(value);
                    }
                }
            }

            bool feasible = true;
            uint64_t product = 1;
            for (int row = 0; row < TEMPLATE_ORDER; ++row) {
                const Polynomial& required = problem.required[
                    static_cast<size_t>(pair_index(row, row))];
                auto& domain =
                    problem.domains[static_cast<size_t>(row)];
                for (uint8_t first :
                     state.domain(row, first_column)) {
                    for (uint8_t second :
                         state.domain(row, second_column)) {
                        bool matches = true;
                        for (int shift = 0;
                             shift < BLOCK_ORDER; ++shift) {
                            const int value =
                                table.correlation(
                                    first, first, shift) +
                                table.correlation(
                                    second, second, shift);
                            if (value != required[
                                             static_cast<size_t>(
                                                 shift)]) {
                                matches = false;
                                break;
                            }
                        }
                        if (matches) {
                            domain.push_back({ first, second });
                        }
                    }
                }
                if (domain.empty()) {
                    feasible = false;
                    break;
                }
                if (product >
                    std::numeric_limits<uint64_t>::max() /
                        domain.size()) {
                    product = std::numeric_limits<uint64_t>::max();
                } else {
                    product *= domain.size();
                }
            }
            if (feasible) {
                problem.domain_product = product;
                problems.push_back(std::move(problem));
            }
        }
    }
    std::sort(
        problems.begin(), problems.end(),
        [](const ColumnPairProblem& first,
           const ColumnPairProblem& second) {
            return first.domain_product < second.domain_product;
        });

    for (const ColumnPairProblem& problem : problems) {
        std::array<WordPair, TEMPLATE_ORDER> assignment{};
        std::array<bool, TEMPLATE_ORDER> assigned{};
        uint64_t nodes = 0;
        bool stopped = false;

        auto compatible = [&problem, &table, &assignment, &assigned](
                              int row, const WordPair& candidate) {
            for (int other = 0; other < TEMPLATE_ORDER; ++other) {
                if (other == row ||
                    !assigned[static_cast<size_t>(other)]) {
                    continue;
                }
                const int first = std::min(row, other);
                const int second = std::max(row, other);
                const Polynomial& required = problem.required[
                    static_cast<size_t>(pair_index(first, second))];
                const WordPair& other_pair =
                    assignment[static_cast<size_t>(other)];
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    const int value =
                        row < other
                            ? table.correlation(
                                  candidate.first,
                                  other_pair.first, shift) +
                                  table.correlation(
                                      candidate.second,
                                      other_pair.second, shift)
                            : table.correlation(
                                  other_pair.first,
                                  candidate.first, shift) +
                                  table.correlation(
                                      other_pair.second,
                                      candidate.second, shift);
                    if (value != required[static_cast<size_t>(shift)])
                        return false;
                }
            }
            return true;
        };

        auto search = [&problem, &assignment, &assigned, &nodes,
                       max_nodes, &stopped, &compatible](
                          int depth, auto&& search_self) -> bool {
            if (depth == TEMPLATE_ORDER) return true;
            if (nodes >= max_nodes ||
                g_stop.load(std::memory_order_relaxed)) {
                stopped = true;
                return false;
            }

            int selected_row = -1;
            size_t selected_count =
                std::numeric_limits<size_t>::max();
            for (int row = 0; row < TEMPLATE_ORDER; ++row) {
                if (assigned[static_cast<size_t>(row)]) continue;
                size_t count = 0;
                for (const WordPair& candidate :
                     problem.domains[static_cast<size_t>(row)]) {
                    count += compatible(row, candidate) ? 1U : 0U;
                }
                if (count == 0) return false;
                if (count < selected_count) {
                    selected_count = count;
                    selected_row = row;
                }
            }

            assigned[static_cast<size_t>(selected_row)] = true;
            for (const WordPair& candidate :
                 problem.domains[static_cast<size_t>(selected_row)]) {
                if (!compatible(selected_row, candidate)) continue;
                ++nodes;
                assignment[static_cast<size_t>(selected_row)] =
                    candidate;
                if (search_self(depth + 1, search_self)) return true;
                if (stopped) break;
            }
            assigned[static_cast<size_t>(selected_row)] = false;
            return false;
        };

        if (!search(0, search)) continue;
        std::array<uint8_t, TEMPLATE_ORDER> first_choices{};
        std::array<uint8_t, TEMPLATE_ORDER> second_choices{};
        for (int row = 0; row < TEMPLATE_ORDER; ++row) {
            first_choices[static_cast<size_t>(row)] =
                assignment[static_cast<size_t>(row)].first;
            second_choices[static_cast<size_t>(row)] =
                assignment[static_cast<size_t>(row)].second;
        }
        State candidate = state;
        candidate.set_two_column_choices(
            problem.first_column, first_choices,
            problem.second_column, second_choices);
        if (!candidate.solved() || !candidate.verify_template_sums() ||
            !candidate.verify_signature_margins()) {
            throw std::runtime_error(
                "exact two-column completion verification failed");
        }
        state = std::move(candidate);
        return true;
    }
    return false;
}

inline bool try_exact_row_completion(State& state, int block_row,
                                     uint64_t max_side_combinations) {
    if (max_side_combinations == 0) return false;

    const auto& table = sequence_table();
    std::array<std::vector<uint8_t>, TEMPLATE_ORDER> options;
    for (int column = 0; column < TEMPLATE_ORDER; ++column) {
        options[static_cast<size_t>(column)] =
            state.domain(block_row, column);
    }

    std::vector<int> build_columns;
    std::vector<int> probe_columns;
    uint64_t build_count = 0;
    uint64_t best_largest = std::numeric_limits<uint64_t>::max();
    for (int mask = 1; mask < (1 << TEMPLATE_ORDER) - 1; ++mask) {
        std::vector<int> first;
        std::vector<int> second;
        for (int column = 0; column < TEMPLATE_ORDER; ++column) {
            ((mask & (1 << column)) != 0 ? first : second)
                .push_back(column);
        }
        const uint64_t first_count = combination_count(
            first, options, max_side_combinations);
        const uint64_t second_count = combination_count(
            second, options, max_side_combinations);
        const uint64_t largest = std::max(first_count, second_count);
        if (largest > max_side_combinations || largest >= best_largest)
            continue;
        best_largest = largest;
        if (first_count <= second_count) {
            build_columns = std::move(first);
            probe_columns = std::move(second);
            build_count = first_count;
        } else {
            build_columns = std::move(second);
            probe_columns = std::move(first);
            build_count = second_count;
        }
    }
    if (build_columns.empty() || probe_columns.empty()) return false;

    using Contributions = std::array<
        std::array<ConstraintKey, SEQUENCE_COUNT>, TEMPLATE_ORDER>;
    Contributions contributions{};
    for (int column = 0; column < TEMPLATE_ORDER; ++column) {
        for (uint8_t candidate :
             options[static_cast<size_t>(column)]) {
            ConstraintKey& key =
                contributions[static_cast<size_t>(column)]
                             [static_cast<size_t>(candidate)];
            for (int other = 0; other < TEMPLATE_ORDER; ++other) {
                const uint8_t other_choice =
                    state.choice(other, column);
                for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                    int value = 0;
                    if (other == block_row) {
                        value = table.correlation(
                            candidate, candidate, shift);
                    } else if (block_row < other) {
                        value = table.correlation(
                            candidate, other_choice, shift);
                    } else {
                        value = table.correlation(
                            other_choice, candidate, shift);
                    }
                    key.values[static_cast<size_t>(
                        other * BLOCK_ORDER + shift)] =
                        static_cast<int8_t>(value);
                }
            }
        }
    }

    using ChoiceArray =
        std::array<uint8_t, TEMPLATE_ORDER * TEMPLATE_ORDER>;
    std::unordered_map<ConstraintKey, ChoiceArray, ConstraintHash>
        partial_rows;
    partial_rows.reserve(static_cast<size_t>(build_count));

    ConstraintKey running;
    ChoiceArray selection = state.choices();
    uint64_t visited = 0;
    bool stopped = false;
    auto enumerate = [&options, &contributions, &running, &selection,
                      &visited, &stopped, block_row](
                         const std::vector<int>& columns, size_t depth,
                         auto&& enumerate_self, auto&& callback) -> void {
        if (stopped) return;
        if (depth == columns.size()) {
            ++visited;
            if ((visited & 65535ULL) == 0 &&
                g_stop.load(std::memory_order_relaxed)) {
                stopped = true;
                return;
            }
            callback();
            return;
        }

        const int column = columns[depth];
        const size_t cell = static_cast<size_t>(
            block_row * TEMPLATE_ORDER + column);
        for (uint8_t candidate :
             options[static_cast<size_t>(column)]) {
            const ConstraintKey& contribution =
                contributions[static_cast<size_t>(column)]
                             [static_cast<size_t>(candidate)];
            for (int index = 0; index < CONSTRAINT_SIZE; ++index) {
                running.values[static_cast<size_t>(index)] =
                    static_cast<int8_t>(
                        running.values[static_cast<size_t>(index)] +
                        contribution.values[static_cast<size_t>(index)]);
            }
            selection[cell] = candidate;
            enumerate_self(
                columns, depth + 1, enumerate_self, callback);
            for (int index = 0; index < CONSTRAINT_SIZE; ++index) {
                running.values[static_cast<size_t>(index)] =
                    static_cast<int8_t>(
                        running.values[static_cast<size_t>(index)] -
                        contribution.values[static_cast<size_t>(index)]);
            }
            if (stopped) return;
        }
    };

    enumerate(
        build_columns, 0, enumerate,
        [&partial_rows, &running, &selection]() {
            partial_rows.emplace(running, selection);
        });
    if (stopped) return false;

    std::optional<ChoiceArray> solution;
    running.values.fill(0);
    visited = 0;
    enumerate(
        probe_columns, 0, enumerate,
        [&partial_rows, &running, &selection, &solution, &stopped,
         &build_columns, block_row]() {
            ConstraintKey needed;
            for (int index = 0; index < CONSTRAINT_SIZE; ++index) {
                const int target =
                    index == block_row * BLOCK_ORDER ? 25 : 0;
                needed.values[static_cast<size_t>(index)] =
                    static_cast<int8_t>(
                        target -
                        running.values[static_cast<size_t>(index)]);
            }
            const auto match = partial_rows.find(needed);
            if (match == partial_rows.end()) return;
            ChoiceArray combined = selection;
            for (int column = 0; column < TEMPLATE_ORDER; ++column) {
                const int cell = block_row * TEMPLATE_ORDER + column;
                if (std::find(build_columns.begin(),
                              build_columns.end(),
                              column) != build_columns.end()) {
                    combined[static_cast<size_t>(cell)] =
                        match->second[static_cast<size_t>(cell)];
                }
            }
            solution = combined;
            stopped = true;
        });

    if (!solution.has_value()) return false;
    State candidate = state;
    candidate.set_row_choices(block_row, *solution);
    if (!candidate.row_is_exact(block_row) ||
        !candidate.verify_template_sums() ||
        !candidate.verify_signature_margins()) {
        throw std::runtime_error("exact row completion verification failed");
    }
    if (candidate.score() > state.score()) {
        throw std::runtime_error("exact row completion increased the score");
    }
    state = std::move(candidate);
    return true;
}

template <class Rng>
inline int select_row(const State& state, const Params& params, Rng& rng) {
    std::uniform_real_distribution<double> real(0.0, 1.0);
    if (real(rng) < params.target_fraction) return state.worst_row();
    std::uniform_int_distribution<int> row(0, TEMPLATE_ORDER - 1);
    return row(rng);
}

template <class Rng>
inline std::optional<CellMove> random_replacement(
    const State& state, int block_row, Rng& rng,
    uint64_t& candidate_evaluations) {
    std::uniform_int_distribution<int> column_distribution(
        0, TEMPLATE_ORDER - 1);
    for (int attempt = 0; attempt < 2 * TEMPLATE_ORDER; ++attempt) {
        const int column = column_distribution(rng);
        const int cell = block_row * TEMPLATE_ORDER + column;
        const auto& candidates = state.domain(block_row, column);
        if (candidates.size() < 2) continue;
        std::uniform_int_distribution<size_t> choose(
            0, candidates.size() - 2);
        size_t index = choose(rng);
        const uint8_t current =
            state.choices()[static_cast<size_t>(cell)];
        if (candidates[index] == current) ++index;
        CellMove move;
        move.cell = cell;
        move.replacement = candidates[index];
        move.evaluation =
            state.evaluate_replacement(cell, move.replacement);
        ++candidate_evaluations;
        return move;
    }
    return std::nullopt;
}

template <class Rng>
inline std::optional<CellMove> best_replacement(
    const State& state, int block_row, Rng& rng,
    uint64_t& candidate_evaluations) {
    std::optional<CellMove> best;
    uint64_t ties = 0;
    for (int column = 0; column < TEMPLATE_ORDER; ++column) {
        const int cell = block_row * TEMPLATE_ORDER + column;
        const auto& candidates = state.domain(block_row, column);
        const uint8_t current =
            state.choices()[static_cast<size_t>(cell)];
        for (uint8_t candidate : candidates) {
            if (candidate == current) continue;
            CellMove move;
            move.cell = cell;
            move.replacement = candidate;
            move.evaluation =
                state.evaluate_replacement(cell, candidate);
            ++candidate_evaluations;
            if (!best.has_value() ||
                move.evaluation.score_delta <
                    best->evaluation.score_delta) {
                best = move;
                ties = 1;
            } else if (move.evaluation.score_delta ==
                       best->evaluation.score_delta) {
                ++ties;
                std::uniform_int_distribution<uint64_t> replace(1, ties);
                if (replace(rng) == 1) best = move;
            }
        }
    }
    return best;
}

inline bool preserves_pair_autocorrelation(
    uint8_t old_first, uint8_t old_second,
    uint8_t new_first, uint8_t new_second) {
    const auto& table = sequence_table();
    for (int shift = 0; shift <= BLOCK_ORDER / 2; ++shift) {
        const int old_value =
            table.correlation(old_first, old_first, shift) +
            table.correlation(old_second, old_second, shift);
        const int new_value =
            table.correlation(new_first, new_first, shift) +
            table.correlation(new_second, new_second, shift);
        if (old_value != new_value) return false;
    }
    return true;
}

template <class Rng>
inline std::optional<PairMove> random_pair_replacement(
    const State& state, int block_row, Rng& rng,
    uint64_t& candidate_evaluations) {
    std::uniform_int_distribution<int> column_distribution(
        0, TEMPLATE_ORDER - 1);

    for (int attempt = 0; attempt < 2 * TEMPLATE_ORDER; ++attempt) {
        int first_column = column_distribution(rng);
        int second_column = column_distribution(rng);
        if (first_column == second_column) continue;
        if (first_column > second_column)
            std::swap(first_column, second_column);
        const int first_cell =
            block_row * TEMPLATE_ORDER + first_column;
        const int second_cell =
            block_row * TEMPLATE_ORDER + second_column;
        const uint8_t old_first =
            state.choices()[static_cast<size_t>(first_cell)];
        const uint8_t old_second =
            state.choices()[static_cast<size_t>(second_cell)];
        const auto& first_domain =
            state.domain(block_row, first_column);
        const auto& second_domain =
            state.domain(block_row, second_column);

        std::optional<PairMove> selected;
        uint64_t compatible = 0;
        for (uint8_t new_first : first_domain) {
            for (uint8_t new_second : second_domain) {
                if (new_first == old_first &&
                    new_second == old_second) {
                    continue;
                }
                if (!preserves_pair_autocorrelation(
                        old_first, old_second,
                        new_first, new_second)) {
                    continue;
                }
                ++compatible;
                std::uniform_int_distribution<uint64_t> choose(
                    1, compatible);
                if (choose(rng) != 1) continue;
                PairMove move;
                move.first_cell = first_cell;
                move.second_cell = second_cell;
                move.first_replacement = new_first;
                move.second_replacement = new_second;
                selected = move;
            }
        }
        if (!selected.has_value()) continue;
        selected->evaluation = state.evaluate_pair_replacement(
            selected->first_cell, selected->first_replacement,
            selected->second_cell, selected->second_replacement);
        ++candidate_evaluations;
        return selected;
    }
    return std::nullopt;
}

template <class Rng>
inline std::optional<PairMove> best_pair_replacement(
    const State& state, int block_row, int pair_samples, Rng& rng,
    uint64_t& candidate_evaluations) {
    std::uniform_int_distribution<int> column_distribution(
        0, TEMPLATE_ORDER - 1);
    std::optional<PairMove> best;
    uint64_t ties = 0;

    for (int sample = 0; sample < pair_samples; ++sample) {
        int first_column = column_distribution(rng);
        int second_column = column_distribution(rng);
        if (first_column == second_column) {
            second_column = (second_column + 1) % TEMPLATE_ORDER;
        }
        if (first_column > second_column)
            std::swap(first_column, second_column);
        const int first_cell =
            block_row * TEMPLATE_ORDER + first_column;
        const int second_cell =
            block_row * TEMPLATE_ORDER + second_column;
        const uint8_t old_first =
            state.choices()[static_cast<size_t>(first_cell)];
        const uint8_t old_second =
            state.choices()[static_cast<size_t>(second_cell)];
        const auto& first_domain =
            state.domain(block_row, first_column);
        const auto& second_domain =
            state.domain(block_row, second_column);

        for (uint8_t new_first : first_domain) {
            for (uint8_t new_second : second_domain) {
                if (new_first == old_first &&
                    new_second == old_second) {
                    continue;
                }
                if (!preserves_pair_autocorrelation(
                        old_first, old_second,
                        new_first, new_second)) {
                    continue;
                }
                PairMove move;
                move.first_cell = first_cell;
                move.second_cell = second_cell;
                move.first_replacement = new_first;
                move.second_replacement = new_second;
                move.evaluation =
                    state.evaluate_pair_replacement(
                        first_cell, new_first,
                        second_cell, new_second);
                ++candidate_evaluations;
                if (!best.has_value() ||
                    move.evaluation.score_delta <
                        best->evaluation.score_delta) {
                    best = move;
                    ties = 1;
                } else if (
                    move.evaluation.score_delta ==
                    best->evaluation.score_delta) {
                    ++ties;
                    std::uniform_int_distribution<uint64_t> replace(
                        1, ties);
                    if (replace(rng) == 1) best = move;
                }
            }
        }
    }
    return best;
}

template <class Rng>
inline double calibrated_temperature(const State& state, Rng& rng) {
    uint64_t evaluations = 0;
    int64_t positive_total = 0;
    int positive_count = 0;
    for (int sample = 0; sample < 256; ++sample) {
        std::uniform_int_distribution<int> row(
            0, TEMPLATE_ORDER - 1);
        const auto move =
            random_pair_replacement(
                state, row(rng), rng, evaluations);
        if (move.has_value() && move->evaluation.score_delta > 0) {
            positive_total += move->evaluation.score_delta;
            ++positive_count;
        }
    }
    if (positive_count == 0) return 1.0;
    const double mean =
        static_cast<double>(positive_total) / positive_count;
    return std::max(0.25, mean / std::log(2.0));
}

inline uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct RunResult {
    State best;
    uint64_t moves = 0;
    uint64_t candidate_evaluations = 0;
    uint64_t restarts = 0;
    uint64_t exact_attempts = 0;
    uint64_t exact_successes = 0;
    double seconds = 0.0;
};

struct SharedSearch {
    explicit SharedSearch(
        std::chrono::steady_clock::time_point start_time)
        : start(start_time) {}

    std::mutex best_mutex;
    std::mutex exact_mutex;
    std::optional<State> best;
    std::atomic<int> best_score{ std::numeric_limits<int>::max() };
    std::atomic<bool> done{ false };
    std::atomic<uint64_t> moves{ 0 };
    std::atomic<uint64_t> candidate_evaluations{ 0 };
    std::atomic<uint64_t> restarts{ 0 };
    std::atomic<uint64_t> exact_attempts{ 0 };
    std::atomic<uint64_t> exact_successes{ 0 };
    std::chrono::steady_clock::time_point start;
    double last_log_seconds = -1.0;
};

inline double elapsed_seconds(const SharedSearch& shared) {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - shared.start)
        .count();
}

inline void publish_best(SharedSearch& shared, const State& state,
                         int worker) {
    if (state.score() >=
        shared.best_score.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lock(shared.best_mutex);
    if (state.score() >=
        shared.best_score.load(std::memory_order_relaxed)) {
        return;
    }
    shared.best = state;
    shared.best_score.store(state.score(), std::memory_order_relaxed);
    const double elapsed = elapsed_seconds(shared);
    if (elapsed - shared.last_log_seconds >= 0.2 || state.solved()) {
        std::cout << "[best] seconds=" << elapsed
                  << " worker=" << worker
                  << " template=" << state.template_index() + 1
                  << ':' << state.template_name()
                  << " score=" << state.score()
                  << " conflicts=" << state.conflicts() << '\n';
        shared.last_log_seconds = elapsed;
    }
    if (state.solved()) {
        shared.done.store(true, std::memory_order_relaxed);
    }
}

inline int scheduled_template(const Params& params, int worker,
                              uint64_t restart) {
    if (params.template_index >= 0) return params.template_index;
    const uint64_t position =
        static_cast<uint64_t>(worker) +
        restart * static_cast<uint64_t>(params.threads);
    return static_cast<int>(position % IW7_TEMPLATES.size());
}

inline void anneal_worker(int worker, uint64_t base_seed,
                          const Params& params,
                          const State* initial_state,
                          SharedSearch& shared) {
    std::mt19937_64 rng(splitmix64(base_seed +
                                  static_cast<uint64_t>(worker)));
    std::uniform_real_distribution<double> real(0.0, 1.0);
    uint64_t restart = 0;
    uint64_t local_moves_total = 0;
    uint64_t local_evaluations_total = 0;

    while (!shared.done.load(std::memory_order_relaxed) &&
           !g_stop.load(std::memory_order_relaxed)) {
        const int template_index =
            scheduled_template(params, worker, restart);
        State state =
            initial_state != nullptr
                ? *initial_state
                : State::random_start(template_index, rng);
        State local_best = state;
        publish_best(shared, state, worker);

        const double initial_temperature =
            params.t_init > 0.0
                ? params.t_init
                : calibrated_temperature(state, rng);
        double temperature =
            std::max(params.t_min, initial_temperature);
        uint64_t moves_without_improvement = 0;
        uint64_t state_moves = 0;
        int last_exact_score = std::numeric_limits<int>::max();

        bool restart_now = false;
        while (!restart_now &&
               !shared.done.load(std::memory_order_relaxed) &&
               !g_stop.load(std::memory_order_relaxed)) {
            const int block_row = select_row(state, params, rng);
            const bool use_pair =
                real(rng) < params.pair_fraction;
            int delta = 0;
            if (use_pair) {
                std::optional<PairMove> move;
                if (real(rng) < params.greedy_fraction) {
                    move = best_pair_replacement(
                        state, block_row, params.pair_samples, rng,
                        local_evaluations_total);
                } else {
                    move = random_pair_replacement(
                        state, block_row, rng,
                        local_evaluations_total);
                }
                if (!move.has_value()) {
                    restart_now = true;
                    continue;
                }
                delta = move->evaluation.score_delta;
                const bool accept =
                    delta <= 0 ||
                    real(rng) <
                        std::exp(-static_cast<double>(delta) /
                                 temperature);
                if (accept) state.commit_pair_replacement(*move);
            } else {
                std::optional<CellMove> move;
                if (real(rng) < params.greedy_fraction) {
                    move = best_replacement(
                        state, block_row, rng,
                        local_evaluations_total);
                } else {
                    move = random_replacement(
                        state, block_row, rng,
                        local_evaluations_total);
                }
                if (!move.has_value()) {
                    restart_now = true;
                    continue;
                }
                delta = move->evaluation.score_delta;
                const bool accept =
                    delta <= 0 ||
                    real(rng) <
                        std::exp(-static_cast<double>(delta) /
                                 temperature);
                if (accept) state.commit_replacement(*move);
            }
            ++local_moves_total;
            ++state_moves;
            ++moves_without_improvement;

            if (state.score() < local_best.score()) {
                local_best = state;
                moves_without_improvement = 0;
                publish_best(shared, state, worker);
            }
            if (state.solved()) break;

            if (params.exact_threshold >= 0 &&
                state.score() <= params.exact_threshold &&
                state.score() < last_exact_score &&
                (params.exact_max_side > 0 ||
                 params.exact_column_nodes > 0 ||
                 params.exact_two_column_nodes > 0)) {
                std::unique_lock<std::mutex> exact_lock(
                    shared.exact_mutex, std::try_to_lock);
                if (exact_lock.owns_lock()) {
                    last_exact_score = state.score();
                    shared.exact_attempts.fetch_add(
                        1, std::memory_order_relaxed);
                    const bool completed_column =
                        try_exact_column_completion(
                            state, params.exact_column_nodes);
                    const bool completed_two_columns =
                        !completed_column &&
                        try_exact_two_column_completion(
                            state,
                            params.exact_two_column_nodes);
                    const bool completed_row =
                        !completed_column &&
                        !completed_two_columns &&
                        try_exact_row_completion(
                            state, state.worst_row(),
                            params.exact_max_side);
                    if (completed_column || completed_two_columns ||
                        completed_row) {
                        shared.exact_successes.fetch_add(
                            1, std::memory_order_relaxed);
                        if (state.score() < local_best.score()) {
                            local_best = state;
                            moves_without_improvement = 0;
                            publish_best(shared, state, worker);
                        }
                    }
                }
            }

            if (local_moves_total %
                    static_cast<uint64_t>(params.moves_per_cool) ==
                0) {
                temperature =
                    std::max(params.t_min,
                             temperature * params.cooling);
            }

            if (moves_without_improvement == params.stuck_threshold ||
                moves_without_improvement ==
                    2 * params.stuck_threshold) {
                state = local_best;
                temperature = std::max(
                    params.t_min,
                    initial_temperature * params.reheat);
            } else if (moves_without_improvement >=
                       3 * params.stuck_threshold) {
                restart_now = true;
            }
            if (params.restart_moves > 0 &&
                state_moves >= params.restart_moves) {
                restart_now = true;
            }

            if ((local_moves_total & 4095ULL) == 0) {
                shared.moves.fetch_add(4096,
                                       std::memory_order_relaxed);
                local_moves_total -= 4096;
                shared.candidate_evaluations.fetch_add(
                    local_evaluations_total,
                    std::memory_order_relaxed);
                local_evaluations_total = 0;
                if (params.max_seconds > 0.0 &&
                    elapsed_seconds(shared) >= params.max_seconds) {
                    shared.done.store(true, std::memory_order_relaxed);
                }
            }
        }

        ++restart;
        shared.restarts.fetch_add(1, std::memory_order_relaxed);
    }

    shared.moves.fetch_add(local_moves_total,
                           std::memory_order_relaxed);
    shared.candidate_evaluations.fetch_add(
        local_evaluations_total, std::memory_order_relaxed);
}

inline RunResult run_annealer(const Params& params,
                              const State* initial_state = nullptr) {
    const auto start = std::chrono::steady_clock::now();
    SharedSearch shared(start);
    const uint64_t seed =
        params.seed != 0
            ? params.seed
            : (static_cast<uint64_t>(std::random_device{}()) << 32) ^
                  std::random_device{}();

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(params.threads));
    for (int worker = 0; worker < params.threads; ++worker) {
        workers.emplace_back(
            anneal_worker, worker, seed, std::cref(params),
            initial_state,
            std::ref(shared));
    }
    for (std::thread& worker : workers) worker.join();

    if (!shared.best.has_value()) {
        throw std::runtime_error("search stopped before initialization");
    }
    RunResult result;
    result.best = std::move(*shared.best);
    result.moves = shared.moves.load(std::memory_order_relaxed);
    result.candidate_evaluations =
        shared.candidate_evaluations.load(
            std::memory_order_relaxed);
    result.restarts =
        shared.restarts.load(std::memory_order_relaxed);
    result.exact_attempts =
        shared.exact_attempts.load(std::memory_order_relaxed);
    result.exact_successes =
        shared.exact_successes.load(std::memory_order_relaxed);
    result.seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    return result;
}

inline void write_state_csv(const State& state,
                            const std::string& filename) {
    const std::string temporary = filename + ".tmp";
    const auto entries = state.expanded_entries();
    {
        std::ofstream out(temporary, std::ios::binary);
        if (!out) {
            throw std::runtime_error(
                "failed to open output file: " + temporary);
        }
        for (int row = 0; row < MATRIX_ORDER; ++row) {
            for (int column = 0; column < MATRIX_ORDER; ++column) {
                out << static_cast<int>(entries[static_cast<size_t>(
                    column * MATRIX_ORDER + row)]);
                out << (column + 1 < MATRIX_ORDER ? ',' : '\n');
            }
        }
        out.flush();
        if (!out) {
            throw std::runtime_error(
                "failed while writing: " + temporary);
        }
    }

#if defined(_WIN32)
    const std::filesystem::path from(temporary);
    const std::filesystem::path to(filename);
    if (!MoveFileExW(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING |
                         MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error(
            "failed to replace output file: " + filename);
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        throw std::runtime_error(
            "failed to replace " + filename + ": " +
            error.message());
    }
#endif
}

}  // namespace xweigh_lift
