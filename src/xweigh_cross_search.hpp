// xweigh_cross_search.hpp: exact row/column frontier search for weighing matrices.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stop_signal.hpp"

namespace xweigh_cross {

constexpr int MAX_ORDER = 63;

struct Params {
    int threads = 1;
    double max_seconds = 0.0;
    size_t seed_limit = 16;
    int seed_index = -1;
    int radius_start = 2;
    int radius_max = 8;
    size_t candidate_limit = 128;
    uint64_t assignment_limit = 5000000;
    uint64_t frontier_node_limit = 1000000;
    bool exhaustive = false;
    bool analyze_only = false;
    std::string output = "W-cross.csv";
    std::string checkpoint = "xweigh-cross.checkpoint";
    std::string resume;
};

class Matrix {
public:
    Matrix() = default;
    Matrix(int order, int weight)
        : order_(order),
          weight_(weight),
          entries_(static_cast<size_t>(order) * order, int8_t(0)) {}

    static Matrix read_csv(const std::string& filename, int weight) {
        std::ifstream input(filename);
        if (!input)
            throw std::runtime_error("failed to open input matrix: " + filename);

        std::vector<std::vector<int8_t>> rows;
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::stringstream stream(line);
            std::vector<int8_t> row;
            std::string token;
            while (std::getline(stream, token, ',')) {
                const size_t first =
                    token.find_first_not_of(" \t\r\n");
                const size_t last =
                    token.find_last_not_of(" \t\r\n");
                if (first == std::string::npos) {
                    throw std::runtime_error(
                        "input matrix contains an empty entry");
                }
                const std::string trimmed =
                    token.substr(first, last - first + 1);
                size_t consumed = 0;
                int value = 0;
                try {
                    value = std::stoi(trimmed, &consumed);
                } catch (const std::exception&) {
                    consumed = 0;
                }
                if (consumed != trimmed.size() ||
                    value < -1 || value > 1) {
                    throw std::runtime_error(
                        "input matrix contains a non-ternary entry");
                }
                row.push_back(static_cast<int8_t>(value));
            }
            rows.push_back(std::move(row));
        }
        if (rows.empty() || rows.size() > static_cast<size_t>(MAX_ORDER))
            throw std::runtime_error("matrix order must be in [1, 63]");
        const int order = static_cast<int>(rows.size());
        if (std::any_of(rows.begin(), rows.end(),
                        [order](const auto& row) {
                            return row.size() != static_cast<size_t>(order);
                        })) {
            throw std::runtime_error("input matrix must be square");
        }

        Matrix matrix(order, weight);
        for (int row = 0; row < order; ++row) {
            for (int column = 0; column < order; ++column) {
                matrix.entry(row, column) =
                    rows[static_cast<size_t>(row)]
                        [static_cast<size_t>(column)];
            }
        }
        return matrix;
    }

    int order() const { return order_; }
    int weight() const { return weight_; }

    int8_t& entry(int row, int column) {
        return entries_[static_cast<size_t>(column) * order_ + row];
    }

    int8_t entry(int row, int column) const {
        return entries_[static_cast<size_t>(column) * order_ + row];
    }

    const std::vector<int8_t>& entries() const { return entries_; }

    std::vector<int8_t> row(int index) const {
        std::vector<int8_t> result(static_cast<size_t>(order_));
        for (int column = 0; column < order_; ++column)
            result[static_cast<size_t>(column)] = entry(index, column);
        return result;
    }

    std::vector<int8_t> column(int index) const {
        std::vector<int8_t> result(static_cast<size_t>(order_));
        for (int row_index = 0; row_index < order_; ++row_index)
            result[static_cast<size_t>(row_index)] = entry(row_index, index);
        return result;
    }

    void set_row(int index, const std::vector<int8_t>& values) {
        if (values.size() != static_cast<size_t>(order_))
            throw std::invalid_argument("row length does not match matrix order");
        for (int column = 0; column < order_; ++column)
            entry(index, column) = values[static_cast<size_t>(column)];
    }

    void set_column(int index, const std::vector<int8_t>& values) {
        if (values.size() != static_cast<size_t>(order_))
            throw std::invalid_argument(
                "column length does not match matrix order");
        for (int row_index = 0; row_index < order_; ++row_index)
            entry(row_index, index) =
                values[static_cast<size_t>(row_index)];
    }

    int row_dot(int first, int second) const {
        int dot = 0;
        for (int column = 0; column < order_; ++column)
            dot += entry(first, column) * entry(second, column);
        return dot;
    }

    int column_dot(int first, int second) const {
        int dot = 0;
        for (int row_index = 0; row_index < order_; ++row_index)
            dot += entry(row_index, first) * entry(row_index, second);
        return dot;
    }

    bool verify_fixed_weight() const {
        if (order_ < 1 || weight_ < 1 || weight_ > order_) return false;
        for (int index = 0; index < order_; ++index) {
            int row_weight = 0;
            int column_weight = 0;
            for (int other = 0; other < order_; ++other) {
                const int row_value = entry(index, other);
                const int column_value = entry(other, index);
                if (row_value < -1 || row_value > 1 ||
                    column_value < -1 || column_value > 1) {
                    return false;
                }
                row_weight += row_value != 0 ? 1 : 0;
                column_weight += column_value != 0 ? 1 : 0;
            }
            if (row_weight != weight_ || column_weight != weight_)
                return false;
        }
        return true;
    }

    bool verify_weighing() const {
        if (!verify_fixed_weight()) return false;
        for (int first = 0; first < order_; ++first) {
            for (int second = first; second < order_; ++second) {
                const int expected = first == second ? weight_ : 0;
                if (row_dot(first, second) != expected) return false;
            }
        }
        return true;
    }

    int64_t score() const {
        int64_t result = 0;
        for (int first = 0; first < order_; ++first) {
            for (int second = first + 1; second < order_; ++second)
                result += std::abs(row_dot(first, second));
        }
        return result;
    }

    int conflicts() const {
        int result = 0;
        for (int first = 0; first < order_; ++first) {
            for (int second = first + 1; second < order_; ++second)
                result += row_dot(first, second) != 0 ? 1 : 0;
        }
        return result;
    }

private:
    int order_ = 0;
    int weight_ = 0;
    std::vector<int8_t> entries_;
};

inline void write_matrix_csv(const Matrix& matrix,
                             const std::string& filename) {
    const std::string temporary = filename + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary);
        if (!output)
            throw std::runtime_error("failed to open output file: " + temporary);
        for (int row = 0; row < matrix.order(); ++row) {
            for (int column = 0; column < matrix.order(); ++column) {
                output << static_cast<int>(matrix.entry(row, column));
                output << (column + 1 < matrix.order() ? ',' : '\n');
            }
        }
        output.flush();
        if (!output)
            throw std::runtime_error("failed while writing: " + temporary);
    }
#if defined(_WIN32)
    const std::filesystem::path from(temporary);
    const std::filesystem::path to(filename);
    if (!MoveFileExW(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error("failed to replace output file: " + filename);
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error)
        throw std::runtime_error("failed to replace " + filename + ": " +
                                 error.message());
#endif
}

enum class Orientation : uint8_t {
    row,
    column
};

inline int oriented_dot(const Matrix& matrix, Orientation orientation,
                        int first, int second) {
    return orientation == Orientation::row
        ? matrix.row_dot(first, second)
        : matrix.column_dot(first, second);
}

class CliqueGraph {
public:
    CliqueGraph() = default;

    CliqueGraph(const Matrix& matrix, Orientation orientation)
        : order_(matrix.order()) {
        for (int first = 0; first < order_; ++first) {
            uint64_t neighbours = 0;
            for (int second = 0; second < order_; ++second) {
                if (first != second &&
                    oriented_dot(matrix, orientation, first, second) == 0) {
                    neighbours |= UINT64_C(1) << second;
                }
            }
            adjacency_[static_cast<size_t>(first)] = neighbours;
        }
    }

    int order() const { return order_; }
    uint64_t neighbours(int vertex) const {
        return adjacency_[static_cast<size_t>(vertex)];
    }

    std::vector<uint64_t> top_cliques(size_t limit,
                                      int minimum_size = 1) const {
        std::vector<uint64_t> best;
        auto keep = [&best, limit, minimum_size](uint64_t clique) {
            const int size = std::popcount(clique);
            if (size < minimum_size) return;
            if (std::find(best.begin(), best.end(), clique) != best.end())
                return;
            best.push_back(clique);
            std::sort(best.begin(), best.end(),
                      [](uint64_t first, uint64_t second) {
                          const int first_size = std::popcount(first);
                          const int second_size = std::popcount(second);
                          if (first_size != second_size)
                              return first_size > second_size;
                          return first < second;
                      });
            if (best.size() > limit) best.resize(limit);
        };

        auto search = [this, &best, limit, minimum_size, &keep](
                          uint64_t clique, uint64_t candidates,
                          uint64_t excluded, auto&& search_self) -> void {
            int cutoff = minimum_size;
            if (best.size() == limit)
                cutoff = std::max(
                    cutoff, std::popcount(best.back()));
            if (std::popcount(clique) + std::popcount(candidates) < cutoff)
                return;
            if (candidates == 0) {
                if (excluded == 0) keep(clique);
                return;
            }

            uint64_t pivot_pool = candidates | excluded;
            int pivot = -1;
            int pivot_degree = -1;
            while (pivot_pool != 0) {
                const int candidate = std::countr_zero(pivot_pool);
                const int degree =
                    std::popcount(candidates & neighbours(candidate));
                if (degree > pivot_degree) {
                    pivot_degree = degree;
                    pivot = candidate;
                }
                pivot_pool &= pivot_pool - 1;
            }
            uint64_t branches =
                pivot >= 0 ? candidates & ~neighbours(pivot) : candidates;
            while (branches != 0) {
                const int vertex = std::countr_zero(branches);
                const uint64_t bit = UINT64_C(1) << vertex;
                search_self(
                    clique | bit,
                    candidates & neighbours(vertex),
                    excluded & neighbours(vertex), search_self);
                candidates &= ~bit;
                excluded |= bit;
                branches &= ~bit;
            }
        };

        const uint64_t vertices =
            order_ == 64
                ? std::numeric_limits<uint64_t>::max()
                : (UINT64_C(1) << order_) - 1;
        search(0, vertices, 0, search);
        return best;
    }

private:
    int order_ = 0;
    std::array<uint64_t, MAX_ORDER> adjacency_{};
};

inline std::vector<int> mask_indices(uint64_t mask) {
    std::vector<int> result;
    while (mask != 0) {
        const int index = std::countr_zero(mask);
        result.push_back(index);
        mask &= mask - 1;
    }
    return result;
}

struct SeedPair {
    uint64_t rows = 0;
    uint64_t columns = 0;
};

inline std::vector<SeedPair> build_seed_portfolio(
    const Matrix& candidate, size_t seed_limit) {
    const CliqueGraph row_graph(candidate, Orientation::row);
    const CliqueGraph column_graph(candidate, Orientation::column);
    const size_t side_limit =
        std::max<size_t>(1, static_cast<size_t>(
            std::ceil(std::sqrt(static_cast<double>(seed_limit)))));
    auto row_cliques = row_graph.top_cliques(side_limit);
    auto column_cliques = column_graph.top_cliques(side_limit);
    if (row_cliques.empty() || column_cliques.empty())
        throw std::runtime_error("failed to extract orthogonal seed cliques");

    std::vector<SeedPair> result;
    for (uint64_t rows : row_cliques) {
        for (uint64_t columns : column_cliques)
            result.push_back({ rows, columns });
    }
    std::sort(result.begin(), result.end(),
              [](const SeedPair& first, const SeedPair& second) {
                  const int first_size =
                      std::popcount(first.rows) +
                      std::popcount(first.columns);
                  const int second_size =
                      std::popcount(second.rows) +
                      std::popcount(second.columns);
                  if (first_size != second_size)
                      return first_size > second_size;
                  if (first.rows != second.rows)
                      return first.rows < second.rows;
                  return first.columns < second.columns;
              });
    if (result.size() > seed_limit) result.resize(seed_limit);
    return result;
}

struct Frontier {
    Matrix matrix;
    uint64_t known_rows = 0;
    uint64_t known_columns = 0;

    int row_count() const { return std::popcount(known_rows); }
    int column_count() const { return std::popcount(known_columns); }
    int progress() const { return row_count() + column_count(); }
};

inline Frontier make_frontier(const Matrix& candidate,
                              const SeedPair& seed) {
    Frontier frontier{ candidate, seed.rows, seed.columns };
    for (int first : mask_indices(seed.rows)) {
        for (int second : mask_indices(seed.rows)) {
            if (first < second && candidate.row_dot(first, second) != 0)
                throw std::runtime_error("row seed is not orthogonal");
        }
    }
    for (int first : mask_indices(seed.columns)) {
        for (int second : mask_indices(seed.columns)) {
            if (first < second &&
                candidate.column_dot(first, second) != 0) {
                throw std::runtime_error("column seed is not orthogonal");
            }
        }
    }
    return frontier;
}

struct ExtensionKey {
    std::array<int8_t, MAX_ORDER> sums{};
    uint8_t nonzeros = 0;
    uint8_t dimensions = 0;

    bool operator==(const ExtensionKey&) const = default;
};

struct ExtensionKeyHash {
    size_t operator()(const ExtensionKey& key) const {
        size_t hash = static_cast<size_t>(1469598103934665603ULL);
        hash ^= key.nonzeros;
        hash *= static_cast<size_t>(1099511628211ULL);
        for (int index = 0; index < key.dimensions; ++index) {
            hash ^= static_cast<uint8_t>(
                key.sums[static_cast<size_t>(index)]);
            hash *= static_cast<size_t>(1099511628211ULL);
        }
        return hash;
    }
};

struct HalfAssignment {
    uint64_t packed = 0;
    uint8_t distance = 0;
};

struct ExtensionCandidate {
    std::vector<int8_t> values;
    int distance = 0;
};

struct ExtensionResult {
    std::vector<ExtensionCandidate> candidates;
    uint64_t left_assignments = 0;
    uint64_t right_assignments = 0;
    bool truncated = false;
};

inline uint8_t encode_trit(int8_t value) {
    return static_cast<uint8_t>(value + 1);
}

inline int8_t decode_trit(uint64_t packed, int index) {
    return static_cast<int8_t>(
        static_cast<int>((packed >> (2 * index)) & 3U) - 1);
}

struct PackedVector {
    uint64_t low = 0;
    uint64_t high = 0;

    bool operator==(const PackedVector&) const = default;
};

struct PackedVectorHash {
    size_t operator()(const PackedVector& value) const {
        uint64_t mixed = value.low ^
            (value.high + UINT64_C(0x9e3779b97f4a7c15) +
             (value.low << 6) + (value.low >> 2));
        mixed ^= mixed >> 30;
        mixed *= UINT64_C(0xbf58476d1ce4e5b9);
        mixed ^= mixed >> 27;
        mixed *= UINT64_C(0x94d049bb133111eb);
        mixed ^= mixed >> 31;
        return static_cast<size_t>(mixed);
    }
};

inline PackedVector pack_vector(const std::vector<int8_t>& values) {
    PackedVector packed;
    for (size_t index = 0; index < values.size(); ++index) {
        const uint64_t code = encode_trit(values[index]);
        if (index < 32)
            packed.low |= code << (2 * index);
        else
            packed.high |= code << (2 * (index - 32));
    }
    return packed;
}

template <class Callback>
inline void enumerate_half_assignments(
    const std::vector<int>& positions,
    const std::vector<int8_t>& reference,
    const std::vector<std::array<int8_t, MAX_ORDER>>& coefficients,
    int dimensions, int radius, uint64_t assignment_limit,
    const std::atomic<bool>* stop_flag, uint64_t& leaves,
    bool& truncated, Callback&& callback) {
    std::array<int8_t, MAX_ORDER> sums{};
    auto enumerate = [&positions, &reference, &coefficients, dimensions,
                      assignment_limit, &leaves, &truncated,
                      stop_flag, &callback, &sums](
                         int target_distance, int depth, int nonzeros,
                         int distance, uint64_t packed,
                         auto&& enumerate_self) -> void {
        if (truncated || g_stop.load(std::memory_order_relaxed) ||
            (stop_flag != nullptr &&
             stop_flag->load(std::memory_order_relaxed))) {
            return;
        }
        if (depth == static_cast<int>(positions.size())) {
            if (distance != target_distance) return;
            if (assignment_limit != 0 && leaves >= assignment_limit) {
                truncated = true;
                return;
            }
            ++leaves;
            ExtensionKey key;
            key.nonzeros = static_cast<uint8_t>(nonzeros);
            key.dimensions = static_cast<uint8_t>(dimensions);
            for (int index = 0; index < dimensions; ++index)
                key.sums[static_cast<size_t>(index)] =
                    sums[static_cast<size_t>(index)];
            callback(key, HalfAssignment{
                packed, static_cast<uint8_t>(distance)
            });
            return;
        }
        const int remaining =
            static_cast<int>(positions.size()) - depth;
        if (distance > target_distance ||
            distance + remaining < target_distance) {
            return;
        }

        const int position = positions[static_cast<size_t>(depth)];
        const int8_t preferred = reference[static_cast<size_t>(position)];
        std::array<int8_t, 3> choices{ preferred, -1, 0 };
        int next = 1;
        for (int8_t value : std::array<int8_t, 3>{ -1, 0, 1 }) {
            if (value != preferred)
                choices[static_cast<size_t>(next++)] = value;
        }

        for (int choice = 0; choice < 3; ++choice) {
            const int8_t value = choices[static_cast<size_t>(choice)];
            const int new_distance =
                distance + (value != preferred ? 1 : 0);
            if (new_distance > target_distance) continue;
            for (int index = 0; index < dimensions; ++index) {
                sums[static_cast<size_t>(index)] =
                    static_cast<int8_t>(
                        sums[static_cast<size_t>(index)] +
                        coefficients[static_cast<size_t>(depth)]
                                    [static_cast<size_t>(index)] *
                            value);
            }
            enumerate_self(
                target_distance, depth + 1,
                nonzeros + (value != 0 ? 1 : 0), new_distance,
                packed | (static_cast<uint64_t>(encode_trit(value)) <<
                          (2 * depth)),
                enumerate_self);
            for (int index = 0; index < dimensions; ++index) {
                sums[static_cast<size_t>(index)] =
                    static_cast<int8_t>(
                        sums[static_cast<size_t>(index)] -
                        coefficients[static_cast<size_t>(depth)]
                                    [static_cast<size_t>(index)] *
                            value);
            }
            if (truncated) return;
        }
    };
    for (int target_distance = 0;
         target_distance <= radius && !truncated; ++target_distance) {
        sums.fill(0);
        enumerate(
            target_distance, 0, 0, 0, 0, enumerate);
    }
}

inline std::vector<int8_t> oriented_vector(
    const Matrix& matrix, Orientation orientation, int index) {
    return orientation == Orientation::row
        ? matrix.row(index)
        : matrix.column(index);
}

inline int8_t oriented_entry(const Matrix& matrix, Orientation orientation,
                             int vector_index, int coordinate) {
    return orientation == Orientation::row
        ? matrix.entry(vector_index, coordinate)
        : matrix.entry(coordinate, vector_index);
}

inline ExtensionResult generate_extensions(
    const Frontier& frontier, const Matrix& reference,
    Orientation orientation, int vector_index, int radius,
    size_t candidate_limit, uint64_t assignment_limit,
    const std::atomic<bool>* stop_flag = nullptr) {
    const int order = reference.order();
    const uint64_t fixed_mask =
        orientation == Orientation::row
            ? frontier.known_columns
            : frontier.known_rows;
    const uint64_t constraint_mask =
        orientation == Orientation::row
            ? frontier.known_rows
            : frontier.known_columns;
    const auto constraints = mask_indices(constraint_mask);
    const int dimensions = static_cast<int>(constraints.size());
    std::vector<int8_t> base =
        oriented_vector(reference, orientation, vector_index);
    std::vector<int> free_positions;
    std::vector<int> fixed_positions;
    for (int coordinate = 0; coordinate < order; ++coordinate) {
        if ((fixed_mask & (UINT64_C(1) << coordinate)) != 0)
            fixed_positions.push_back(coordinate);
        else
            free_positions.push_back(coordinate);
    }

    int fixed_nonzeros = 0;
    int fixed_distance = 0;
    std::array<int8_t, MAX_ORDER> targets{};
    for (int coordinate : fixed_positions) {
        const int8_t value = oriented_entry(
            frontier.matrix, orientation, vector_index, coordinate);
        fixed_nonzeros += value != 0 ? 1 : 0;
        fixed_distance +=
            value != base[static_cast<size_t>(coordinate)] ? 1 : 0;
        base[static_cast<size_t>(coordinate)] = value;
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            const int coefficient = oriented_entry(
                frontier.matrix, orientation,
                constraints[static_cast<size_t>(dimension)], coordinate);
            targets[static_cast<size_t>(dimension)] =
                static_cast<int8_t>(
                    targets[static_cast<size_t>(dimension)] -
                    coefficient * value);
        }
    }

    ExtensionResult result;
    const int required_nonzeros = reference.weight() - fixed_nonzeros;
    const int remaining_radius = radius - fixed_distance;
    if (required_nonzeros < 0 ||
        required_nonzeros > static_cast<int>(free_positions.size()) ||
        remaining_radius < 0) {
        return result;
    }

    const size_t split = free_positions.size() / 2;
    std::vector<int> left_positions(
        free_positions.begin(), free_positions.begin() +
        static_cast<std::ptrdiff_t>(split));
    std::vector<int> right_positions(
        free_positions.begin() + static_cast<std::ptrdiff_t>(split),
        free_positions.end());

    auto build_coefficients =
        [&frontier, orientation, &constraints, dimensions](
            const std::vector<int>& positions) {
            std::vector<std::array<int8_t, MAX_ORDER>> coefficients(
                positions.size());
            for (size_t position = 0; position < positions.size(); ++position) {
                for (int dimension = 0; dimension < dimensions; ++dimension) {
                    coefficients[position][static_cast<size_t>(dimension)] =
                        static_cast<int8_t>(oriented_entry(
                            frontier.matrix, orientation,
                            constraints[static_cast<size_t>(dimension)],
                            positions[position]));
                }
            }
            return coefficients;
        };
    const auto left_coefficients = build_coefficients(left_positions);
    const auto right_coefficients = build_coefficients(right_positions);

    std::unordered_map<
        ExtensionKey, std::vector<HalfAssignment>, ExtensionKeyHash>
        left_table;
    bool left_truncated = false;
    enumerate_half_assignments(
        left_positions, base, left_coefficients, dimensions,
        remaining_radius, assignment_limit, stop_flag,
        result.left_assignments, left_truncated,
        [&left_table](const ExtensionKey& key,
                      const HalfAssignment& assignment) {
            left_table[key].push_back(assignment);
        });
    result.truncated = left_truncated;
    if (left_truncated || g_stop.load(std::memory_order_relaxed) ||
        (stop_flag != nullptr &&
         stop_flag->load(std::memory_order_relaxed))) {
        return result;
    }

    std::unordered_set<PackedVector, PackedVectorHash> seen;
    auto keep_candidate =
        [&result, candidate_limit, &seen](ExtensionCandidate candidate) {
            const PackedVector packed = pack_vector(candidate.values);
            if (!seen.insert(packed).second) return;
            if (candidate_limit == 0 ||
                result.candidates.size() < candidate_limit) {
                result.candidates.push_back(std::move(candidate));
                return;
            }
            auto worst = std::max_element(
                result.candidates.begin(), result.candidates.end(),
                [](const ExtensionCandidate& first,
                   const ExtensionCandidate& second) {
                    return first.distance < second.distance;
                });
            if (worst != result.candidates.end() &&
                candidate.distance < worst->distance) {
                seen.erase(pack_vector(worst->values));
                *worst = std::move(candidate);
            }
        };

    bool right_truncated = false;
    enumerate_half_assignments(
        right_positions, base, right_coefficients, dimensions,
        remaining_radius, assignment_limit, stop_flag,
        result.right_assignments, right_truncated,
        [&](const ExtensionKey& right_key,
            const HalfAssignment& right_assignment) {
            const int left_nonzeros =
                required_nonzeros - right_key.nonzeros;
            if (left_nonzeros < 0 ||
                left_nonzeros >
                    static_cast<int>(left_positions.size())) {
                return;
            }
            ExtensionKey needed;
            needed.nonzeros = static_cast<uint8_t>(left_nonzeros);
            needed.dimensions = static_cast<uint8_t>(dimensions);
            for (int dimension = 0; dimension < dimensions; ++dimension) {
                const int value =
                    targets[static_cast<size_t>(dimension)] -
                    right_key.sums[static_cast<size_t>(dimension)];
                if (value < std::numeric_limits<int8_t>::min() ||
                    value > std::numeric_limits<int8_t>::max()) {
                    return;
                }
                needed.sums[static_cast<size_t>(dimension)] =
                    static_cast<int8_t>(value);
            }
            const auto match = left_table.find(needed);
            if (match == left_table.end()) return;
            for (const HalfAssignment& left_assignment : match->second) {
                const int distance =
                    fixed_distance + left_assignment.distance +
                    right_assignment.distance;
                if (distance > radius) continue;
                std::vector<int8_t> values = base;
                for (size_t index = 0; index < left_positions.size(); ++index) {
                    values[static_cast<size_t>(left_positions[index])] =
                        decode_trit(left_assignment.packed,
                                    static_cast<int>(index));
                }
                for (size_t index = 0; index < right_positions.size(); ++index) {
                    values[static_cast<size_t>(right_positions[index])] =
                        decode_trit(right_assignment.packed,
                                    static_cast<int>(index));
                }
                keep_candidate({ std::move(values), distance });
            }
        });
    result.truncated = result.truncated || right_truncated;
    std::sort(
        result.candidates.begin(), result.candidates.end(),
        [](const ExtensionCandidate& first,
           const ExtensionCandidate& second) {
            if (first.distance != second.distance)
                return first.distance < second.distance;
            return first.values < second.values;
        });
    return result;
}

inline bool frontier_feasible(const Frontier& frontier) {
    const int order = frontier.matrix.order();
    const int weight = frontier.matrix.weight();
    const auto rows = mask_indices(frontier.known_rows);
    const auto columns = mask_indices(frontier.known_columns);
    const int free_rows = order - static_cast<int>(rows.size());
    const int free_columns = order - static_cast<int>(columns.size());

    for (int row : rows) {
        int support = 0;
        for (int column = 0; column < order; ++column)
            support += frontier.matrix.entry(row, column) != 0 ? 1 : 0;
        if (support != weight) return false;
    }
    for (size_t first = 0; first < rows.size(); ++first) {
        for (size_t second = first + 1; second < rows.size(); ++second) {
            if (frontier.matrix.row_dot(rows[first], rows[second]) != 0)
                return false;
        }
    }
    for (int column : columns) {
        int support = 0;
        for (int row = 0; row < order; ++row)
            support += frontier.matrix.entry(row, column) != 0 ? 1 : 0;
        if (support != weight) return false;
    }
    for (size_t first = 0; first < columns.size(); ++first) {
        for (size_t second = first + 1; second < columns.size(); ++second) {
            if (frontier.matrix.column_dot(
                    columns[first], columns[second]) != 0) {
                return false;
            }
        }
    }

    for (int row = 0; row < order; ++row) {
        if ((frontier.known_rows & (UINT64_C(1) << row)) != 0) continue;
        int fixed_support = 0;
        for (int column : columns)
            fixed_support +=
                frontier.matrix.entry(row, column) != 0 ? 1 : 0;
        if (fixed_support > weight ||
            fixed_support + free_columns < weight) {
            return false;
        }
        for (int known_row : rows) {
            int partial_dot = 0;
            int available = 0;
            for (int column = 0; column < order; ++column) {
                if ((frontier.known_columns &
                     (UINT64_C(1) << column)) != 0) {
                    partial_dot +=
                        frontier.matrix.entry(row, column) *
                        frontier.matrix.entry(known_row, column);
                } else {
                    available +=
                        frontier.matrix.entry(known_row, column) != 0
                            ? 1 : 0;
                }
            }
            if (std::abs(partial_dot) > available) return false;
        }
    }

    for (int column = 0; column < order; ++column) {
        if ((frontier.known_columns & (UINT64_C(1) << column)) != 0)
            continue;
        int fixed_support = 0;
        for (int row : rows)
            fixed_support +=
                frontier.matrix.entry(row, column) != 0 ? 1 : 0;
        if (fixed_support > weight ||
            fixed_support + free_rows < weight) {
            return false;
        }
        for (int known_column : columns) {
            int partial_dot = 0;
            int available = 0;
            for (int row = 0; row < order; ++row) {
                if ((frontier.known_rows &
                     (UINT64_C(1) << row)) != 0) {
                    partial_dot +=
                        frontier.matrix.entry(row, column) *
                        frontier.matrix.entry(row, known_column);
                } else {
                    available +=
                        frontier.matrix.entry(row, known_column) != 0
                            ? 1 : 0;
                }
            }
            if (std::abs(partial_dot) > available) return false;
        }
    }

    for (int first = 0; first < order; ++first) {
        if ((frontier.known_rows & (UINT64_C(1) << first)) != 0) continue;
        for (int second = first + 1; second < order; ++second) {
            if ((frontier.known_rows & (UINT64_C(1) << second)) != 0)
                continue;
            int partial_dot = 0;
            for (int column : columns) {
                partial_dot +=
                    frontier.matrix.entry(first, column) *
                    frontier.matrix.entry(second, column);
            }
            if (std::abs(partial_dot) > free_columns) return false;
        }
    }
    for (int first = 0; first < order; ++first) {
        if ((frontier.known_columns & (UINT64_C(1) << first)) != 0)
            continue;
        for (int second = first + 1; second < order; ++second) {
            if ((frontier.known_columns & (UINT64_C(1) << second)) != 0)
                continue;
            int partial_dot = 0;
            for (int row : rows) {
                partial_dot +=
                    frontier.matrix.entry(row, first) *
                    frontier.matrix.entry(row, second);
            }
            if (std::abs(partial_dot) > free_rows) return false;
        }
    }
    return true;
}

inline int64_t frontier_partial_penalty(const Frontier& frontier) {
    const int order = frontier.matrix.order();
    const auto rows = mask_indices(frontier.known_rows);
    const auto columns = mask_indices(frontier.known_columns);
    int64_t penalty = 0;
    for (int first = 0; first < order; ++first) {
        if ((frontier.known_rows & (UINT64_C(1) << first)) != 0) continue;
        for (int second = first + 1; second < order; ++second) {
            if ((frontier.known_rows & (UINT64_C(1) << second)) != 0)
                continue;
            int partial_dot = 0;
            for (int column : columns) {
                partial_dot +=
                    frontier.matrix.entry(first, column) *
                    frontier.matrix.entry(second, column);
            }
            penalty += std::abs(partial_dot);
        }
    }
    for (int first = 0; first < order; ++first) {
        if ((frontier.known_columns & (UINT64_C(1) << first)) != 0)
            continue;
        for (int second = first + 1; second < order; ++second) {
            if ((frontier.known_columns & (UINT64_C(1) << second)) != 0)
                continue;
            int partial_dot = 0;
            for (int row : rows) {
                partial_dot +=
                    frontier.matrix.entry(row, first) *
                    frontier.matrix.entry(row, second);
            }
            penalty += std::abs(partial_dot);
        }
    }
    return penalty;
}

inline uint64_t bounded_assignment_estimate(int positions, int radius) {
    if (radius < 0) return 0;
    radius = std::min(radius, positions);
    uint64_t total = 0;
    uint64_t combinations = 1;
    uint64_t powers = 1;
    for (int distance = 0; distance <= radius; ++distance) {
        if (distance > 0) {
            combinations =
                combinations *
                static_cast<uint64_t>(positions - distance + 1) /
                static_cast<uint64_t>(distance);
            if (powers > std::numeric_limits<uint64_t>::max() / 2)
                return std::numeric_limits<uint64_t>::max();
            powers *= 2;
        }
        if (combinations >
            (std::numeric_limits<uint64_t>::max() - total) / powers) {
            return std::numeric_limits<uint64_t>::max();
        }
        total += combinations * powers;
    }
    return total;
}

struct ExtensionChoice {
    Orientation orientation = Orientation::row;
    int index = -1;
    uint64_t estimate = std::numeric_limits<uint64_t>::max();
    int free_coordinates = MAX_ORDER + 1;
};

inline ExtensionChoice choose_extension(
    const Frontier& frontier, const Matrix& reference, int radius) {
    ExtensionChoice best;
    const int order = reference.order();
    for (Orientation orientation :
         { Orientation::row, Orientation::column }) {
        const uint64_t known =
            orientation == Orientation::row
                ? frontier.known_rows
                : frontier.known_columns;
        const uint64_t fixed =
            orientation == Orientation::row
                ? frontier.known_columns
                : frontier.known_rows;
        const int fixed_count = std::popcount(fixed);
        const int free_count = order - fixed_count;
        const int half = (free_count + 1) / 2;
        const int constraints = std::popcount(known);
        for (int index = 0; index < order; ++index) {
            if ((known & (UINT64_C(1) << index)) != 0) continue;
            int fixed_distance = 0;
            for (int coordinate : mask_indices(fixed)) {
                fixed_distance +=
                    oriented_entry(frontier.matrix, orientation,
                                   index, coordinate) !=
                    oriented_entry(reference, orientation,
                                   index, coordinate) ? 1 : 0;
            }
            uint64_t estimate =
                bounded_assignment_estimate(
                    half, radius - fixed_distance);
            if (estimate != 0) {
                estimate = std::max<uint64_t>(
                    1, estimate /
                        static_cast<uint64_t>(constraints + 1));
            }
            if (estimate < best.estimate ||
                (estimate == best.estimate &&
                 free_count < best.free_coordinates) ||
                (estimate == best.estimate &&
                 free_count == best.free_coordinates &&
                 static_cast<int>(orientation) <
                     static_cast<int>(best.orientation)) ||
                (estimate == best.estimate &&
                 free_count == best.free_coordinates &&
                 orientation == best.orientation &&
                 index < best.index)) {
                best = {
                    orientation, index, estimate, free_count
                };
            }
        }
    }
    return best;
}

inline void write_checkpoint(const Frontier& frontier,
                             const std::string& filename) {
    if (filename.empty()) return;
    const std::string temporary = filename + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary);
        if (!output)
            throw std::runtime_error(
                "failed to open checkpoint: " + temporary);
        output << "XWEIGH_CROSS 1\n";
        output << frontier.matrix.order() << ' '
               << frontier.matrix.weight() << '\n';
        output << frontier.known_rows << ' '
               << frontier.known_columns << '\n';
        for (int row = 0; row < frontier.matrix.order(); ++row) {
            for (int column = 0;
                 column < frontier.matrix.order(); ++column) {
                output << static_cast<int>(
                    frontier.matrix.entry(row, column));
                output << (column + 1 < frontier.matrix.order()
                               ? ',' : '\n');
            }
        }
        output.flush();
        if (!output)
            throw std::runtime_error(
                "failed while writing checkpoint: " + temporary);
    }
#if defined(_WIN32)
    const std::filesystem::path from(temporary);
    const std::filesystem::path to(filename);
    if (!MoveFileExW(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING |
                         MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error(
            "failed to replace checkpoint: " + filename);
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error)
        throw std::runtime_error(
            "failed to replace checkpoint " + filename + ": " +
            error.message());
#endif
}

inline Frontier read_checkpoint(const std::string& filename,
                                const Matrix& reference) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("failed to open checkpoint: " + filename);
    std::string magic;
    int version = 0;
    input >> magic >> version;
    if (magic != "XWEIGH_CROSS" || version != 1)
        throw std::runtime_error("unsupported checkpoint format");
    int order = 0;
    int weight = 0;
    uint64_t rows = 0;
    uint64_t columns = 0;
    input >> order >> weight >> rows >> columns;
    std::string line;
    std::getline(input, line);
    if (order != reference.order() || weight != reference.weight())
        throw std::runtime_error(
            "checkpoint dimensions do not match input matrix");
    const uint64_t valid_mask = (UINT64_C(1) << order) - 1;
    if ((rows & ~valid_mask) != 0 || (columns & ~valid_mask) != 0)
        throw std::runtime_error("checkpoint contains invalid frontier masks");
    Matrix matrix(order, weight);
    for (int row = 0; row < order; ++row) {
        if (!std::getline(input, line))
            throw std::runtime_error("truncated checkpoint matrix");
        std::stringstream stream(line);
        std::string token;
        for (int column = 0; column < order; ++column) {
            if (!std::getline(stream, token, ','))
                throw std::runtime_error("truncated checkpoint row");
            const int value = std::stoi(token);
            if (value < -1 || value > 1)
                throw std::runtime_error(
                    "checkpoint contains a non-ternary entry");
            matrix.entry(row, column) = static_cast<int8_t>(value);
        }
    }
    Frontier frontier{ std::move(matrix), rows, columns };
    if (!frontier_feasible(frontier))
        throw std::runtime_error("checkpoint frontier is inconsistent");
    return frontier;
}

struct RootTask {
    Frontier frontier;
    int radius = 0;
};

struct SearchResult {
    std::optional<Matrix> solution;
    Frontier best;
    uint64_t frontier_nodes = 0;
    uint64_t generated_extensions = 0;
    uint64_t generated_row_extensions = 0;
    uint64_t generated_column_extensions = 0;
    uint64_t left_assignments = 0;
    uint64_t right_assignments = 0;
    uint64_t truncated_generators = 0;
    double seconds = 0.0;
};

struct SharedSearch {
    SharedSearch(const Matrix& reference_,
                 const Params& params_,
                 std::vector<RootTask> tasks_)
        : reference(reference_),
          params(params_),
          tasks(std::move(tasks_)),
          start(std::chrono::steady_clock::now()) {}

    const Matrix& reference;
    const Params& params;
    std::vector<RootTask> tasks;
    std::atomic<size_t> next_task{ 0 };
    std::atomic<bool> done{ false };
    std::atomic<uint64_t> frontier_nodes{ 0 };
    std::atomic<uint64_t> generated_extensions{ 0 };
    std::atomic<uint64_t> generated_row_extensions{ 0 };
    std::atomic<uint64_t> generated_column_extensions{ 0 };
    std::atomic<uint64_t> left_assignments{ 0 };
    std::atomic<uint64_t> right_assignments{ 0 };
    std::atomic<uint64_t> truncated_generators{ 0 };
    std::mutex best_mutex;
    std::exception_ptr error;
    std::mutex timer_mutex;
    std::condition_variable timer_condition;
    std::optional<Matrix> solution;
    std::optional<Frontier> best;
    std::chrono::steady_clock::time_point start;
};

inline double elapsed_seconds(const SharedSearch& shared) {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - shared.start)
        .count();
}

inline bool should_stop(SharedSearch& shared) {
    if (shared.done.load(std::memory_order_relaxed) ||
        g_stop.load(std::memory_order_relaxed)) {
        return true;
    }
    if (shared.params.max_seconds > 0.0 &&
        elapsed_seconds(shared) >= shared.params.max_seconds) {
        shared.done.store(true, std::memory_order_relaxed);
        return true;
    }
    if (shared.params.frontier_node_limit != 0 &&
        shared.frontier_nodes.load(std::memory_order_relaxed) >=
            shared.params.frontier_node_limit) {
        return true;
    }
    return false;
}

inline void publish_frontier(SharedSearch& shared,
                             const Frontier& frontier,
                             int worker, int radius) {
    std::lock_guard<std::mutex> lock(shared.best_mutex);
    if (shared.best.has_value() &&
        frontier.progress() <= shared.best->progress()) {
        return;
    }
    shared.best = frontier;
    const double elapsed = elapsed_seconds(shared);
    std::cout << "[frontier] seconds=" << elapsed
              << " worker=" << worker
              << " radius=" << radius
              << " rows=" << frontier.row_count()
              << " columns=" << frontier.column_count()
              << " progress=" << frontier.progress() << '\n';
    write_checkpoint(frontier, shared.params.checkpoint);
}

inline void search_frontier(SharedSearch& shared, Frontier frontier,
                            int radius, int worker) {
    if (should_stop(shared)) return;
    const uint64_t node =
        shared.frontier_nodes.fetch_add(1, std::memory_order_relaxed) + 1;
    if (shared.params.frontier_node_limit != 0 &&
        node > shared.params.frontier_node_limit) {
        return;
    }
    if (!frontier_feasible(frontier)) return;
    publish_frontier(shared, frontier, worker, radius);

    const uint64_t all =
        (UINT64_C(1) << frontier.matrix.order()) - 1;
    if (frontier.known_rows == all ||
        frontier.known_columns == all) {
        if (frontier.matrix.verify_weighing()) {
            std::lock_guard<std::mutex> lock(shared.best_mutex);
            if (!shared.solution.has_value()) {
                shared.solution = frontier.matrix;
                shared.done.store(true, std::memory_order_relaxed);
                std::cout << "[solution] seconds="
                          << elapsed_seconds(shared)
                          << " worker=" << worker << '\n';
            }
        }
        return;
    }

    const ExtensionChoice choice =
        choose_extension(frontier, shared.reference, radius);
    if (choice.index < 0 || choice.estimate == 0) return;
    ExtensionResult extensions = generate_extensions(
        frontier, shared.reference, choice.orientation, choice.index,
        radius,
        shared.params.exhaustive ? 0 : shared.params.candidate_limit,
        shared.params.exhaustive ? 0 : shared.params.assignment_limit,
        &shared.done);
    shared.generated_extensions.fetch_add(
        extensions.candidates.size(), std::memory_order_relaxed);
    if (choice.orientation == Orientation::row) {
        shared.generated_row_extensions.fetch_add(
            extensions.candidates.size(), std::memory_order_relaxed);
    } else {
        shared.generated_column_extensions.fetch_add(
            extensions.candidates.size(), std::memory_order_relaxed);
    }
    shared.left_assignments.fetch_add(
        extensions.left_assignments, std::memory_order_relaxed);
    shared.right_assignments.fetch_add(
        extensions.right_assignments, std::memory_order_relaxed);
    if (extensions.truncated) {
        shared.truncated_generators.fetch_add(
            1, std::memory_order_relaxed);
    }

    struct RankedFrontier {
        Frontier frontier;
        int distance = 0;
        int64_t partial_penalty = 0;
    };
    std::vector<RankedFrontier> branches;
    branches.reserve(extensions.candidates.size());
    for (const ExtensionCandidate& extension : extensions.candidates) {
        Frontier next = frontier;
        if (choice.orientation == Orientation::row) {
            next.matrix.set_row(choice.index, extension.values);
            next.known_rows |= UINT64_C(1) << choice.index;
        } else {
            next.matrix.set_column(choice.index, extension.values);
            next.known_columns |= UINT64_C(1) << choice.index;
        }
        if (frontier_feasible(next)) {
            const int64_t partial_penalty =
                frontier_partial_penalty(next);
            branches.push_back({
                std::move(next), extension.distance,
                partial_penalty
            });
        }
    }
    std::sort(
        branches.begin(), branches.end(),
        [](const RankedFrontier& first, const RankedFrontier& second) {
            if (first.distance != second.distance)
                return first.distance < second.distance;
            return first.partial_penalty < second.partial_penalty;
        });
    for (RankedFrontier& branch : branches) {
        if (should_stop(shared)) return;
        search_frontier(
            shared, std::move(branch.frontier), radius, worker);
        if (shared.done.load(std::memory_order_relaxed)) return;
    }
}

inline void search_worker(SharedSearch& shared, int worker) {
    try {
        while (!should_stop(shared)) {
            const size_t task_index =
                shared.next_task.fetch_add(1, std::memory_order_relaxed);
            if (task_index >= shared.tasks.size()) return;
            RootTask task = shared.tasks[task_index];
            search_frontier(
                shared, std::move(task.frontier), task.radius, worker);
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(shared.best_mutex);
        if (shared.error == nullptr) shared.error = std::current_exception();
        shared.done.store(true, std::memory_order_relaxed);
    }
}

inline SearchResult run_search(
    const Matrix& reference, const Params& params,
    const std::vector<SeedPair>* supplied_portfolio = nullptr) {
    std::vector<RootTask> tasks;
    if (!params.resume.empty()) {
        Frontier resumed = read_checkpoint(params.resume, reference);
        const int first_radius =
            params.exhaustive ? reference.order() : params.radius_start;
        const int last_radius =
            params.exhaustive ? reference.order() : params.radius_max;
        for (int radius = first_radius; radius <= last_radius; ++radius)
            tasks.push_back({ resumed, radius });
    } else {
        std::vector<SeedPair> owned_portfolio;
        if (supplied_portfolio == nullptr) {
            owned_portfolio =
                build_seed_portfolio(reference, params.seed_limit);
            supplied_portfolio = &owned_portfolio;
        }
        const auto& portfolio = *supplied_portfolio;
        const int first_radius =
            params.exhaustive ? reference.order() : params.radius_start;
        const int last_radius =
            params.exhaustive ? reference.order() : params.radius_max;
        for (int radius = first_radius; radius <= last_radius; ++radius) {
            for (size_t seed = 0; seed < portfolio.size(); ++seed) {
                if (params.seed_index >= 0 &&
                    static_cast<int>(seed) != params.seed_index) {
                    continue;
                }
                tasks.push_back({
                    make_frontier(reference, portfolio[seed]),
                    radius
                });
            }
        }
    }
    if (tasks.empty()) throw std::runtime_error("no Criss-Cross root tasks");

    SharedSearch shared(reference, params, std::move(tasks));
    std::thread timer;
    if (params.max_seconds > 0.0) {
        timer = std::thread([&shared]() {
            std::unique_lock<std::mutex> lock(shared.timer_mutex);
            const bool finished = shared.timer_condition.wait_for(
                lock,
                std::chrono::duration<double>(
                    shared.params.max_seconds),
                [&shared]() {
                    return shared.done.load(
                        std::memory_order_relaxed);
                });
            if (!finished)
                shared.done.store(true, std::memory_order_relaxed);
        });
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(params.threads));
    for (int worker = 0; worker < params.threads; ++worker)
        workers.emplace_back(search_worker, std::ref(shared), worker);
    for (std::thread& worker : workers) worker.join();
    shared.done.store(true, std::memory_order_relaxed);
    shared.timer_condition.notify_all();
    if (timer.joinable()) timer.join();
    if (shared.error != nullptr) std::rethrow_exception(shared.error);

    SearchResult result;
    {
        std::lock_guard<std::mutex> lock(shared.best_mutex);
        result.solution = std::move(shared.solution);
        if (shared.best.has_value())
            result.best = std::move(*shared.best);
        else
            result.best = shared.tasks.front().frontier;
    }
    result.frontier_nodes =
        shared.frontier_nodes.load(std::memory_order_relaxed);
    result.generated_extensions =
        shared.generated_extensions.load(std::memory_order_relaxed);
    result.generated_row_extensions =
        shared.generated_row_extensions.load(std::memory_order_relaxed);
    result.generated_column_extensions =
        shared.generated_column_extensions.load(std::memory_order_relaxed);
    result.left_assignments =
        shared.left_assignments.load(std::memory_order_relaxed);
    result.right_assignments =
        shared.right_assignments.load(std::memory_order_relaxed);
    result.truncated_generators =
        shared.truncated_generators.load(std::memory_order_relaxed);
    result.seconds = elapsed_seconds(shared);
    return result;
}

}  // namespace xweigh_cross
