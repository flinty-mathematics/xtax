// xweigh_lift_exact.hpp: exhaustive 5-circulant block lift search.
//
// xweigh_lift explores lifts of the 44 inequivalent IW(7,25) templates by
// simulated annealing. This header solves the same lifting problem exactly:
// it enumerates every assignment of ternary 5-sequences (circulant first
// rows) to template cells and reports every weighing-matrix lift, or proves
// that a template admits none. The search runs row by row:
//
//   - a cell with template entry S may only hold a sequence whose entries
//     sum to S;
//   - a block row is feasible iff its supports sum to the weight and both
//     autocorrelation components sum to zero (the diagonal Gram blocks);
//   - every later block row must additionally zero all five cross
//     correlation sums against each earlier row (the off-diagonal Gram
//     blocks).
//
// Each row is enumerated with a meet-in-the-middle join: one half of the
// cells is hashed by its exact constraint contribution, the other half is
// walked depth first with per-component suffix intervals and probes the
// table for exact complements. Joins are verified componentwise, so hash
// collisions cannot lose or fabricate rows.
//
// Cyclic symmetry: rotating every block in block row i by s_i and every
// block in block column k by t_k rotates cell (i, k) by s_i + t_k and maps
// lifts to lifts of the same template. The search fixes this freedom
// exactly: cells of the first searched row are restricted to canonical
// rotations (constant sequences are trivially canonical), and every later
// row keeps only tuples whose first non-constant cell holds a canonical
// sequence. Each symmetry orbit of lifts retains at least one
// representative, so an empty search proves nonexistence for the class.
// --no-canonical disables both filters for validation runs.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stop_signal.hpp"
#include "xweigh_lift_search.hpp"

namespace xweigh_lift_exact {

using xweigh_lift::BLOCK_ORDER;
using xweigh_lift::SEQUENCE_COUNT;
using xweigh_lift::sequence_table;

constexpr int MAX_TEMPLATE_ORDER = 7;
// Component layout: [support sum, first autocorrelation sum, second
// autocorrelation sum, then five cross correlation sums per earlier row].
constexpr int KEY_CAP = 3 + BLOCK_ORDER * (MAX_TEMPLATE_ORDER - 1);
// Largest per-cell candidate list (ternary 5-sequences with sum zero).
constexpr int MAX_LIST_SIZE = 51;
constexpr uint64_t MAP_PRODUCT_CAP = uint64_t(1) << 22;
// Root row lifts are materialized for work distribution; refuse absurd
// portfolios (only reachable with --no-canonical on large templates).
constexpr uint64_t ROOT_ROWS_CAP = uint64_t(1) << 27;

struct Params {
    int template_index = -1;   // -1 = all built-in classes
    std::string template_file; // overrides the built-in templates
    int threads = 1;
    bool count_only = false;
    bool no_canonical = false;
    bool find_all = false;
    uint64_t node_limit = 0;   // 0 = unlimited
    double max_seconds = 0.0;  // 0 = unlimited
    std::string output = "exact_W35_25.csv";
};

using Key = std::array<int8_t, KEY_CAP>;

struct KeyHash {
    size_t operator()(const Key& key) const {
        uint64_t hash = 14695981039346656037ull;
        for (const int8_t component : key) {
            hash ^= static_cast<uint8_t>(component);
            hash *= 1099511628211ull;
        }
        return static_cast<size_t>(hash);
    }
};

inline uint64_t hash_components(const int16_t* components, int count) {
    uint64_t hash = 14695981039346656037ull;
    for (int index = 0; index < count; ++index) {
        hash ^= static_cast<uint8_t>(
            static_cast<int8_t>(components[index]));
        hash *= 1099511628211ull;
    }
    return hash;
}

// Open-addressing multimap from a component hash to packed cell choices.
// Entries with equal hashes chain through the probe sequence; callers
// verify the actual components, so collisions are harmless.
struct JoinTable {
    std::vector<uint64_t> hashes;
    std::vector<uint64_t> packeds;
    std::vector<uint32_t> slots;
    uint32_t mask = 0;

    void reset(uint64_t expected) {
        hashes.clear();
        packeds.clear();
        uint64_t size = 16;
        while (size < 2 * (expected + 1)) size <<= 1;
        slots.assign(static_cast<size_t>(size), 0);
        mask = static_cast<uint32_t>(size - 1);
    }

    void insert(uint64_t hash, uint64_t packed) {
        hashes.push_back(hash);
        packeds.push_back(packed);
        uint32_t slot = static_cast<uint32_t>(hash) & mask;
        while (slots[slot] != 0) slot = (slot + 1) & mask;
        slots[slot] = static_cast<uint32_t>(hashes.size());
    }

    bool empty() const { return hashes.empty(); }

    template <typename Visit>
    void for_each_match(uint64_t hash, Visit&& visit) const {
        uint32_t slot = static_cast<uint32_t>(hash) & mask;
        while (slots[slot] != 0) {
            const uint32_t index = slots[slot] - 1;
            if (hashes[index] == hash) visit(packeds[index]);
            slot = (slot + 1) & mask;
        }
    }
};

struct SequenceMeta {
    bool constant = false;
    bool canonical = false;
};

inline int encode_values(const std::array<int8_t, BLOCK_ORDER>& values) {
    int code = 0;
    for (int index = BLOCK_ORDER - 1; index >= 0; --index) {
        code = code * 3 + values[static_cast<size_t>(index)] + 1;
    }
    return code;
}

inline const std::array<SequenceMeta, SEQUENCE_COUNT>& sequence_meta() {
    static const std::array<SequenceMeta, SEQUENCE_COUNT> table = [] {
        std::array<SequenceMeta, SEQUENCE_COUNT> result{};
        for (int code = 0; code < SEQUENCE_COUNT; ++code) {
            const auto& values =
                sequence_table().sequence(static_cast<uint8_t>(code)).values;
            bool constant = true;
            for (int index = 1; index < BLOCK_ORDER; ++index) {
                constant = constant &&
                           values[static_cast<size_t>(index)] == values[0];
            }
            int min_code = code;
            for (int shift = 1; shift < BLOCK_ORDER; ++shift) {
                std::array<int8_t, BLOCK_ORDER> rotated{};
                for (int index = 0; index < BLOCK_ORDER; ++index) {
                    rotated[static_cast<size_t>(index)] =
                        values[static_cast<size_t>(
                            (index + shift) % BLOCK_ORDER)];
                }
                min_code = std::min(min_code, encode_values(rotated));
            }
            result[static_cast<size_t>(code)].constant = constant;
            result[static_cast<size_t>(code)].canonical = min_code == code;
        }
        return result;
    }();
    return table;
}

// True when the first non-constant cell (in template column order) holds a
// canonical sequence. Rotating a whole block row rotates every cell by the
// same amount and 5 is prime, so each row-shift orbit keeps exactly one
// such tuple.
inline bool row_is_canonical(const uint8_t* codes, int order) {
    const auto& meta = sequence_meta();
    for (int cell = 0; cell < order; ++cell) {
        const SequenceMeta& entry = meta[codes[cell]];
        if (!entry.constant) return entry.canonical;
    }
    return true;
}

struct TemplateInput {
    std::string name;
    int order = 0;
    int weight = 0;
    std::array<int8_t, MAX_TEMPLATE_ORDER * MAX_TEMPLATE_ORDER> entries{};

    int entry(int row, int column) const {
        return entries[static_cast<size_t>(row * order + column)];
    }
};

inline void validate_template(const TemplateInput& tmpl) {
    if (tmpl.order < 1 || tmpl.order > MAX_TEMPLATE_ORDER) {
        throw std::runtime_error(
            "template order must be in [1, 7]: " + tmpl.name);
    }
    for (int row = 0; row < tmpl.order; ++row) {
        for (int column = 0; column < tmpl.order; ++column) {
            const int value = tmpl.entry(row, column);
            if (value < -BLOCK_ORDER || value > BLOCK_ORDER) {
                throw std::runtime_error(
                    "template entries must lie in [-5, 5]: " + tmpl.name);
            }
        }
    }
    if (tmpl.weight < 1 || tmpl.weight > BLOCK_ORDER * tmpl.order) {
        throw std::runtime_error(
            "template weight must lie in [1, 5 * order]: " + tmpl.name);
    }
    for (int first = 0; first < tmpl.order; ++first) {
        for (int second = 0; second < tmpl.order; ++second) {
            int dot = 0;
            for (int column = 0; column < tmpl.order; ++column) {
                dot += tmpl.entry(first, column) *
                       tmpl.entry(second, column);
            }
            const int target = first == second ? tmpl.weight : 0;
            if (dot != target) {
                throw std::runtime_error(
                    "template rows are not orthogonal with equal norm: " +
                    tmpl.name);
            }
        }
    }
}

inline TemplateInput built_in_template(int index) {
    const auto& record =
        xweigh_lift::IW7_TEMPLATES[static_cast<size_t>(index)];
    TemplateInput tmpl;
    tmpl.name = std::string(record.name);
    tmpl.order = xweigh_lift::TEMPLATE_ORDER;
    tmpl.weight = 25;
    for (size_t cell = 0; cell < record.entries.size(); ++cell) {
        tmpl.entries[cell] = record.entries[cell];
    }
    validate_template(tmpl);
    return tmpl;
}

inline TemplateInput read_template_csv(const std::string& filename) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("failed to open template file: " + filename);
    }
    std::vector<std::vector<int>> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::stringstream stream(line);
        std::vector<int> row;
        std::string token;
        while (std::getline(stream, token, ',')) {
            size_t consumed = 0;
            int value = 0;
            try {
                value = std::stoi(token, &consumed);
            } catch (const std::exception&) {
                consumed = 0;
            }
            while (consumed < token.size() &&
                   (token[consumed] == ' ' || token[consumed] == '\r')) {
                ++consumed;
            }
            if (consumed != token.size()) {
                throw std::runtime_error(
                    "invalid integer entry in template file");
            }
            row.push_back(value);
        }
        rows.push_back(std::move(row));
    }
    const int order = static_cast<int>(rows.size());
    if (order < 1 || order > MAX_TEMPLATE_ORDER) {
        throw std::runtime_error("template order must be in [1, 7]");
    }
    TemplateInput tmpl;
    tmpl.name = std::filesystem::path(filename).stem().string();
    tmpl.order = order;
    for (int row = 0; row < order; ++row) {
        if (static_cast<int>(rows[static_cast<size_t>(row)].size()) !=
            order) {
            throw std::runtime_error("template file must be square");
        }
        for (int column = 0; column < order; ++column) {
            const int value =
                rows[static_cast<size_t>(row)][static_cast<size_t>(column)];
            if (value < -BLOCK_ORDER || value > BLOCK_ORDER) {
                throw std::runtime_error(
                    "template entries must lie in [-5, 5]");
            }
            tmpl.entries[static_cast<size_t>(row * order + column)] =
                static_cast<int8_t>(value);
        }
    }
    int weight = 0;
    for (int column = 0; column < order; ++column) {
        weight += tmpl.entry(0, column) * tmpl.entry(0, column);
    }
    tmpl.weight = weight;
    validate_template(tmpl);
    return tmpl;
}

struct DepthPlan {
    int row = 0;
    int key_size = 3;
    std::array<std::vector<uint8_t>, MAX_TEMPLATE_ORDER> lists{};
    // Per-cell (support, first, second) for each candidate, in list order.
    std::array<std::vector<std::array<int8_t, 3>>, MAX_TEMPLATE_ORDER>
        signatures{};
    std::vector<int> map_cells;
    std::vector<int> stream_cells;
    uint64_t map_product = 1;
    uint64_t stream_product = 1;
    std::array<int16_t, MAX_TEMPLATE_ORDER> min_support{};
    std::array<int16_t, MAX_TEMPLATE_ORDER> max_support{};
};

inline std::vector<uint8_t> sequences_with_sum(int sum, bool canonical_only) {
    std::vector<uint8_t> result;
    const auto& meta = sequence_meta();
    for (int code = 0; code < SEQUENCE_COUNT; ++code) {
        const auto& sequence =
            sequence_table().sequence(static_cast<uint8_t>(code));
        if (sequence.sum != sum) continue;
        if (canonical_only &&
            !meta[static_cast<size_t>(code)].canonical) {
            continue;
        }
        result.push_back(static_cast<uint8_t>(code));
    }
    return result;
}

inline void choose_split(DepthPlan& plan, int order) {
    uint64_t best_cost = 0;
    uint32_t best_mask = 0;
    bool have_best = false;
    for (uint32_t mask = 0; mask < (uint32_t(1) << order); ++mask) {
        uint64_t map_product = 1;
        uint64_t stream_product = 1;
        for (int cell = 0; cell < order; ++cell) {
            const uint64_t size =
                plan.lists[static_cast<size_t>(cell)].size();
            if ((mask >> cell) & 1u) {
                map_product *= size;
            } else {
                stream_product *= size;
            }
        }
        if (map_product > MAP_PRODUCT_CAP) continue;
        const uint64_t cost = map_product + stream_product;
        if (!have_best || cost < best_cost) {
            have_best = true;
            best_cost = cost;
            best_mask = mask;
        }
    }
    plan.map_cells.clear();
    plan.stream_cells.clear();
    plan.map_product = 1;
    plan.stream_product = 1;
    for (int cell = 0; cell < order; ++cell) {
        const uint64_t size = plan.lists[static_cast<size_t>(cell)].size();
        if ((best_mask >> cell) & 1u) {
            plan.map_cells.push_back(cell);
            plan.map_product *= size;
        } else {
            plan.stream_cells.push_back(cell);
            plan.stream_product *= size;
        }
    }
    // Walk small candidate lists first so stream pruning cuts early.
    std::stable_sort(
        plan.stream_cells.begin(), plan.stream_cells.end(),
        [&plan](int left, int right) {
            return plan.lists[static_cast<size_t>(left)].size() <
                   plan.lists[static_cast<size_t>(right)].size();
        });
}

// Deep rows are overdetermined and most of their joins fail, so the fixed
// per-node cost matters more than balance: hash only the two largest cell
// lists and let the many-component interval pruning carry the stream walk.
inline void choose_small_map_split(DepthPlan& plan, int order) {
    std::vector<int> cells;
    for (int cell = 0; cell < order; ++cell) cells.push_back(cell);
    std::stable_sort(cells.begin(), cells.end(),
                     [&plan](int left, int right) {
                         return plan.lists[static_cast<size_t>(left)].size() >
                                plan.lists[static_cast<size_t>(right)].size();
                     });
    const size_t map_count = std::min<size_t>(2, cells.size());
    plan.map_cells.assign(cells.begin(),
                          cells.begin() + static_cast<ptrdiff_t>(map_count));
    plan.stream_cells.assign(
        cells.begin() + static_cast<ptrdiff_t>(map_count), cells.end());
    plan.map_product = 1;
    for (const int cell : plan.map_cells) {
        plan.map_product *= plan.lists[static_cast<size_t>(cell)].size();
    }
    plan.stream_product = 1;
    for (const int cell : plan.stream_cells) {
        plan.stream_product *= plan.lists[static_cast<size_t>(cell)].size();
    }
    std::stable_sort(
        plan.stream_cells.begin(), plan.stream_cells.end(),
        [&plan](int left, int right) {
            return plan.lists[static_cast<size_t>(left)].size() <
                   plan.lists[static_cast<size_t>(right)].size();
        });
}

inline DepthPlan build_depth_plan(const TemplateInput& tmpl, int row,
                                  bool canonical_cells) {
    DepthPlan plan;
    plan.row = row;
    for (int cell = 0; cell < tmpl.order; ++cell) {
        auto& list = plan.lists[static_cast<size_t>(cell)];
        list = sequences_with_sum(tmpl.entry(row, cell), canonical_cells);
        if (list.empty() ||
            static_cast<int>(list.size()) > MAX_LIST_SIZE) {
            throw std::logic_error("unexpected candidate list size");
        }
        auto& signatures = plan.signatures[static_cast<size_t>(cell)];
        signatures.resize(list.size());
        int16_t min_support = BLOCK_ORDER;
        int16_t max_support = 0;
        for (size_t index = 0; index < list.size(); ++index) {
            const auto& signature =
                sequence_table().sequence(list[index]).signature;
            signatures[index][0] = static_cast<int8_t>(signature.support);
            signatures[index][1] = signature.first;
            signatures[index][2] = signature.second;
            min_support = std::min(
                min_support, static_cast<int16_t>(signature.support));
            max_support = std::max(
                max_support, static_cast<int16_t>(signature.support));
        }
        plan.min_support[static_cast<size_t>(cell)] = min_support;
        plan.max_support[static_cast<size_t>(cell)] = max_support;
    }
    choose_split(plan, tmpl.order);
    return plan;
}

// Enumerates every tuple over the given cells, accumulating the three
// diagonal components and the packed per-cell list indices (8 bits each).
template <typename Emit>
inline void enumerate_tuples(const DepthPlan& plan,
                             const std::vector<int>& cells, size_t position,
                             std::array<int16_t, 3>& partial,
                             uint64_t packed, Emit&& emit) {
    if (position == cells.size()) {
        emit(partial, packed);
        return;
    }
    const int cell = cells[position];
    const auto& list = plan.lists[static_cast<size_t>(cell)];
    const auto& signatures = plan.signatures[static_cast<size_t>(cell)];
    for (size_t index = 0; index < list.size(); ++index) {
        for (int component = 0; component < 3; ++component) {
            partial[static_cast<size_t>(component)] = static_cast<int16_t>(
                partial[static_cast<size_t>(component)] +
                signatures[index][static_cast<size_t>(component)]);
        }
        enumerate_tuples(
            plan, cells, position + 1, partial,
            packed | (static_cast<uint64_t>(index) << (8 * cell)), emit);
        for (int component = 0; component < 3; ++component) {
            partial[static_cast<size_t>(component)] = static_cast<int16_t>(
                partial[static_cast<size_t>(component)] -
                signatures[index][static_cast<size_t>(component)]);
        }
    }
}

// Number of block-row lifts satisfying the diagonal Gram condition alone.
inline uint64_t count_row_lifts(const DepthPlan& plan, int weight) {
    std::unordered_map<Key, uint64_t, KeyHash> buckets;
    std::array<int16_t, 3> partial{};
    enumerate_tuples(
        plan, plan.map_cells, 0, partial, 0,
        [&buckets](const std::array<int16_t, 3>& components, uint64_t) {
            Key key{};
            for (int component = 0; component < 3; ++component) {
                key[static_cast<size_t>(component)] = static_cast<int8_t>(
                    components[static_cast<size_t>(component)]);
            }
            ++buckets[key];
        });
    uint64_t total = 0;
    partial = {};
    enumerate_tuples(
        plan, plan.stream_cells, 0, partial, 0,
        [&buckets, weight, &total](
            const std::array<int16_t, 3>& components, uint64_t) {
            Key key{};
            key[0] = static_cast<int8_t>(weight - components[0]);
            key[1] = static_cast<int8_t>(-components[1]);
            key[2] = static_cast<int8_t>(-components[2]);
            const auto found = buckets.find(key);
            if (found != buckets.end()) total += found->second;
        });
    return total;
}

struct ClassPlan {
    TemplateInput tmpl;
    std::vector<DepthPlan> depths;
    // suffix_*_support[d][cell]: bounds on the support still to be added to
    // a block column by the rows at depths d, d + 1, ...
    std::vector<std::array<int16_t, MAX_TEMPLATE_ORDER>> suffix_min_support;
    std::vector<std::array<int16_t, MAX_TEMPLATE_ORDER>> suffix_max_support;
    std::vector<uint64_t> row_counts_raw;      // indexed by template row
    std::vector<uint64_t> row_counts_filtered; // root-style cell filtering
};

inline ClassPlan build_class_plan(const TemplateInput& tmpl,
                                  const Params& params) {
    ClassPlan plan;
    plan.tmpl = tmpl;
    plan.row_counts_raw.assign(static_cast<size_t>(tmpl.order), 0);
    plan.row_counts_filtered.assign(static_cast<size_t>(tmpl.order), 0);
    for (int row = 0; row < tmpl.order; ++row) {
        const DepthPlan raw = build_depth_plan(tmpl, row, false);
        plan.row_counts_raw[static_cast<size_t>(row)] =
            count_row_lifts(raw, tmpl.weight);
        if (params.no_canonical) {
            plan.row_counts_filtered[static_cast<size_t>(row)] =
                plan.row_counts_raw[static_cast<size_t>(row)];
        } else {
            const DepthPlan filtered = build_depth_plan(tmpl, row, true);
            plan.row_counts_filtered[static_cast<size_t>(row)] =
                count_row_lifts(filtered, tmpl.weight);
        }
    }

    int root = 0;
    for (int row = 1; row < tmpl.order; ++row) {
        if (plan.row_counts_filtered[static_cast<size_t>(row)] <
            plan.row_counts_filtered[static_cast<size_t>(root)]) {
            root = row;
        }
    }
    std::vector<int> order;
    order.push_back(root);
    std::vector<int> rest;
    for (int row = 0; row < tmpl.order; ++row) {
        if (row != root) rest.push_back(row);
    }
    std::stable_sort(rest.begin(), rest.end(),
                     [&plan](int left, int right) {
                         return plan.row_counts_raw[
                                    static_cast<size_t>(left)] <
                                plan.row_counts_raw[
                                    static_cast<size_t>(right)];
                     });
    order.insert(order.end(), rest.begin(), rest.end());

    for (int depth = 0; depth < tmpl.order; ++depth) {
        DepthPlan depth_plan = build_depth_plan(
            tmpl, order[static_cast<size_t>(depth)],
            depth == 0 && !params.no_canonical);
        depth_plan.key_size = 3 + BLOCK_ORDER * depth;
        if (depth >= 2) choose_small_map_split(depth_plan, tmpl.order);
        plan.depths.push_back(std::move(depth_plan));
    }

    plan.suffix_min_support.assign(
        static_cast<size_t>(tmpl.order + 1),
        std::array<int16_t, MAX_TEMPLATE_ORDER>{});
    plan.suffix_max_support.assign(
        static_cast<size_t>(tmpl.order + 1),
        std::array<int16_t, MAX_TEMPLATE_ORDER>{});
    for (int depth = tmpl.order - 1; depth >= 0; --depth) {
        for (int cell = 0; cell < tmpl.order; ++cell) {
            plan.suffix_min_support[static_cast<size_t>(depth)]
                                   [static_cast<size_t>(cell)] =
                static_cast<int16_t>(
                    plan.suffix_min_support[static_cast<size_t>(depth + 1)]
                                           [static_cast<size_t>(cell)] +
                    plan.depths[static_cast<size_t>(depth)]
                        .min_support[static_cast<size_t>(cell)]);
            plan.suffix_max_support[static_cast<size_t>(depth)]
                                   [static_cast<size_t>(cell)] =
                static_cast<int16_t>(
                    plan.suffix_max_support[static_cast<size_t>(depth + 1)]
                                           [static_cast<size_t>(cell)] +
                    plan.depths[static_cast<size_t>(depth)]
                        .max_support[static_cast<size_t>(cell)]);
        }
    }
    return plan;
}

// Expansion and independent verification. Cell (i, k) expands to the 5 by 5
// circulant whose (r, c) entry is values[(c - r) mod 5]; codes are indexed
// by template row and column.
inline std::vector<int> expand_solution(const TemplateInput& tmpl,
                                        const std::vector<uint8_t>& codes) {
    const int n = BLOCK_ORDER * tmpl.order;
    std::vector<int> matrix(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (int block_row = 0; block_row < tmpl.order; ++block_row) {
        for (int block_column = 0; block_column < tmpl.order;
             ++block_column) {
            const auto& values =
                sequence_table()
                    .sequence(codes[static_cast<size_t>(
                        block_row * tmpl.order + block_column)])
                    .values;
            for (int row = 0; row < BLOCK_ORDER; ++row) {
                for (int column = 0; column < BLOCK_ORDER; ++column) {
                    matrix[static_cast<size_t>(
                        (block_row * BLOCK_ORDER + row) * n +
                        block_column * BLOCK_ORDER + column)] =
                        values[static_cast<size_t>(
                            (column - row + BLOCK_ORDER) % BLOCK_ORDER)];
                }
            }
        }
    }
    return matrix;
}

inline bool verify_expanded(const TemplateInput& tmpl,
                            const std::vector<int>& matrix) {
    const int n = BLOCK_ORDER * tmpl.order;
    for (int first = 0; first < n; ++first) {
        for (int second = first; second < n; ++second) {
            int dot = 0;
            for (int column = 0; column < n; ++column) {
                dot += matrix[static_cast<size_t>(first * n + column)] *
                       matrix[static_cast<size_t>(second * n + column)];
            }
            const int target = first == second ? tmpl.weight : 0;
            if (dot != target) return false;
        }
    }
    return true;
}

inline void write_matrix_csv(const std::vector<int>& matrix, int n,
                             const std::string& filename) {
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + filename);
    }
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            if (column > 0) output << ',';
            output << matrix[static_cast<size_t>(row * n + column)];
        }
        output << '\n';
    }
}

struct ClassOutcome {
    bool solved = false;
    bool completed = false;
    uint64_t solutions = 0;
    uint64_t nodes = 0;
    std::array<uint64_t, MAX_TEMPLATE_ORDER> depth_nodes{};
    double seconds = 0.0;
    std::vector<uint8_t> solution; // order * order codes, template layout
    std::vector<int> row_order;
    std::vector<uint64_t> row_counts_raw;
    std::vector<uint64_t> row_counts_filtered;
};

class ClassSearch {
public:
    ClassSearch(const ClassPlan& plan, const Params& params,
                std::optional<std::chrono::steady_clock::time_point>
                    deadline,
                std::string label)
        : plan_(plan), params_(params), deadline_(deadline),
          label_(std::move(label)) {}

    ClassOutcome run() {
        const auto start = std::chrono::steady_clock::now();
        build_root_rows();

        std::vector<std::thread> workers;
        const int thread_count = std::max(1, params_.threads);
        active_workers_.store(thread_count, std::memory_order_relaxed);
        workers.reserve(static_cast<size_t>(thread_count));
        for (int index = 0; index < thread_count; ++index) {
            workers.emplace_back([this] { worker(); });
        }

        auto last_report = start;
        while (active_workers_.load(std::memory_order_relaxed) > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const auto now = std::chrono::steady_clock::now();
            if (g_stop.load(std::memory_order_relaxed) ||
                (deadline_.has_value() && now >= *deadline_)) {
                external_stop_ = true;
                stop_.store(true, std::memory_order_relaxed);
            }
            if (now - last_report >= std::chrono::seconds(5)) {
                last_report = now;
                const uint64_t consumed = std::min<uint64_t>(
                    next_root_.load(std::memory_order_relaxed),
                    root_rows_.size());
                const double percent =
                    !root_rows_.empty()
                        ? 100.0 * static_cast<double>(consumed) /
                              static_cast<double>(root_rows_.size())
                        : 100.0;
                const double elapsed =
                    std::chrono::duration<double>(now - start).count();
                std::cout << "[progress] class=" << label_
                          << " root=" << percent << "% nodes="
                          << nodes_.load(std::memory_order_relaxed)
                          << " depth_nodes=" << depth_nodes_string()
                          << " solutions="
                          << solutions_.load(std::memory_order_relaxed)
                          << " elapsed=" << elapsed << "s\n"
                          << std::flush;
            }
        }
        for (auto& worker_thread : workers) worker_thread.join();

        ClassOutcome outcome;
        outcome.solutions = solutions_.load(std::memory_order_relaxed);
        outcome.solved = outcome.solutions > 0;
        outcome.nodes = nodes_.load(std::memory_order_relaxed);
        for (int depth = 0; depth < plan_.tmpl.order; ++depth) {
            outcome.depth_nodes[static_cast<size_t>(depth)] =
                depth_nodes_[static_cast<size_t>(depth)].load(
                    std::memory_order_relaxed);
        }
        outcome.completed = !external_stop_ &&
                            !limit_stop_.load(std::memory_order_relaxed) &&
                            (!outcome.solved || params_.find_all);
        outcome.seconds = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - start)
                              .count();
        outcome.solution = first_solution_;
        for (const auto& depth : plan_.depths) {
            outcome.row_order.push_back(depth.row);
        }
        outcome.row_counts_raw = plan_.row_counts_raw;
        outcome.row_counts_filtered = plan_.row_counts_filtered;
        return outcome;
    }

private:
    struct Scratch {
        // contrib[cell][candidate][component], candidate in list order.
        std::array<std::array<std::array<int8_t, KEY_CAP>, MAX_LIST_SIZE>,
                   MAX_TEMPLATE_ORDER>
            contrib{};
        // Suffix component intervals over stream cell positions.
        std::array<std::array<int16_t, KEY_CAP>, MAX_TEMPLATE_ORDER + 1>
            suffix_min{};
        std::array<std::array<int16_t, KEY_CAP>, MAX_TEMPLATE_ORDER + 1>
            suffix_max{};
        std::array<int16_t, KEY_CAP> partial{};
        std::array<int16_t, KEY_CAP> map_min{};
        std::array<int16_t, KEY_CAP> map_max{};
        std::array<uint8_t, MAX_TEMPLATE_ORDER> stream_choice{};
        JoinTable table;
    };

    struct WorkerState {
        std::array<std::array<uint8_t, MAX_TEMPLATE_ORDER>,
                   MAX_TEMPLATE_ORDER>
            codes{};
        std::array<int16_t, MAX_TEMPLATE_ORDER> col_support{};
        std::vector<Scratch> scratch;
        uint64_t pending_nodes = 0;
    };

    std::string depth_nodes_string() const {
        std::string result = "[";
        for (int depth = 0; depth < plan_.tmpl.order; ++depth) {
            if (depth > 0) result += ' ';
            result += std::to_string(
                depth_nodes_[static_cast<size_t>(depth)].load(
                    std::memory_order_relaxed));
        }
        result += ']';
        return result;
    }

    // Materialize every root row lift (packed as 8-bit sequence codes per
    // cell); workers then pull individual root rows for load balancing.
    void build_root_rows() {
        const DepthPlan& root = plan_.depths[0];
        JoinTable table;
        table.reset(root.map_product);
        std::array<int16_t, 3> partial{};
        enumerate_tuples(
            root, root.map_cells, 0, partial, 0,
            [&table](const std::array<int16_t, 3>& components,
                     uint64_t packed) {
                table.insert(hash_components(components.data(), 3),
                             packed);
            });
        partial = {};
        root_rows_.clear();
        enumerate_tuples(
            root, root.stream_cells, 0, partial, 0,
            [this, &root, &table](
                const std::array<int16_t, 3>& components,
                uint64_t stream_packed) {
                std::array<int16_t, 3> need{};
                need[0] = static_cast<int16_t>(plan_.tmpl.weight -
                                               components[0]);
                need[1] = static_cast<int16_t>(-components[1]);
                need[2] = static_cast<int16_t>(-components[2]);
                const uint64_t hash = hash_components(need.data(), 3);
                table.for_each_match(hash, [&](uint64_t map_packed) {
                    std::array<int16_t, 3> sums{};
                    for (const int cell : root.map_cells) {
                        const size_t choice = static_cast<size_t>(
                            (map_packed >> (8 * cell)) & 0xFF);
                        const auto& signature =
                            root.signatures[static_cast<size_t>(cell)]
                                           [choice];
                        for (int component = 0; component < 3;
                             ++component) {
                            sums[static_cast<size_t>(component)] =
                                static_cast<int16_t>(
                                    sums[static_cast<size_t>(component)] +
                                    signature[static_cast<size_t>(
                                        component)]);
                        }
                    }
                    if (sums != need) return;
                    if (root_rows_.size() >= ROOT_ROWS_CAP) {
                        throw std::runtime_error(
                            "root row portfolio too large; run with the "
                            "canonical filters enabled");
                    }
                    uint64_t codes = 0;
                    for (const int cell : root.stream_cells) {
                        const size_t choice = static_cast<size_t>(
                            (stream_packed >> (8 * cell)) & 0xFF);
                        codes |= static_cast<uint64_t>(
                                     root.lists[static_cast<size_t>(cell)]
                                               [choice])
                                 << (8 * cell);
                    }
                    for (const int cell : root.map_cells) {
                        const size_t choice = static_cast<size_t>(
                            (map_packed >> (8 * cell)) & 0xFF);
                        codes |= static_cast<uint64_t>(
                                     root.lists[static_cast<size_t>(cell)]
                                               [choice])
                                 << (8 * cell);
                    }
                    root_rows_.push_back(codes);
                });
            });
    }

    void worker() {
        WorkerState state;
        state.scratch.resize(static_cast<size_t>(plan_.tmpl.order));
        const int order = plan_.tmpl.order;

        while (!stop_.load(std::memory_order_relaxed)) {
            const uint64_t index =
                next_root_.fetch_add(1, std::memory_order_relaxed);
            if (index >= root_rows_.size()) break;
            const uint64_t codes = root_rows_[index];
            for (int cell = 0; cell < order; ++cell) {
                state.codes[0][static_cast<size_t>(cell)] =
                    static_cast<uint8_t>((codes >> (8 * cell)) & 0xFF);
            }
            try_row(state, 0);
        }
        nodes_.fetch_add(state.pending_nodes, std::memory_order_relaxed);
        state.pending_nodes = 0;
        active_workers_.fetch_sub(1, std::memory_order_relaxed);
    }

    void bump_node(WorkerState& state) {
        if (++state.pending_nodes >= 4096) {
            const uint64_t total =
                nodes_.fetch_add(state.pending_nodes,
                                 std::memory_order_relaxed) +
                state.pending_nodes;
            state.pending_nodes = 0;
            if (params_.node_limit > 0 && total >= params_.node_limit) {
                limit_stop_.store(true, std::memory_order_relaxed);
                stop_.store(true, std::memory_order_relaxed);
            }
        }
    }

    // The codes for the row at `depth` are already in state.codes[depth];
    // apply the row filters and either record a solution or descend.
    void try_row(WorkerState& state, int depth) {
        bump_node(state);
        depth_nodes_[static_cast<size_t>(depth)].fetch_add(
            1, std::memory_order_relaxed);
        if (stop_.load(std::memory_order_relaxed)) return;
        const int order = plan_.tmpl.order;
        if (depth > 0 && !params_.no_canonical &&
            !row_is_canonical(state.codes[static_cast<size_t>(depth)].data(),
                              order)) {
            return;
        }
        for (int cell = 0; cell < order; ++cell) {
            state.col_support[static_cast<size_t>(cell)] =
                static_cast<int16_t>(
                    state.col_support[static_cast<size_t>(cell)] +
                    sequence_table()
                        .sequence(state.codes[static_cast<size_t>(depth)]
                                             [static_cast<size_t>(cell)])
                        .signature.support);
        }
        bool feasible = true;
        for (int cell = 0; cell < order; ++cell) {
            const int16_t support =
                state.col_support[static_cast<size_t>(cell)];
            if (support +
                        plan_.suffix_min_support[static_cast<size_t>(
                            depth + 1)][static_cast<size_t>(cell)] >
                    plan_.tmpl.weight ||
                support +
                        plan_.suffix_max_support[static_cast<size_t>(
                            depth + 1)][static_cast<size_t>(cell)] <
                    plan_.tmpl.weight) {
                feasible = false;
                break;
            }
        }
        if (feasible) {
            if (depth + 1 == order) {
                record_solution(state);
            } else {
                join_row(state, depth + 1);
            }
        }
        for (int cell = 0; cell < order; ++cell) {
            state.col_support[static_cast<size_t>(cell)] =
                static_cast<int16_t>(
                    state.col_support[static_cast<size_t>(cell)] -
                    sequence_table()
                        .sequence(state.codes[static_cast<size_t>(depth)]
                                             [static_cast<size_t>(cell)])
                        .signature.support);
        }
    }

    // Enumerate the row at `depth` with a per-node meet-in-the-middle join
    // against the rows already fixed in state.codes[0 .. depth - 1].
    void join_row(WorkerState& state, int depth) {
        const DepthPlan& plan = plan_.depths[static_cast<size_t>(depth)];
        Scratch& scratch = state.scratch[static_cast<size_t>(depth)];
        const int order = plan_.tmpl.order;
        const int key_size = plan.key_size;

        for (int cell = 0; cell < order; ++cell) {
            const auto& list = plan.lists[static_cast<size_t>(cell)];
            const auto& signatures =
                plan.signatures[static_cast<size_t>(cell)];
            for (size_t index = 0; index < list.size(); ++index) {
                auto& components =
                    scratch.contrib[static_cast<size_t>(cell)][index];
                components[0] = signatures[index][0];
                components[1] = signatures[index][1];
                components[2] = signatures[index][2];
                for (int previous = 0; previous < depth; ++previous) {
                    const uint8_t previous_code =
                        state.codes[static_cast<size_t>(previous)]
                                   [static_cast<size_t>(cell)];
                    for (int shift = 0; shift < BLOCK_ORDER; ++shift) {
                        components[static_cast<size_t>(
                            3 + BLOCK_ORDER * previous + shift)] =
                            static_cast<int8_t>(sequence_table().correlation(
                                previous_code, list[index], shift));
                    }
                }
            }
        }

        scratch.table.reset(plan.map_product);
        for (int component = 0; component < key_size; ++component) {
            scratch.map_min[static_cast<size_t>(component)] = 127;
            scratch.map_max[static_cast<size_t>(component)] = -128;
        }
        scratch.partial.fill(0);
        map_cell(state, depth, 0, 0);
        if (scratch.table.empty()) return;

        const size_t stream_count = plan.stream_cells.size();
        for (int component = 0; component < key_size; ++component) {
            scratch.suffix_min[stream_count]
                              [static_cast<size_t>(component)] = 0;
            scratch.suffix_max[stream_count]
                              [static_cast<size_t>(component)] = 0;
        }
        for (size_t position = stream_count; position > 0; --position) {
            const int cell = plan.stream_cells[position - 1];
            const auto& list = plan.lists[static_cast<size_t>(cell)];
            for (int component = 0; component < key_size; ++component) {
                int16_t cell_min = 127;
                int16_t cell_max = -128;
                for (size_t index = 0; index < list.size(); ++index) {
                    const int16_t value =
                        scratch.contrib[static_cast<size_t>(cell)][index]
                                       [static_cast<size_t>(component)];
                    cell_min = std::min(cell_min, value);
                    cell_max = std::max(cell_max, value);
                }
                scratch.suffix_min[position - 1]
                                  [static_cast<size_t>(component)] =
                    static_cast<int16_t>(
                        scratch.suffix_min[position]
                                          [static_cast<size_t>(component)] +
                        cell_min);
                scratch.suffix_max[position - 1]
                                  [static_cast<size_t>(component)] =
                    static_cast<int16_t>(
                        scratch.suffix_max[position]
                                          [static_cast<size_t>(component)] +
                        cell_max);
            }
        }

        scratch.partial.fill(0);
        stream_cell(state, depth, 0);
    }

    // Build the map half of the join for the row at `depth`.
    void map_cell(WorkerState& state, int depth, size_t position,
                  uint64_t packed) {
        const DepthPlan& plan = plan_.depths[static_cast<size_t>(depth)];
        Scratch& scratch = state.scratch[static_cast<size_t>(depth)];
        const int key_size = plan.key_size;
        if (position == plan.map_cells.size()) {
            scratch.table.insert(
                hash_components(scratch.partial.data(), key_size), packed);
            for (int component = 0; component < key_size; ++component) {
                scratch.map_min[static_cast<size_t>(component)] = std::min(
                    scratch.map_min[static_cast<size_t>(component)],
                    scratch.partial[static_cast<size_t>(component)]);
                scratch.map_max[static_cast<size_t>(component)] = std::max(
                    scratch.map_max[static_cast<size_t>(component)],
                    scratch.partial[static_cast<size_t>(component)]);
            }
            return;
        }
        const int cell = plan.map_cells[position];
        const auto& list = plan.lists[static_cast<size_t>(cell)];
        const int16_t column_support =
            state.col_support[static_cast<size_t>(cell)];
        const int16_t column_min =
            plan_.suffix_min_support[static_cast<size_t>(depth + 1)]
                                    [static_cast<size_t>(cell)];
        const int16_t column_max =
            plan_.suffix_max_support[static_cast<size_t>(depth + 1)]
                                    [static_cast<size_t>(cell)];
        for (size_t index = 0; index < list.size(); ++index) {
            const auto& components =
                scratch.contrib[static_cast<size_t>(cell)][index];
            const int16_t support = components[0];
            if (column_support + support + column_min > plan_.tmpl.weight ||
                column_support + support + column_max < plan_.tmpl.weight) {
                continue;
            }
            for (int component = 0; component < key_size; ++component) {
                scratch.partial[static_cast<size_t>(component)] =
                    static_cast<int16_t>(
                        scratch.partial[static_cast<size_t>(component)] +
                        components[static_cast<size_t>(component)]);
            }
            map_cell(state, depth, position + 1,
                     packed | (static_cast<uint64_t>(index) << (8 * cell)));
            for (int component = 0; component < key_size; ++component) {
                scratch.partial[static_cast<size_t>(component)] =
                    static_cast<int16_t>(
                        scratch.partial[static_cast<size_t>(component)] -
                        components[static_cast<size_t>(component)]);
            }
        }
    }

    // Walk the stream half; at each leaf probe the map for the exact
    // complement of the accumulated components.
    void stream_cell(WorkerState& state, int depth, size_t position) {
        if (stop_.load(std::memory_order_relaxed)) return;
        const DepthPlan& plan = plan_.depths[static_cast<size_t>(depth)];
        Scratch& scratch = state.scratch[static_cast<size_t>(depth)];
        const int key_size = plan.key_size;
        const int16_t weight = static_cast<int16_t>(plan_.tmpl.weight);
        if (position == plan.stream_cells.size()) {
            std::array<int16_t, KEY_CAP> need{};
            for (int component = 0; component < key_size; ++component) {
                const int16_t target =
                    component == 0 ? weight : int16_t(0);
                need[static_cast<size_t>(component)] =
                    static_cast<int16_t>(
                        target -
                        scratch.partial[static_cast<size_t>(component)]);
                if (need[static_cast<size_t>(component)] <
                        scratch.map_min[static_cast<size_t>(component)] ||
                    need[static_cast<size_t>(component)] >
                        scratch.map_max[static_cast<size_t>(component)]) {
                    return;
                }
            }
            const uint64_t hash = hash_components(need.data(), key_size);
            scratch.table.for_each_match(hash, [&](uint64_t packed) {
                if (stop_.load(std::memory_order_relaxed)) return;
                std::array<int16_t, KEY_CAP> sums{};
                for (const int cell : plan.map_cells) {
                    const size_t choice = static_cast<size_t>(
                        (packed >> (8 * cell)) & 0xFF);
                    const auto& components =
                        scratch.contrib[static_cast<size_t>(cell)][choice];
                    for (int component = 0; component < key_size;
                         ++component) {
                        sums[static_cast<size_t>(component)] =
                            static_cast<int16_t>(
                                sums[static_cast<size_t>(component)] +
                                components[static_cast<size_t>(
                                    component)]);
                    }
                }
                for (int component = 0; component < key_size;
                     ++component) {
                    if (sums[static_cast<size_t>(component)] !=
                        need[static_cast<size_t>(component)]) {
                        return;
                    }
                }
                auto& codes = state.codes[static_cast<size_t>(depth)];
                for (size_t stream_position = 0;
                     stream_position < plan.stream_cells.size();
                     ++stream_position) {
                    const int stream_cell_index =
                        plan.stream_cells[stream_position];
                    codes[static_cast<size_t>(stream_cell_index)] =
                        plan.lists[static_cast<size_t>(stream_cell_index)]
                                  [scratch.stream_choice[stream_position]];
                }
                for (const int map_cell_index : plan.map_cells) {
                    const size_t choice = static_cast<size_t>(
                        (packed >> (8 * map_cell_index)) & 0xFF);
                    codes[static_cast<size_t>(map_cell_index)] =
                        plan.lists[static_cast<size_t>(map_cell_index)]
                                  [choice];
                }
                try_row(state, depth);
            });
            return;
        }
        const int cell = plan.stream_cells[position];
        const auto& list = plan.lists[static_cast<size_t>(cell)];
        const int16_t column_support =
            state.col_support[static_cast<size_t>(cell)];
        const int16_t column_min =
            plan_.suffix_min_support[static_cast<size_t>(depth + 1)]
                                    [static_cast<size_t>(cell)];
        const int16_t column_max =
            plan_.suffix_max_support[static_cast<size_t>(depth + 1)]
                                    [static_cast<size_t>(cell)];
        for (size_t index = 0; index < list.size(); ++index) {
            const auto& components =
                scratch.contrib[static_cast<size_t>(cell)][index];
            const int16_t support = components[0];
            if (column_support + support + column_min > weight ||
                column_support + support + column_max < weight) {
                continue;
            }
            for (int component = 0; component < key_size; ++component) {
                scratch.partial[static_cast<size_t>(component)] =
                    static_cast<int16_t>(
                        scratch.partial[static_cast<size_t>(component)] +
                        components[static_cast<size_t>(component)]);
            }
            bool feasible = true;
            for (int component = 0; component < key_size; ++component) {
                const int16_t target =
                    component == 0 ? weight : int16_t(0);
                const int16_t value =
                    scratch.partial[static_cast<size_t>(component)];
                if (value +
                            scratch.suffix_min[position + 1]
                                              [static_cast<size_t>(
                                                  component)] +
                            scratch.map_min[static_cast<size_t>(
                                component)] >
                        target ||
                    value +
                            scratch.suffix_max[position + 1]
                                              [static_cast<size_t>(
                                                  component)] +
                            scratch.map_max[static_cast<size_t>(
                                component)] <
                        target) {
                    feasible = false;
                    break;
                }
            }
            if (feasible) {
                scratch.stream_choice[position] =
                    static_cast<uint8_t>(index);
                stream_cell(state, depth, position + 1);
            }
            for (int component = 0; component < key_size; ++component) {
                scratch.partial[static_cast<size_t>(component)] =
                    static_cast<int16_t>(
                        scratch.partial[static_cast<size_t>(component)] -
                        components[static_cast<size_t>(component)]);
            }
            if (stop_.load(std::memory_order_relaxed)) return;
        }
    }

    void record_solution(WorkerState& state) {
        const int order = plan_.tmpl.order;
        std::vector<uint8_t> codes(
            static_cast<size_t>(order) * static_cast<size_t>(order));
        for (int depth = 0; depth < order; ++depth) {
            const int row = plan_.depths[static_cast<size_t>(depth)].row;
            for (int cell = 0; cell < order; ++cell) {
                codes[static_cast<size_t>(row * order + cell)] =
                    state.codes[static_cast<size_t>(depth)]
                               [static_cast<size_t>(cell)];
            }
        }
        const std::vector<int> matrix = expand_solution(plan_.tmpl, codes);
        if (!verify_expanded(plan_.tmpl, matrix)) {
            std::lock_guard<std::mutex> lock(solution_mutex_);
            std::cout << "[error] class=" << label_
                      << " candidate failed exact verification\n"
                      << std::flush;
            return;
        }
        {
            std::lock_guard<std::mutex> lock(solution_mutex_);
            if (first_solution_.empty()) first_solution_ = codes;
        }
        solutions_.fetch_add(1, std::memory_order_relaxed);
        if (!params_.find_all) {
            stop_.store(true, std::memory_order_relaxed);
        }
    }

    const ClassPlan& plan_;
    const Params& params_;
    std::optional<std::chrono::steady_clock::time_point> deadline_;
    std::string label_;

    std::vector<uint64_t> root_rows_;

    std::atomic<uint64_t> next_root_{0};
    std::atomic<uint64_t> nodes_{0};
    std::array<std::atomic<uint64_t>, MAX_TEMPLATE_ORDER> depth_nodes_{};
    std::atomic<uint64_t> solutions_{0};
    std::atomic<bool> stop_{false};
    std::atomic<int> active_workers_{0};
    bool external_stop_ = false;
    std::atomic<bool> limit_stop_{false};
    std::mutex solution_mutex_;
    std::vector<uint8_t> first_solution_;
};

inline ClassOutcome search_class(
    const TemplateInput& tmpl, const Params& params,
    std::optional<std::chrono::steady_clock::time_point> deadline,
    const std::string& label) {
    const ClassPlan plan = build_class_plan(tmpl, params);
    ClassSearch search(plan, params, deadline, label);
    return search.run();
}

}  // namespace xweigh_lift_exact
