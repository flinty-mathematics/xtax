// xweigh_anneal.hpp: standalone fixed-degree ternary matrix annealer.

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

namespace xweigh {

constexpr int MAX_ORDER = std::numeric_limits<int16_t>::max();

struct Params {
    int threads = 1;
    uint64_t seed = 0;
    double max_seconds = 0.0;
    double save_interval = 2.0;
    std::string output = "best_W.csv";

    double sign_fraction = 0.5;
    double greedy_fraction = 0.5;
    int candidate_samples = 4;
    double target_fraction = 0.7;
    int target_samples = 8;

    bool tempering = true;
    int exchange_interval = 2000;
    double t_init = 0.0;
    double t_min = 0.25;
    double cooling = 0.999;
    int moves_per_cool = 500;
    int stuck_threshold = 50000;
    double reheat = 1.0;
    double reseed_factor = 1.25;

    bool pin_threads = true;
    bool use_hyperthreads = false;
};

enum class MoveKind : uint8_t {
    sign_flip,
    support_switch
};

struct Move {
    MoveKind kind = MoveKind::sign_flip;
    int row1 = 0;
    int row2 = 0;
    int col1 = 0;
    int col2 = 0;
    int minority_pos1 = 0;
    int minority_pos2 = 0;
};

struct Evaluation {
    int64_t score_delta = 0;
    int64_t conflict_delta = 0;
};

inline bool better_pair(int64_t score_a, int64_t conflicts_a,
                        int64_t score_b, int64_t conflicts_b) {
    return score_a < score_b ||
           (score_a == score_b && conflicts_a < conflicts_b);
}

inline int random_index(std::mt19937& rng, int count) {
    return std::uniform_int_distribution<int>(0, count - 1)(rng);
}

inline double random_unit(std::mt19937& rng) {
    return std::generate_canonical<double, 53>(rng);
}

class State {
public:
    State() = default;

    State(int order, int weight)
        : n_(order),
          w_(weight),
          sparse_side_(weight <= order - weight),
          minority_count_(std::min(weight, order - weight)),
          entries_((size_t)order * order, int8_t(0)),
          gram_((size_t)order * order, int16_t(0)),
          row_residual_((size_t)order, int64_t(0)),
          minority_((size_t)order * minority_count_, uint16_t(0)) {}

    static State random_start(int order, int weight, uint32_t seed) {
        State state(order, weight);
        std::mt19937 rng(seed);

        for (int row = 0; row < order; ++row) {
            for (int k = 0; k < weight; ++k) {
                const int col = (row + k) % order;
                state.entry(row, col) = (rng() & 1u) ? int8_t(1) : int8_t(-1);
            }
        }
        state.rebuild_minority();
        state.randomize_support(rng);
        state.rebuild_gram_and_residual();
        return state;
    }

    static State sylvester_blocks(int order, int weight) {
        State state(order, weight);
        for (int block = 0; block < order / weight; ++block) {
            const int base = block * weight;
            for (int row = 0; row < weight; ++row) {
                for (int col = 0; col < weight; ++col) {
                    const int parity = std::popcount(
                        static_cast<unsigned int>(row & col)) & 1;
                    state.entry(base + row, base + col) =
                        parity == 0 ? int8_t(1) : int8_t(-1);
                }
            }
        }
        state.rebuild_minority();
        for (int i = 0; i < order; ++i) state.gram_at(i, i) = (int16_t)weight;
        return state;
    }

    static State from_entries(int order, int weight,
                              std::vector<int8_t> entries) {
        if (entries.size() != (size_t)order * order)
            throw std::runtime_error("invalid xweigh entry buffer size");
        State state(order, weight);
        state.entries_ = std::move(entries);
        state.rebuild_minority();
        state.rebuild_gram_and_residual();
        return state;
    }

    static void apply_to_entries(std::vector<int8_t>& entries, int order,
                                 const Move& move) {
        auto at = [&](int row, int col) -> int8_t& {
            return entries[(size_t)col * order + row];
        };
        if (move.kind == MoveKind::sign_flip) {
            at(move.row1, move.col1) = (int8_t)-at(move.row1, move.col1);
            return;
        }
        const int8_t value1 = at(move.row1, move.col1);
        const int8_t value2 = at(move.row2, move.col2);
        at(move.row1, move.col1) = 0;
        at(move.row2, move.col2) = 0;
        at(move.row1, move.col2) = value1;
        at(move.row2, move.col1) = value2;
    }

    int order() const { return n_; }
    int weight() const { return w_; }
    int64_t score() const { return score_; }
    int64_t conflicts() const { return conflicts_; }
    bool solved() const { return score_ == 0; }

    size_t estimated_bytes() const {
        return entries_.capacity() * sizeof(entries_[0]) +
               gram_.capacity() * sizeof(gram_[0]) +
               row_residual_.capacity() * sizeof(row_residual_[0]) +
               minority_.capacity() * sizeof(minority_[0]);
    }

    bool propose_move(std::mt19937& rng, const Params& params,
                      bool targeted, Move& move) const {
        const int row = targeted ? hot_row(rng, params.target_samples)
                                 : random_index(rng, n_);
        if (w_ == n_ || random_unit(rng) < params.sign_fraction) {
            return propose_sign_flip(rng, row, move);
        }
        if (propose_support_switch(rng, row, move)) return true;
        return propose_sign_flip(rng, row, move);
    }

    Evaluation evaluate(const Move& move) const {
        return move.kind == MoveKind::sign_flip
            ? evaluate_sign_flip(move)
            : evaluate_support_switch(move);
    }

    void commit(const Move& move, const Evaluation& evaluation) {
        const Evaluation actual = move.kind == MoveKind::sign_flip
            ? commit_sign_flip(move)
            : commit_support_switch(move);
        if (actual.score_delta != evaluation.score_delta ||
            actual.conflict_delta != evaluation.conflict_delta) {
            throw std::runtime_error("xweigh incremental move evaluation mismatch");
        }
        score_ += actual.score_delta;
        conflicts_ += actual.conflict_delta;
    }

    bool verify_support() const {
        std::vector<int> col_weights((size_t)n_, 0);
        for (int row = 0; row < n_; ++row) {
            int row_weight = 0;
            for (int col = 0; col < n_; ++col) {
                const int value = (int)entry(row, col);
                if (value < -1 || value > 1) return false;
                if (value != 0) {
                    ++row_weight;
                    ++col_weights[(size_t)col];
                }
            }
            if (row_weight != w_) return false;
        }
        return std::all_of(col_weights.begin(), col_weights.end(),
                           [&](int value) { return value == w_; });
    }

    bool verify_weighing() const {
        if (!verify_support()) return false;

        const size_t words = ((size_t)n_ + 63) / 64;
        std::vector<uint64_t> positive((size_t)n_ * words, 0);
        std::vector<uint64_t> negative((size_t)n_ * words, 0);
        build_sign_bits(positive, negative, words);

        for (int row1 = 0; row1 < n_; ++row1) {
            const uint64_t* p1 = positive.data() + (size_t)row1 * words;
            const uint64_t* m1 = negative.data() + (size_t)row1 * words;
            for (int row2 = row1 + 1; row2 < n_; ++row2) {
                const uint64_t* p2 = positive.data() + (size_t)row2 * words;
                const uint64_t* m2 = negative.data() + (size_t)row2 * words;
                int dot = 0;
                for (size_t word = 0; word < words; ++word) {
                    dot += (int)std::popcount(p1[word] & p2[word]);
                    dot += (int)std::popcount(m1[word] & m2[word]);
                    dot -= (int)std::popcount(p1[word] & m2[word]);
                    dot -= (int)std::popcount(m1[word] & p2[word]);
                }
                if (dot != 0) return false;
            }
        }
        return true;
    }

    void print() const {
        for (int row = 0; row < n_; ++row) {
            for (int col = 0; col < n_; ++col) {
                std::cout << (int)entry(row, col);
                if (col + 1 < n_) std::cout << ',';
            }
            std::cout << '\n';
        }
    }

    const std::vector<int8_t>& entries() const { return entries_; }

private:
    int n_ = 0;
    int w_ = 0;
    bool sparse_side_ = true;
    int minority_count_ = 0;
    std::vector<int8_t> entries_;
    std::vector<int16_t> gram_;
    std::vector<int64_t> row_residual_;
    std::vector<uint16_t> minority_;
    int64_t score_ = 0;
    int64_t conflicts_ = 0;

    int8_t& entry(int row, int col) {
        return entries_[(size_t)col * n_ + row];
    }

    int8_t entry(int row, int col) const {
        return entries_[(size_t)col * n_ + row];
    }

    int16_t& gram_at(int row, int col) {
        return gram_[(size_t)row * n_ + col];
    }

    int16_t gram_at(int row, int col) const {
        return gram_[(size_t)row * n_ + col];
    }

    uint16_t& minority_at(int row, int pos) {
        return minority_[(size_t)row * minority_count_ + pos];
    }

    uint16_t minority_at(int row, int pos) const {
        return minority_[(size_t)row * minority_count_ + pos];
    }

    void rebuild_minority() {
        if (minority_count_ == 0) return;
        for (int row = 0; row < n_; ++row) {
            int pos = 0;
            for (int col = 0; col < n_; ++col) {
                const bool nonzero = entry(row, col) != 0;
                if (nonzero == sparse_side_) {
                    if (pos >= minority_count_)
                        throw std::runtime_error("invalid regular support");
                    minority_at(row, pos++) = (uint16_t)col;
                }
            }
            if (pos != minority_count_)
                throw std::runtime_error("invalid regular support");
        }
    }

    bool raw_support_switch(std::mt19937& rng) {
        if (minority_count_ == 0 || n_ < 2) return false;
        const int row1 = random_index(rng, n_);
        int row2 = random_index(rng, n_ - 1);
        if (row2 >= row1) ++row2;
        const int pos1 = random_index(rng, minority_count_);
        const int pos2 = random_index(rng, minority_count_);

        int col1 = 0;
        int col2 = 0;
        if (sparse_side_) {
            col1 = minority_at(row1, pos1);
            col2 = minority_at(row2, pos2);
        } else {
            col2 = minority_at(row1, pos1);
            col1 = minority_at(row2, pos2);
        }
        if (col1 == col2 || entry(row1, col2) != 0 ||
            entry(row2, col1) != 0) {
            return false;
        }

        const int8_t value1 = entry(row1, col1);
        const int8_t value2 = entry(row2, col2);
        if (value1 == 0 || value2 == 0) return false;

        entry(row1, col1) = 0;
        entry(row2, col2) = 0;
        entry(row1, col2) = value1;
        entry(row2, col1) = value2;
        if (sparse_side_) {
            minority_at(row1, pos1) = (uint16_t)col2;
            minority_at(row2, pos2) = (uint16_t)col1;
        } else {
            minority_at(row1, pos1) = (uint16_t)col1;
            minority_at(row2, pos2) = (uint16_t)col2;
        }
        return true;
    }

    void randomize_support(std::mt19937& rng) {
        if (minority_count_ == 0) return;
        const int per_row = std::min(minority_count_, 64);
        const uint64_t target = (uint64_t)4 * (uint64_t)n_ * (uint64_t)per_row;
        uint64_t accepted = 0;
        uint64_t attempts = 0;
        while (accepted < target && attempts < target * 12) {
            ++attempts;
            if (raw_support_switch(rng)) ++accepted;
        }
    }

    void build_sign_bits(std::vector<uint64_t>& positive,
                         std::vector<uint64_t>& negative,
                         size_t words) const {
        for (int col = 0; col < n_; ++col) {
            const size_t word = (size_t)col / 64;
            const uint64_t bit = uint64_t(1) << (col % 64);
            const int8_t* column = entries_.data() + (size_t)col * n_;
            for (int row = 0; row < n_; ++row) {
                if (column[row] > 0)
                    positive[(size_t)row * words + word] |= bit;
                else if (column[row] < 0)
                    negative[(size_t)row * words + word] |= bit;
            }
        }
    }

    void rebuild_gram_sparse() {
        std::vector<int> rows;
        std::vector<int8_t> signs;
        rows.reserve((size_t)w_);
        signs.reserve((size_t)w_);

        for (int col = 0; col < n_; ++col) {
            rows.clear();
            signs.clear();
            const int8_t* column = entries_.data() + (size_t)col * n_;
            for (int row = 0; row < n_; ++row) {
                if (column[row] != 0) {
                    rows.push_back(row);
                    signs.push_back(column[row]);
                }
            }
            for (int i = 0; i < (int)rows.size(); ++i) {
                for (int j = i + 1; j < (int)rows.size(); ++j) {
                    const int row1 = rows[(size_t)i];
                    const int row2 = rows[(size_t)j];
                    const int value = gram_at(row1, row2) +
                                      signs[(size_t)i] * signs[(size_t)j];
                    gram_at(row1, row2) = (int16_t)value;
                    gram_at(row2, row1) = (int16_t)value;
                }
            }
        }
        for (int row = 0; row < n_; ++row) gram_at(row, row) = (int16_t)w_;
    }

    void rebuild_gram_bits() {
        const size_t words = ((size_t)n_ + 63) / 64;
        std::vector<uint64_t> positive((size_t)n_ * words, 0);
        std::vector<uint64_t> negative((size_t)n_ * words, 0);
        build_sign_bits(positive, negative, words);

        for (int row1 = 0; row1 < n_; ++row1) {
            const uint64_t* p1 = positive.data() + (size_t)row1 * words;
            const uint64_t* m1 = negative.data() + (size_t)row1 * words;
            gram_at(row1, row1) = (int16_t)w_;
            for (int row2 = row1 + 1; row2 < n_; ++row2) {
                const uint64_t* p2 = positive.data() + (size_t)row2 * words;
                const uint64_t* m2 = negative.data() + (size_t)row2 * words;
                int dot = 0;
                for (size_t word = 0; word < words; ++word) {
                    dot += (int)std::popcount(p1[word] & p2[word]);
                    dot += (int)std::popcount(m1[word] & m2[word]);
                    dot -= (int)std::popcount(p1[word] & m2[word]);
                    dot -= (int)std::popcount(m1[word] & p2[word]);
                }
                gram_at(row1, row2) = (int16_t)dot;
                gram_at(row2, row1) = (int16_t)dot;
            }
        }
    }

    void rebuild_gram_and_residual() {
        std::fill(gram_.begin(), gram_.end(), int16_t(0));
        if (w_ <= std::max(1, n_ / 8))
            rebuild_gram_sparse();
        else
            rebuild_gram_bits();

        std::fill(row_residual_.begin(), row_residual_.end(), int64_t(0));
        score_ = 0;
        conflicts_ = 0;
        for (int row1 = 0; row1 < n_; ++row1) {
            for (int row2 = row1 + 1; row2 < n_; ++row2) {
                const int value = gram_at(row1, row2);
                const int64_t residual = std::abs(value);
                score_ += residual;
                row_residual_[(size_t)row1] += residual;
                row_residual_[(size_t)row2] += residual;
                if (value != 0) ++conflicts_;
            }
        }
    }

    int hot_row(std::mt19937& rng, int samples) const {
        int best = random_index(rng, n_);
        for (int sample = 1; sample < samples; ++sample) {
            const int candidate = random_index(rng, n_);
            if (row_residual_[(size_t)candidate] >
                row_residual_[(size_t)best]) {
                best = candidate;
            }
        }
        return best;
    }

    bool propose_sign_flip(std::mt19937& rng, int row, Move& move) const {
        int col = 0;
        if (sparse_side_ && minority_count_ > 0) {
            col = minority_at(row, random_index(rng, minority_count_));
        } else {
            const int start = random_index(rng, n_);
            col = start;
            while (entry(row, col) == 0) {
                if (++col == n_) col = 0;
                if (col == start) return false;
            }
        }
        move.kind = MoveKind::sign_flip;
        move.row1 = row;
        move.col1 = col;
        return true;
    }

    bool propose_support_switch(std::mt19937& rng, int row1, Move& move) const {
        if (minority_count_ == 0 || n_ < 2) return false;
        for (int attempt = 0; attempt < 24; ++attempt) {
            int row2 = random_index(rng, n_ - 1);
            if (row2 >= row1) ++row2;
            const int pos1 = random_index(rng, minority_count_);
            const int pos2 = random_index(rng, minority_count_);
            int col1 = 0;
            int col2 = 0;
            if (sparse_side_) {
                col1 = minority_at(row1, pos1);
                col2 = minority_at(row2, pos2);
            } else {
                col2 = minority_at(row1, pos1);
                col1 = minority_at(row2, pos2);
            }
            if (col1 == col2 || entry(row1, col2) != 0 ||
                entry(row2, col1) != 0) {
                continue;
            }
            if (entry(row1, col1) == 0 || entry(row2, col2) == 0)
                continue;

            move.kind = MoveKind::support_switch;
            move.row1 = row1;
            move.row2 = row2;
            move.col1 = col1;
            move.col2 = col2;
            move.minority_pos1 = pos1;
            move.minority_pos2 = pos2;
            return true;
        }
        return false;
    }

    static int64_t residual_delta(int old_value, int new_value) {
        return (int64_t)std::abs(new_value) - std::abs(old_value);
    }

    static int64_t conflict_delta(int old_value, int new_value) {
        if (old_value == 0 && new_value != 0) return 1;
        if (old_value != 0 && new_value == 0) return -1;
        return 0;
    }

    Evaluation evaluate_sign_flip(const Move& move) const {
        Evaluation result;
        const int row = move.row1;
        const int col = move.col1;
        const int change = -2 * (int)entry(row, col);
        const int8_t* column = entries_.data() + (size_t)col * n_;
        const int16_t* gram_row = gram_.data() + (size_t)row * n_;
        for (int other = 0; other < n_; ++other) {
            if (other == row) continue;
            const int old_value = gram_row[other];
            const int new_value = old_value + change * (int)column[other];
            result.score_delta += residual_delta(old_value, new_value);
            result.conflict_delta += conflict_delta(old_value, new_value);
        }
        return result;
    }

    Evaluation evaluate_support_switch(const Move& move) const {
        Evaluation result;
        const int row1 = move.row1;
        const int row2 = move.row2;
        const int col1 = move.col1;
        const int col2 = move.col2;
        const int value1 = entry(row1, col1);
        const int value2 = entry(row2, col2);
        const int8_t* column1 = entries_.data() + (size_t)col1 * n_;
        const int8_t* column2 = entries_.data() + (size_t)col2 * n_;
        const int16_t* gram_row1 = gram_.data() + (size_t)row1 * n_;
        const int16_t* gram_row2 = gram_.data() + (size_t)row2 * n_;

        for (int other = 0; other < n_; ++other) {
            if (other == row1 || other == row2) continue;
            const int old1 = gram_row1[other];
            const int old2 = gram_row2[other];
            const int new1 = old1 + value1 *
                ((int)column2[other] - (int)column1[other]);
            const int new2 = old2 + value2 *
                ((int)column1[other] - (int)column2[other]);
            result.score_delta += residual_delta(old1, new1);
            result.score_delta += residual_delta(old2, new2);
            result.conflict_delta += conflict_delta(old1, new1);
            result.conflict_delta += conflict_delta(old2, new2);
        }
        return result;
    }

    Evaluation commit_sign_flip(const Move& move) {
        Evaluation result;
        const int row = move.row1;
        const int col = move.col1;
        const int change = -2 * (int)entry(row, col);
        const int8_t* column = entries_.data() + (size_t)col * n_;
        for (int other = 0; other < n_; ++other) {
            if (other == row) continue;
            const int old_value = gram_at(row, other);
            const int new_value = old_value + change * (int)column[other];
            const int64_t delta = residual_delta(old_value, new_value);
            gram_at(row, other) = (int16_t)new_value;
            gram_at(other, row) = (int16_t)new_value;
            row_residual_[(size_t)row] += delta;
            row_residual_[(size_t)other] += delta;
            result.score_delta += delta;
            result.conflict_delta += conflict_delta(old_value, new_value);
        }
        entry(row, col) = (int8_t)-entry(row, col);
        return result;
    }

    Evaluation commit_support_switch(const Move& move) {
        Evaluation result;
        const int row1 = move.row1;
        const int row2 = move.row2;
        const int col1 = move.col1;
        const int col2 = move.col2;
        const int8_t value1 = entry(row1, col1);
        const int8_t value2 = entry(row2, col2);
        const int8_t* column1 = entries_.data() + (size_t)col1 * n_;
        const int8_t* column2 = entries_.data() + (size_t)col2 * n_;

        for (int other = 0; other < n_; ++other) {
            if (other == row1 || other == row2) continue;
            const int old1 = gram_at(row1, other);
            const int old2 = gram_at(row2, other);
            const int new1 = old1 + (int)value1 *
                ((int)column2[other] - (int)column1[other]);
            const int new2 = old2 + (int)value2 *
                ((int)column1[other] - (int)column2[other]);
            const int64_t delta1 = residual_delta(old1, new1);
            const int64_t delta2 = residual_delta(old2, new2);

            gram_at(row1, other) = (int16_t)new1;
            gram_at(other, row1) = (int16_t)new1;
            gram_at(row2, other) = (int16_t)new2;
            gram_at(other, row2) = (int16_t)new2;
            row_residual_[(size_t)row1] += delta1;
            row_residual_[(size_t)row2] += delta2;
            row_residual_[(size_t)other] += delta1 + delta2;
            result.score_delta += delta1 + delta2;
            result.conflict_delta += conflict_delta(old1, new1);
            result.conflict_delta += conflict_delta(old2, new2);
        }

        entry(row1, col1) = 0;
        entry(row2, col2) = 0;
        entry(row1, col2) = value1;
        entry(row2, col1) = value2;
        if (sparse_side_) {
            minority_at(row1, move.minority_pos1) = (uint16_t)col2;
            minority_at(row2, move.minority_pos2) = (uint16_t)col1;
        } else {
            minority_at(row1, move.minority_pos1) = (uint16_t)col1;
            minority_at(row2, move.minority_pos2) = (uint16_t)col2;
        }
        return result;
    }
};

inline State read_state_csv(const std::string& filename,
                            int order, int weight) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error(
            "failed to open start matrix: " + filename);

    std::vector<int8_t> entries(
        static_cast<size_t>(order) * order);
    std::vector<int> column_weights(static_cast<size_t>(order), 0);
    std::string line;
    int row = 0;
    while (std::getline(input, line)) {
        if (row >= order)
            throw std::runtime_error(
                "start matrix has more than " +
                std::to_string(order) + " rows");
        std::stringstream stream(line);
        std::string token;
        int row_weight = 0;
        for (int column = 0; column < order; ++column) {
            if (!std::getline(stream, token, ','))
                throw std::runtime_error(
                    "start matrix row " + std::to_string(row + 1) +
                    " has fewer than " + std::to_string(order) +
                    " entries");
            const size_t first =
                token.find_first_not_of(" \t\r\n");
            const size_t last =
                token.find_last_not_of(" \t\r\n");
            if (first == std::string::npos)
                throw std::runtime_error(
                    "start matrix contains an empty entry");
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
                    "start matrix contains a non-ternary entry");
            }
            entries[static_cast<size_t>(column) * order + row] =
                static_cast<int8_t>(value);
            if (value != 0) {
                ++row_weight;
                ++column_weights[static_cast<size_t>(column)];
            }
        }
        if (std::getline(stream, token, ','))
            throw std::runtime_error(
                "start matrix row " + std::to_string(row + 1) +
                " has more than " + std::to_string(order) +
                " entries");
        if (row_weight != weight)
            throw std::runtime_error(
                "start matrix row " + std::to_string(row + 1) +
                " has weight " + std::to_string(row_weight) +
                ", expected " + std::to_string(weight));
        ++row;
    }
    if (row != order)
        throw std::runtime_error(
            "start matrix has " + std::to_string(row) +
            " rows, expected " + std::to_string(order));
    for (int column = 0; column < order; ++column) {
        const int actual =
            column_weights[static_cast<size_t>(column)];
        if (actual != weight)
            throw std::runtime_error(
                "start matrix column " +
                std::to_string(column + 1) + " has weight " +
                std::to_string(actual) + ", expected " +
                std::to_string(weight));
    }

    State state =
        State::from_entries(order, weight, std::move(entries));
    if (!state.verify_support())
        throw std::runtime_error(
            "start matrix failed fixed-support verification");
    return state;
}

inline void write_entries_csv(const std::vector<int8_t>& entries, int order,
                              const std::string& filename) {
    const std::string temporary = filename + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary);
        if (!out) throw std::runtime_error("failed to open output file: " + temporary);
        std::string row;
        row.reserve((size_t)order * 3);
        for (int r = 0; r < order; ++r) {
            row.clear();
            for (int c = 0; c < order; ++c) {
                const int value = entries[(size_t)c * order + r];
                if (value < 0) {
                    row += "-1";
                } else {
                    row.push_back(value == 0 ? '0' : '1');
                }
                row.push_back(c + 1 < order ? ',' : '\n');
            }
            out.write(row.data(), (std::streamsize)row.size());
        }
        out.flush();
        if (!out) throw std::runtime_error("failed while writing: " + temporary);
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

inline void write_state_csv(const State& state, const std::string& filename) {
    write_entries_csv(state.entries(), state.order(), filename);
}

#if defined(_WIN32)
inline std::vector<DWORD_PTR> physical_core_masks() {
    std::vector<DWORD_PTR> masks;
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length == 0) return masks;
    std::vector<char> buffer((size_t)length);
    auto* first =
        reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &length))
        return masks;
    char* current = buffer.data();
    char* end = buffer.data() + length;
    while (current < end) {
        auto* info =
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(current);
        if (info->Relationship == RelationProcessorCore &&
            info->Processor.GroupCount >= 1 &&
            info->Processor.GroupMask[0].Group == 0) {
            masks.push_back((DWORD_PTR)info->Processor.GroupMask[0].Mask);
        }
        current += info->Size;
    }
    return masks;
}
#endif

inline int physical_core_count() {
#if defined(_WIN32)
    const auto masks = physical_core_masks();
    if (!masks.empty()) return (int)masks.size();
#endif
    return std::max(1, (int)std::thread::hardware_concurrency());
}

struct alignas(64) WorkerStatus {
    std::atomic<int64_t> score{ 0 };
    std::atomic<int> rung{ 0 };
    uint8_t padding[64 - sizeof(std::atomic<int64_t>) -
                    sizeof(std::atomic<int>)]{};
};

static_assert(sizeof(WorkerStatus) == 64);

struct WorkerBest {
    std::mutex mutex;
    std::vector<int8_t> entries;
    int64_t score = 0;
    int64_t conflicts = 0;
};

struct Shared {
    explicit Shared(const State& start, int workers)
        : statuses(std::make_unique<WorkerStatus[]>((size_t)workers)),
          worker_bests(std::make_unique<WorkerBest[]>((size_t)workers)) {
        best_score.store(start.score(), std::memory_order_relaxed);
        best_conflicts.store(start.conflicts(), std::memory_order_relaxed);
        for (int worker = 0; worker < workers; ++worker) {
            worker_bests[(size_t)worker].entries = start.entries();
            worker_bests[(size_t)worker].score = start.score();
            worker_bests[(size_t)worker].conflicts = start.conflicts();
        }
    }

    std::mutex best_mutex;
    std::mutex print_mutex;
    std::atomic<int64_t> best_score{ 0 };
    std::atomic<int64_t> best_conflicts{ 0 };
    int best_owner = 0;
    std::atomic<bool> done{ false };
    std::atomic<uint64_t> total_moves{ 0 };
    std::atomic<uint64_t> best_version{ 1 };
    std::unique_ptr<WorkerStatus[]> statuses;
    std::unique_ptr<WorkerBest[]> worker_bests;
    std::vector<double> temperatures;
    double last_log_elapsed = -1e30;
};

inline double calibrate_temperature(const State& state, const Params& params,
                                    uint32_t seed) {
    if (params.t_init > 0.0) return params.t_init;
    std::mt19937 rng(seed ^ 0xa511e9b3u);
    long double positive_sum = 0.0L;
    int positive_count = 0;
    for (int sample = 0; sample < 256; ++sample) {
        Move move;
        if (!state.propose_move(rng, params, false, move)) continue;
        const Evaluation evaluation = state.evaluate(move);
        if (evaluation.score_delta > 0) {
            positive_sum += (long double)evaluation.score_delta;
            ++positive_count;
        }
    }
    if (positive_count == 0) return 1.0;
    const double mean = (double)(positive_sum / positive_count);
    return std::max(1.0, mean / std::log(2.0));
}

inline bool should_publish(const State& state, const Shared& shared) {
    const int64_t best_score = shared.best_score.load(std::memory_order_relaxed);
    const int64_t best_conflicts =
        shared.best_conflicts.load(std::memory_order_relaxed);
    return better_pair(state.score(), state.conflicts(),
                       best_score, best_conflicts);
}

inline void anneal_worker(const Params& params, uint32_t seed, int worker_id,
                          double initial_temperature, const State& start,
                          Shared& shared,
                          std::chrono::steady_clock::time_point start_time) {
    State current = start;
    std::mt19937 rng(seed);
    int64_t local_best_score = current.score();
    int64_t local_best_conflicts = current.conflicts();
    std::vector<Move> accepted_path;
    accepted_path.reserve((size_t)std::min(params.stuck_threshold, 250000));
    int moves_since_improvement = 0;
    int cooling_counter = 0;
    uint64_t local_moves = 0;
    uint64_t unreported_moves = 0;
    double temperature = initial_temperature;

    auto save_local_best = [&]() {
        WorkerBest& best = shared.worker_bests[(size_t)worker_id];
        std::lock_guard<std::mutex> lock(best.mutex);
        for (const Move& move : accepted_path)
            State::apply_to_entries(best.entries, current.order(), move);
        best.score = current.score();
        best.conflicts = current.conflicts();
        accepted_path.clear();
    };

    auto restore_global_best = [&]() {
        int owner = 0;
        {
            std::lock_guard<std::mutex> lock(shared.best_mutex);
            owner = shared.best_owner;
        }
        std::vector<int8_t> entries;
        int64_t score = 0;
        int64_t conflicts = 0;
        {
            WorkerBest& best = shared.worker_bests[(size_t)owner];
            std::lock_guard<std::mutex> lock(best.mutex);
            entries = best.entries;
            score = best.score;
            conflicts = best.conflicts;
        }
        current = State::from_entries(current.order(), current.weight(),
                                      std::move(entries));
        if (current.score() != score || current.conflicts() != conflicts)
            throw std::runtime_error("xweigh best snapshot score mismatch");
        local_best_score = score;
        local_best_conflicts = conflicts;
        accepted_path.clear();
        WorkerBest& local = shared.worker_bests[(size_t)worker_id];
        std::lock_guard<std::mutex> lock(local.mutex);
        local.entries = current.entries();
        local.score = score;
        local.conflicts = conflicts;
    };

    while (!shared.done.load(std::memory_order_relaxed)) {
        if (params.tempering) {
            const int rung =
                shared.statuses[(size_t)worker_id].rung.load(std::memory_order_relaxed);
            temperature = shared.temperatures[(size_t)rung];
        }

        Move selected;
        Evaluation selected_evaluation;
        bool have_move = false;
        const bool greedy = random_unit(rng) < params.greedy_fraction;
        const int samples = greedy ? params.candidate_samples : 1;
        for (int sample = 0; sample < samples; ++sample) {
            Move candidate;
            const bool targeted = random_unit(rng) < params.target_fraction;
            if (!current.propose_move(rng, params, targeted, candidate)) continue;
            const Evaluation evaluation = current.evaluate(candidate);
            if (!have_move ||
                better_pair(evaluation.score_delta, evaluation.conflict_delta,
                            selected_evaluation.score_delta,
                            selected_evaluation.conflict_delta)) {
                selected = candidate;
                selected_evaluation = evaluation;
                have_move = true;
            }
        }
        if (!have_move) continue;

        ++local_moves;
        ++unreported_moves;
        const bool accept = selected_evaluation.score_delta <= 0 ||
            random_unit(rng) <
                std::exp(-(double)selected_evaluation.score_delta /
                         std::max(temperature, 1e-12));
        if (accept) {
            current.commit(selected, selected_evaluation);
            accepted_path.push_back(selected);
        }

        if (better_pair(current.score(), current.conflicts(),
                        local_best_score, local_best_conflicts)) {
            local_best_score = current.score();
            local_best_conflicts = current.conflicts();
            moves_since_improvement = 0;
            save_local_best();
        } else {
            ++moves_since_improvement;
        }

        if (accept && should_publish(current, shared)) {
            bool published = false;
            {
                std::lock_guard<std::mutex> lock(shared.best_mutex);
                const int64_t best_score =
                    shared.best_score.load(std::memory_order_relaxed);
                const int64_t best_conflicts =
                    shared.best_conflicts.load(std::memory_order_relaxed);
                if (better_pair(current.score(), current.conflicts(),
                                best_score, best_conflicts)) {
                    shared.best_score.store(current.score(),
                                            std::memory_order_relaxed);
                    shared.best_conflicts.store(current.conflicts(),
                                                std::memory_order_relaxed);
                    shared.best_owner = worker_id;
                    shared.best_version.fetch_add(1, std::memory_order_relaxed);
                    published = true;
                    if (current.solved())
                        shared.done.store(true, std::memory_order_relaxed);
                }
            }
            if (published) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start_time).count();
                std::lock_guard<std::mutex> lock(shared.print_mutex);
                if (current.solved() ||
                    elapsed - shared.last_log_elapsed >= 0.05) {
                    shared.last_log_elapsed = elapsed;
                    std::cout << "[t=" << elapsed << "s] new best score="
                              << current.score() << " conflicts="
                              << current.conflicts() << " (thread "
                              << worker_id << ")\n";
                }
            }
        }

        if (!params.tempering) {
            if (++cooling_counter >= params.moves_per_cool) {
                cooling_counter = 0;
                temperature = std::max(params.t_min,
                                       temperature * params.cooling);
            }
            if (moves_since_improvement >= params.stuck_threshold) {
                moves_since_improvement = 0;
                temperature = std::max(params.t_min,
                    initial_temperature * params.reheat);
                const int64_t global_score =
                    shared.best_score.load(std::memory_order_relaxed);
                if ((double)current.score() >
                    params.reseed_factor * (double)std::max<int64_t>(1, global_score)) {
                    restore_global_best();
                }
            }
        }

        if (accepted_path.size() >= 250000) {
            int owner = 0;
            {
                std::lock_guard<std::mutex> lock(shared.best_mutex);
                owner = shared.best_owner;
            }
            if (owner == worker_id) {
                std::vector<int8_t> entries;
                {
                    WorkerBest& best = shared.worker_bests[(size_t)worker_id];
                    std::lock_guard<std::mutex> lock(best.mutex);
                    entries = best.entries;
                }
                current = State::from_entries(current.order(), current.weight(),
                                              std::move(entries));
                accepted_path.clear();
                moves_since_improvement = 0;
            } else {
                restore_global_best();
            }
        }

        if ((local_moves & 255u) == 0) {
            shared.statuses[(size_t)worker_id].score.store(
                current.score(), std::memory_order_relaxed);
        }
        if (unreported_moves >= 4096) {
            shared.total_moves.fetch_add(unreported_moves,
                                         std::memory_order_relaxed);
            unreported_moves = 0;
        }
    }
    shared.total_moves.fetch_add(unreported_moves, std::memory_order_relaxed);
    shared.statuses[(size_t)worker_id].score.store(current.score(),
                                                   std::memory_order_relaxed);
}

inline State run_annealer(State start, Params params) {
    if (params.threads < 2) params.tempering = false;
    if (start.solved()) return start;

    const uint32_t base_seed = params.seed
        ? (uint32_t)params.seed
        : (uint32_t)std::random_device{}();
    const double initial_temperature =
        calibrate_temperature(start, params, base_seed);
    Shared shared(start, params.threads);

    if (params.tempering) {
        shared.temperatures.resize((size_t)params.threads);
        const double bottom = std::min(params.t_min, initial_temperature);
        const double ratio = std::pow(
            initial_temperature / std::max(bottom, 1e-12),
            1.0 / (double)(params.threads - 1));
        double temperature = bottom;
        for (int worker = 0; worker < params.threads; ++worker) {
            shared.temperatures[(size_t)worker] = temperature;
            shared.statuses[(size_t)worker].rung.store(worker,
                                                       std::memory_order_relaxed);
            shared.statuses[(size_t)worker].score.store(
                start.score(), std::memory_order_relaxed);
            temperature *= ratio;
        }
    }

    const auto start_time = std::chrono::steady_clock::now();
    const double worker_mib =
        (double)(start.estimated_bytes() + start.entries().size()) /
        (1024.0 * 1024.0);
    std::cout << "[t=0s] start score=" << start.score()
              << " conflicts=" << start.conflicts()
              << " n=" << start.order()
              << " w=" << start.weight()
              << " threads=" << params.threads
              << " worker_mib=" << worker_mib
              << " t_init=" << initial_temperature
              << (params.tempering ? " (parallel tempering)" : "") << "\n";

#if defined(_WIN32)
    const std::vector<DWORD_PTR> core_masks =
        params.pin_threads ? physical_core_masks() : std::vector<DWORD_PTR>{};
#endif

    std::mt19937 seed_rng(base_seed);
    std::vector<std::thread> workers;
    workers.reserve((size_t)params.threads);
    for (int worker = 0; worker < params.threads; ++worker) {
        const uint32_t seed =
            seed_rng() ^ ((uint32_t)worker * 0x9e3779b9u);
        workers.emplace_back([&, seed, worker]() {
#if defined(_WIN32)
            if (!core_masks.empty()) {
                SetThreadAffinityMask(
                    GetCurrentThread(),
                    core_masks[(size_t)worker % core_masks.size()]);
            }
#endif
            anneal_worker(params, seed, worker, initial_temperature,
                          start, shared, start_time);
        });
    }

    std::thread monitor([&]() {
        std::mt19937 exchange_rng(base_seed ^ 0xc0ffeeu);
        uint64_t last_exchange_moves = 0;
        uint64_t saved_version = 0;
        double last_save_time = -1e30;
        while (!shared.done.load(std::memory_order_relaxed)) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
            if (g_stop.load(std::memory_order_relaxed) ||
                (params.max_seconds > 0.0 && elapsed >= params.max_seconds)) {
                shared.done.store(true, std::memory_order_relaxed);
                break;
            }

            const uint64_t moves =
                shared.total_moves.load(std::memory_order_relaxed);
            const uint64_t exchange_stride =
                (uint64_t)params.exchange_interval * (uint64_t)params.threads;
            if (params.tempering &&
                moves - last_exchange_moves >= exchange_stride) {
                last_exchange_moves = moves;
                std::vector<int> owner((size_t)params.threads);
                for (int worker = 0; worker < params.threads; ++worker) {
                    const int rung = shared.statuses[(size_t)worker].rung.load(
                        std::memory_order_relaxed);
                    owner[(size_t)rung] = worker;
                }
                for (int rung = 0; rung + 1 < params.threads; ++rung) {
                    const int worker1 = owner[(size_t)rung];
                    const int worker2 = owner[(size_t)rung + 1];
                    const double t1 = shared.temperatures[(size_t)rung];
                    const double t2 = shared.temperatures[(size_t)rung + 1];
                    const double score1 = (double)
                        shared.statuses[(size_t)worker1].score.load(
                            std::memory_order_relaxed);
                    const double score2 = (double)
                        shared.statuses[(size_t)worker2].score.load(
                            std::memory_order_relaxed);
                    const double exponent =
                        (1.0 / t1 - 1.0 / t2) * (score1 - score2);
                    if (exponent >= 0.0 ||
                        random_unit(exchange_rng) < std::exp(exponent)) {
                        shared.statuses[(size_t)worker1].rung.store(
                            rung + 1, std::memory_order_relaxed);
                        shared.statuses[(size_t)worker2].rung.store(
                            rung, std::memory_order_relaxed);
                        std::swap(owner[(size_t)rung],
                                  owner[(size_t)rung + 1]);
                    }
                }
            }

            const uint64_t version =
                shared.best_version.load(std::memory_order_relaxed);
            if (version != saved_version &&
                elapsed - last_save_time >= params.save_interval) {
                std::vector<int8_t> snapshot;
                const int order = start.order();
                uint64_t snapshot_version = 0;
                int owner = 0;
                {
                    std::lock_guard<std::mutex> lock(shared.best_mutex);
                    owner = shared.best_owner;
                    snapshot_version =
                        shared.best_version.load(std::memory_order_relaxed);
                }
                {
                    WorkerBest& best = shared.worker_bests[(size_t)owner];
                    std::lock_guard<std::mutex> lock(best.mutex);
                    snapshot = best.entries;
                }
                try {
                    write_entries_csv(snapshot, order, params.output);
                    saved_version = snapshot_version;
                    last_save_time = elapsed;
                } catch (const std::exception& error) {
                    std::lock_guard<std::mutex> lock(shared.print_mutex);
                    std::cerr << "Warning: " << error.what() << "\n";
                    last_save_time = elapsed;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    for (std::thread& worker : workers) worker.join();
    monitor.join();

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    const uint64_t moves =
        shared.total_moves.load(std::memory_order_relaxed);
    std::cout << "done moves=" << moves << " seconds=" << elapsed
              << " moves_per_sec="
              << (elapsed > 0.0 ? (double)moves / elapsed : 0.0) << "\n";

    int best_owner = 0;
    int64_t best_score = std::numeric_limits<int64_t>::max();
    int64_t best_conflicts = std::numeric_limits<int64_t>::max();
    for (int worker = 0; worker < params.threads; ++worker) {
        WorkerBest& candidate = shared.worker_bests[(size_t)worker];
        std::lock_guard<std::mutex> lock(candidate.mutex);
        if (better_pair(candidate.score, candidate.conflicts,
                        best_score, best_conflicts)) {
            best_owner = worker;
            best_score = candidate.score;
            best_conflicts = candidate.conflicts;
        }
    }
    std::vector<int8_t> best_entries;
    {
        WorkerBest& best = shared.worker_bests[(size_t)best_owner];
        std::lock_guard<std::mutex> lock(best.mutex);
        best_entries = std::move(best.entries);
    }
    State best = State::from_entries(start.order(), start.weight(),
                                     std::move(best_entries));
    if (best.score() != best_score || best.conflicts() != best_conflicts)
        throw std::runtime_error("xweigh final best snapshot mismatch");
    return best;
}

}  // namespace xweigh
