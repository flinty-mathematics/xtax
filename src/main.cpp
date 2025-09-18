#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>

#include <immintrin.h>

#include "CLI11.hpp"

constexpr int64_t MAGNITUDE_LIMIT = 1ll << 48;

struct Matrix {
    size_t n = 0;
    std::vector<int64_t> data;

    Matrix() = default;
    Matrix(size_t n_) : n(n_), data(n_* n_, 0) {}
    int64_t* buf() { return data.data(); }
    void fill_identity() {
        std::fill(data.begin(), data.end(), 0);
        for (int i = 0; ((size_t)i) < n; ++i) data[(size_t)i * n + i] = 1;
    }
    inline int64_t& at(int i, int j) { return data[(size_t)i * n + j]; }
    inline const int64_t& at(int i, int j) const { return data[(size_t)i * n + j]; }

    void print() const
    {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::cout << at(i, j);
                if (j + 1 < n) std::cout << ',';
            }
            std::cout << '\n';
        }
    }

    bool is_diagonal() const {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j && at(i, j) != 0) return false;
            }
        }
        return true;
    }

    double score() const {
        double full_sum = 0;
        for (const auto e : data) {
            full_sum += std::llabs(e);
        }

        double diag_sum = 0;
        for (size_t i = 0; i < n; ++i) {
            diag_sum += std::llabs(data[i*n+i]);
        }

        return 2 * full_sum - diag_sum;
    }

    bool is_within_magnitude_limit(int64_t limit) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const int64_t e = at(i, j);
                if (e > limit || e < -limit) {
                    return false;
                }
            }
        }
        return true;
    }
};


enum class TType { Add, Swap, Neg };
struct Trans { TType type; int i, j; int s; };

void apply_to_X(Matrix& X, const Trans& t) {
    const size_t n = X.n;

    if (t.type == TType::Add) {
        for (size_t r = 0; r < n; ++r)
            X.at(r, t.j) += (int64_t)t.s * X.at(r, t.i);
    }
    else if (t.type == TType::Swap) {
        for (size_t r = 0; r < n; ++r)
            std::swap(X.at(r, t.i), X.at(r, t.j));
    }
    else {
        for (size_t r = 0; r < n; ++r)
            X.at(r, t.i) = -X.at(r, t.i);
    }
}


void apply_to_A(Matrix& mtx, const Trans& t) {
    const size_t n = mtx.n;

    switch (t.type) {
        case TType::Neg: {
            int i = t.i;
            for (size_t k = 0; k < n; ++k) {
                mtx.at(i, k) = -mtx.at(i, k);
                mtx.at(k, i) = -mtx.at(k, i);
            }
            break;
        }
        case TType::Swap: {
            int a = t.i, b = t.j;
            for (size_t k = 0; k < n; ++k) {
                if (k != (size_t)a && k != (size_t)b) {
                    std::swap(mtx.at(a, k), mtx.at(b, k));
                    std::swap(mtx.at(k, a), mtx.at(k, b));
                }
            }
            std::swap(mtx.at(a, a), mtx.at(b, b));
            std::swap(mtx.at(a, b), mtx.at(b, a));
            break;
        }
        case TType::Add: {
            int i = t.i, j = t.j, s = t.s;
            for (size_t k = 0; k < n; ++k) {
                if (k != (size_t)i && k != (size_t)j) {
                    int64_t v = mtx.at(j, k) + (int64_t)s * mtx.at(i, k);
                    mtx.at(j, k) = v;
                    mtx.at(k, j) = v;
                }
            }
            int64_t a_ji = mtx.at(j, i) + (int64_t)s * mtx.at(i, i);
            mtx.at(j, i) = a_ji;
            mtx.at(i, j) = a_ji;

            // new_jj = old_jj + 2*s*old_ji + s*s*old_ii
            // expressed with a_ji (new_ji): new_jj = old_jj + 2*s*new_ji - s*s*old_ii
            mtx.at(j, j) += (int64_t)2 * s * mtx.at(j, i) - (int64_t)s * s * mtx.at(i, i);
            break;
        }
    }
}


// helper: apply transform to cur.A and cur.X in-place
inline void apply_trans_inplace(Matrix& A, Matrix& X, const Trans& t) {
    // apply to X
    apply_to_X(X, t);
    // apply to A & return new score
    apply_to_A(A, t);
}

// helper: produce inverse transform
inline Trans inverse_trans(const Trans& t) {
    if (t.type == TType::Add) return Trans{ TType::Add, t.i, t.j, -t.s };
    if (t.type == TType::Swap) return t; // self-inverse
    return t; // Neg is self-inverse
}

Trans random_transformation(int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> idx(0, n - 1);
    std::uniform_int_distribution<int> choice(0, 2);
    std::uniform_int_distribution<int> sign01(0, 1);
    Trans t;
    t.type = static_cast<TType>(choice(rng));
    t.i = idx(rng);
    if (t.type != TType::Neg) {
        do {
            t.j = idx(rng);
        } while (t.j == t.i);
        t.s = (sign01(rng) ? 1 : -1);
    }
    else {
        t.j = -1;
        t.s = 0;
    }
    return t;
}

struct Congruence { Matrix A; Matrix X; double score; };

void write_matrix_csv(const Matrix& M, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    int n = static_cast<int>(M.n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << M.at(i, j);
            if (j + 1 < n) out << ",";
        }
        out << "\n";
    }
}

Matrix read_matrix_csv(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) { std::cerr << "Cannot open " << filename << "\n"; return 1; }

    std::vector<int64_t> raw;
    std::string line;
    size_t rows = 0, cols = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        ++rows;
        std::stringstream ss(line);
        std::string token;
        size_t inner = 0;
        while (std::getline(ss, token, ',')) {
            ++inner;
            raw.push_back(static_cast<int64_t>(std::stoll(token)));
        }
        if (cols == 0) cols = inner;
        if (inner != cols) { std::cerr << "Bad CSV row length\n"; return 1; }
    }
    if (rows != cols) { std::cerr << "Matrix must be square\n"; return 1; }
    int n = static_cast<int>(rows);

    Matrix A(n);
    A.data = std::move(raw);

    return A;
}

struct Params {
    int stuck_threshold = 20000;        // moves without improvement before warming
    double warm_target_fraction = 0.05; // warm until score increases at least 5%
    int max_warm_moves = 1000;         // cap warm burst
};

Congruence anneal_worker(const Matrix& initialA,
    Params params,
    std::mt19937& rng_local,
    Congruence& global_best,
    std::atomic<double>& best_score_atomic,
    std::mutex& best_mtx,
    std::mutex& print_mtx,
    std::atomic<bool>& restart_flag,
    std::condition_variable& cv,
    std::atomic<bool>& done_flag,
    int thread_id)
{
    int n = initialA.n;
    Congruence cur;
    {
        std::lock_guard<std::mutex> lk(best_mtx);
        cur = global_best; // start from best
    }

    int moves_since_improvement = 0;
    double best_cooling_score = cur.score;

    while (true) {
        if (done_flag.load(std::memory_order_relaxed))
            return cur;

        // ----- ADAPTIVE WARM: accept repeated uphill moves until we've warmed enough -----
        if (moves_since_improvement > params.stuck_threshold) {
            int warm_count = 0;
            double start_score = static_cast<double>(cur.score);
            double target_score = start_score * (1.0 + params.warm_target_fraction);

            // Keep accepting *increasing* moves (subject to magnitude limit).
            // Stop when cur.score >= target_score or max_warm_moves reached.
            while (warm_count < params.max_warm_moves && cur.score < target_score) {
                Trans tw = random_transformation(n, rng_local);
                apply_trans_inplace(cur.A, cur.X, tw);
                double new_score_w = cur.A.score();

                if (!cur.X.is_within_magnitude_limit(MAGNITUDE_LIMIT)) {
                    // revert blow-up
                    Trans invw = inverse_trans(tw);
                    apply_trans_inplace(cur.A, cur.X, invw);
                }
                else if (new_score_w >= cur.score) {
                    // accept uphill or 'sideways' move, update score and best_cooling_score if appropriate
                    cur.score = new_score_w;
                    if (cur.score < best_cooling_score) best_cooling_score = cur.score;
                }
                else {
                    // reject downhill move during warming
                    Trans invw = inverse_trans(tw);
                    apply_trans_inplace(cur.A, cur.X, invw);
                }

                ++warm_count;
            }

            // finished warming burst
            moves_since_improvement = 0;
            continue;
        }
        // ----- end adaptive warm -----

        if (restart_flag.load(std::memory_order_relaxed)) {
            std::unique_lock<std::mutex> lk(best_mtx);
            cur = global_best;
            restart_flag.store(false, std::memory_order_relaxed);
            lk.unlock();
            continue;
        }

        // regular proposal
        Trans t = random_transformation(n, rng_local);
        apply_trans_inplace(cur.A, cur.X, t);
        double new_score = cur.A.score();
        if (new_score < cur.score && cur.X.is_within_magnitude_limit(MAGNITUDE_LIMIT)) {
            // accept improvement
            cur.score = new_score;
            if (new_score < best_cooling_score) best_cooling_score = new_score;
            moves_since_improvement = 0;
        }
        else {
            // reject -> revert
            Trans inv = inverse_trans(t);
            apply_trans_inplace(cur.A, cur.X, inv);
            moves_since_improvement++;
        }

        // check & publish global best
        if (cur.score < best_score_atomic.load()) {
            std::lock_guard<std::mutex> lk(best_mtx);
            if (cur.score < best_score_atomic.load()) {
                global_best = cur;
                best_score_atomic.store(cur.score);
                {
                    std::lock_guard<std::mutex> plk(print_mtx);
                    std::cout << "[Thread " << thread_id << "] New best score: " << cur.score << "\n";
                    write_matrix_csv(cur.X, "best_X.csv");
                    write_matrix_csv(cur.A, "best_A.csv");
                }
                if (global_best.A.is_diagonal()) {
                    {
                        std::lock_guard<std::mutex> plk(print_mtx);
                        std::cout << "[Thread " << thread_id << "] Matrix is diagonal. Stopping.\n";
                    }
                    done_flag.store(true, std::memory_order_relaxed);
                    cv.notify_all();
                    return cur;
                }
                restart_flag.store(true, std::memory_order_relaxed);
                cv.notify_all();
            }
        }
    }
}

int main(int argc, char** argv) {
    CLI::App app{ "xtax cpu congruence annealer" };

    std::string a_csv;
    std::string x_csv;
    int workers = std::max(1u, std::thread::hardware_concurrency());

    Params params;

    app.add_option("-A", a_csv, "CSV file for A (n x n integers)")->required();
    app.add_option("-X", x_csv, "Path to the X matrix. If none, start from identity");
    app.add_option("-w,--workers", workers, "Number of workers (blocks)");
    app.add_option("--warm-fraction", params.warm_target_fraction, "%age target to warm in case it gets stuck e.g 0.05 is 5%");
    app.add_option("--max-warm", params.max_warm_moves, "Cap on the number of warming moves");
    app.add_option("--stuck-threshold", params.stuck_threshold, "If stuck and we've exceeded this many cooling steps, will warm.");
    CLI11_PARSE(app, argc, argv);

    Matrix A = read_matrix_csv(a_csv);
    const int n = A.n;
    Matrix X(n);
    if (!x_csv.empty()) {
        X = read_matrix_csv(x_csv);
    } else {
        X.fill_identity();
    }

    std::mutex best_mtx;
    std::mutex print_mtx;
    std::condition_variable cv;
    std::atomic<bool> restart_flag{ false };
    std::atomic<bool> done_flag{ false };

    std::atomic<double> best_score_atomic{ A.score() };
    Congruence global_best{ A, Matrix(n), best_score_atomic.load() };
    global_best.X.fill_identity();

    std::vector<std::thread> threads;
    for (int t = 0; t < workers; ++t) {
        threads.emplace_back([&, t]() {
            std::random_device rd;
            std::mt19937 rng_local(static_cast<uint32_t>(rd()) ^ (uint32_t)(t * 0x9e3779b9u));
            anneal_worker(A, params,
                rng_local, global_best, best_score_atomic,
                best_mtx, print_mtx, restart_flag, cv, done_flag, t);
        });
    }
    for (auto& th : threads) th.join();

    std::cout << "Final best score: " << global_best.score << "\n";
    if (n <= 20) {
        std::cout << "A:\n";
        global_best.A.print();
        std::cout << "----\nX:\n";
        global_best.X.print();
    }

    write_matrix_csv(global_best.X, "best_X.csv");
    write_matrix_csv(global_best.A, "best_A.csv");

    return 0;
}
