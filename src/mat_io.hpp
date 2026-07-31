// mat_io.hpp: shared matrix / lattice types, CSV IO, and small integer helpers
// used across the congruence annealers (xtax, xdual) and the BKZ reducer (xbkz).
//
// The pieces here were previously copy-pasted across the tools. They are the
// genuinely identical parts: the dense integer Matrix and its scores, the
// double-precision Matrixd for a floating-point dual, the Lattice basis type,
// atomic CSV writes and readers, the transpose / Gram / X^T L helpers, the
// nearest-integer division and extended gcd, the int64 axpy overflow guard, and
// the mod-prime unimodularity test.
//
// Header-only: free functions are inline and there are no globals, so each
// executable's single translation unit gets exactly one definition. This header
// pulls in no profiling or signal machinery (see stop_signal.hpp for that), so
// it can be included anywhere cheaply.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// Hard bound on the magnitude of any matrix / transform entry the annealers
// keep. Moves that would push an entry beyond this are rejected. This keeps the
// search numerically sane and guarantees the int64 score arithmetic cannot
// overflow in practice: with entries bounded by 2^48, products s*entry and
// s*s*entry (|s| <= SHEAR_CAP) stay well inside the int64 range.
constexpr int64_t MAGNITUDE_LIMIT = 1ll << 48;

// Bound on |s| for an Add (shear) move. See MAGNITUDE_LIMIT.
constexpr int64_t SHEAR_CAP = 64;

// Round a/b to the nearest integer (ties away from zero). Requires b != 0.
inline int64_t rounded_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a - q * b;
    if (2 * std::llabs(r) >= std::llabs(b)) {
        q += ((a < 0) == (b < 0)) ? 1 : -1;
    }
    return q;
}

// Returns g = gcd(|a|, |b|) >= 0 with x*a + y*b = g.
inline int64_t ext_gcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
    int64_t old_r = a, r = b;
    int64_t old_s = 1, s = 0;
    int64_t old_t = 0, t = 1;
    while (r != 0) {
        int64_t q = old_r / r;
        int64_t tmp = old_r - q * r; old_r = r; r = tmp;
        tmp = old_s - q * s; old_s = s; s = tmp;
        tmp = old_t - q * t; old_t = t; t = tmp;
    }
    if (old_r < 0) { old_r = -old_r; old_s = -old_s; old_t = -old_t; }
    x = old_s; y = old_t;
    return old_r;
}

// Reduce every entry of a flat integer buffer modulo m, in place. This is a
// one-shot preprocessing pass applied to freshly loaded input (--modulus);
// nothing downstream carries a modulus, so the hot paths are unaffected.
//
// For m > 1 each entry is mapped to its balanced (least absolute) residue in
// (-m/2, m/2], which keeps magnitudes as small as possible. m == 1 is a
// special saturation mode: every nonzero entry becomes 1 and zero stays 0
// (a plain mod 1 would zero the whole input). The caller validates that m
// fits in int64.
inline void reduce_entries_mod(std::vector<int64_t>& data, uint64_t m) {
    if (m == 1) {
        for (int64_t& v : data) v = (v != 0) ? 1 : 0;
        return;
    }
    const int64_t mm = (int64_t)m;
    const int64_t half = mm / 2;
    for (int64_t& v : data) {
        int64_t r = v % mm;        // truncated: sign follows v, |r| < m
        if (r < 0) r += mm;        // now in [0, m)
        if (r > half) r -= mm;     // balanced: (-m/2, m/2]
        v = r;
    }
}

// Parse an index-list option value against the valid range [0, n): comma-
// separated 0-based indices and inclusive ranges written lo..hi (so
// "0,3..6,9" selects 0, 3, 4, 5, 6, 9). Empty tokens and surrounding
// whitespace are tolerated. opt is the option name used in error messages
// (e.g. "--gram-rows"). Returns the sorted, deduplicated index list. Throws
// std::runtime_error on a malformed token, a reversed range, or an index
// outside [0, n).
inline std::vector<int> parse_index_spec(const std::string& spec, int n,
                                         const std::string& opt) {
    auto parse_index = [n, &opt](const std::string& tok) -> int {
        size_t pos = 0;
        long long v = -1;
        try {
            v = std::stoll(tok, &pos);
        } catch (const std::exception&) {
            pos = 0;
        }
        if (pos != tok.size() || tok.empty() || v < 0)
            throw std::runtime_error(opt + ": invalid index '" + tok + "'");
        if (v >= (long long)n)
            throw std::runtime_error(opt + ": index " + tok +
                                     " out of range (valid indices are 0.." +
                                     std::to_string(n - 1) + ")");
        return (int)v;
    };
    auto trim = [](std::string s) {
        const auto b = s.find_first_not_of(" \t");
        if (b == std::string::npos) return std::string();
        const auto e = s.find_last_not_of(" \t");
        return s.substr(b, e - b + 1);
    };
    std::vector<int> idxs;
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (tok.empty()) continue;   // tolerate stray commas
        const size_t dots = tok.find("..");
        if (dots == std::string::npos) {
            idxs.push_back(parse_index(tok));
        } else {
            const int lo = parse_index(trim(tok.substr(0, dots)));
            const int hi = parse_index(trim(tok.substr(dots + 2)));
            if (lo > hi)
                throw std::runtime_error(opt + ": reversed range '" + tok + "'");
            for (int r = lo; r <= hi; ++r) idxs.push_back(r);
        }
    }
    std::sort(idxs.begin(), idxs.end());
    idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
    return idxs;
}

// One-line human description of what reduce_entries_mod(m) did, for the tools'
// startup logs.
inline std::string modulus_note(uint64_t m) {
    if (m == 1) return "saturated entries (nonzero -> 1, 0 stays 0)";
    return "reduced entries mod " + std::to_string(m) +
           " (balanced residues in (-m/2, m/2])";
}

// Clamp every entry of a flat integer buffer into [-n, n], in place. Like
// reduce_entries_mod this is a one-shot preprocessing pass on freshly loaded
// input (--saturate); nothing downstream carries the bound. The caller
// validates that n > 0. A nonzero entry stays nonzero (its magnitude is at
// least 1 <= n), so saturation never turns a nonzero row into a zero row.
inline void saturate_entries(std::vector<int64_t>& data, int64_t n) {
    for (int64_t& v : data) {
        if (v > n) v = n;
        else if (v < -n) v = -n;
    }
}

// One-line human description of what saturate_entries(n) did, for startup logs.
inline std::string saturate_note(int64_t n) {
    return "saturated entries to [" + std::to_string(-n) + ", " +
           std::to_string(n) + "]";
}

// dst += c * src with overflow detection. Returns true on overflow.
inline bool axpy_overflow(int64_t& dst, int64_t c, int64_t src) {
#if defined(__GNUC__) || defined(__clang__)
    int64_t p;
    if (__builtin_mul_overflow(c, src, &p)) return true;
    return __builtin_add_overflow(dst, p, &dst);
#else
    long double approx = (long double)dst + (long double)c * (long double)src;
    if (approx > 9.0e18L || approx < -9.0e18L) return true;
    dst += c * src;
    return false;
#endif
}

// A dense symmetric integer matrix, stored row-major. The annealers keep the
// working Gram here (exact int64) and maintain its scores incrementally.
struct Matrix {
    size_t n = 0;
    std::vector<int64_t> data;

    Matrix() = default;
    explicit Matrix(size_t n_) : n(n_), data(n_ * n_, 0) {}

    void fill_identity() {
        std::fill(data.begin(), data.end(), 0);
        for (size_t i = 0; i < n; ++i) data[i * n + i] = 1;
    }
    inline int64_t& at(int i, int j) { return data[(size_t)i * n + j]; }
    inline const int64_t& at(int i, int j) const { return data[(size_t)i * n + j]; }

    void print() const {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::cout << at((int)i, (int)j);
                if (j + 1 < n) std::cout << ',';
            }
            std::cout << '\n';
        }
    }

    bool is_diagonal() const {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j && at((int)i, (int)j) != 0) return false;
            }
        }
        return true;
    }

    // Number of nonzero off-diagonal pairs {i,j}, i < j. Zero iff diagonal.
    int64_t count_offdiag_nonzero() const {
        int64_t c = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                if (at((int)i, (int)j) != 0) ++c;
        return c;
    }

    // L1 sparsity score: 2 * sum|A_ij| - sum|A_ii|. Lower is more diagonal.
    // Computed once up front, then maintained incrementally by the worker.
    int64_t score() const {
        int64_t full_sum = 0;
        for (const auto e : data) full_sum += std::llabs(e);
        int64_t diag_sum = 0;
        for (size_t i = 0; i < n; ++i) diag_sum += std::llabs(data[i * n + i]);
        return 2 * full_sum - diag_sum;
    }

    // Squared Frobenius norm of the off-diagonal part: sum over i != j of M_ij^2.
    // Accumulated in long double. Entries are bounded by MAGNITUDE_LIMIT so this
    // is exact for modest entries and only loses low bits for very large ones.
    long double offdiag_frob2() const {
        long double s = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            const int64_t* row = data.data() + i * n;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                const long double v = (long double)row[j];
                s += v * v;
            }
        }
        return s;
    }
};

// A dense matrix of doubles, used for the floating-point dual Q = P^{-1}. The
// dual is only a search-guidance penalty, not part of the exact output, so
// double precision is enough. Keeping it floating-point also sidesteps the
// determinant entirely: for a large lattice det(G) has hundreds of digits, so an
// exact integer dual (det(G) P^{-1}) cannot fit in int64 and would need bignum.
struct Matrixd {
    size_t n = 0;
    std::vector<double> data;

    Matrixd() = default;
    explicit Matrixd(size_t n_) : n(n_), data(n_ * n_, 0.0) {}

    inline double& at(int i, int j) { return data[(size_t)i * n + j]; }
    inline const double& at(int i, int j) const { return data[(size_t)i * n + j]; }

    void print() const {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::cout << at((int)i, (int)j);
                if (j + 1 < n) std::cout << ',';
            }
            std::cout << '\n';
        }
    }

    // Squared Frobenius norm of the off-diagonal part: sum over i != j of M_ij^2.
    long double offdiag_frob2() const {
        long double s = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            const double* row = data.data() + i * n;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                const long double v = (long double)row[j];
                s += v * v;
            }
        }
        return s;
    }
};

// Invert the symmetric integer matrix P into the double matrix Q = P^{-1} by
// Gauss-Jordan elimination with partial pivoting. Returns false if P is singular
// (not full rank). The result is symmetrized to remove numerical asymmetry.
inline bool invert_to(const Matrix& P, Matrixd& Q) {
    const int n = (int)P.n;
    const int w = 2 * n;
    std::vector<double> a((size_t)n * w, 0.0);
    auto A = [&](int i, int j) -> double& { return a[(size_t)i * w + j]; };
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) A(i, j) = (double)P.at(i, j);
        A(i, n + i) = 1.0;
    }
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(A(col, col));
        for (int r = col + 1; r < n; ++r) {
            const double v = std::fabs(A(r, col));
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) return false;   // singular
        if (piv != col)
            for (int j = 0; j < w; ++j) std::swap(A(piv, j), A(col, j));
        const double inv = 1.0 / A(col, col);
        for (int j = 0; j < w; ++j) A(col, j) *= inv;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A(r, col);
            if (f == 0.0) continue;
            for (int j = 0; j < w; ++j) A(r, j) -= f * A(col, j);
        }
    }
    Q = Matrixd((size_t)n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Q.at(i, j) = A(i, n + j);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double m = 0.5 * (Q.at(i, j) + Q.at(j, i));
            Q.at(i, j) = m;
            Q.at(j, i) = m;
        }
    return true;
}

// Exact congruence X^T A X in int64. Used once at startup when the user
// supplies an initial transform, so the working matrix starts consistent with
// it. O(n^3), negligible outside the hot path.
inline Matrix congruence_of(const Matrix& A, const Matrix& X) {
    const int n = (int)A.n;
    Matrix AX((size_t)n);          // AX = A * X
    for (int i = 0; i < n; ++i) {
        const int64_t* ai = A.data.data() + (size_t)i * n;
        int64_t* ri = AX.data.data() + (size_t)i * n;
        for (int k = 0; k < n; ++k) {
            const int64_t a = ai[k];
            if (a == 0) continue;
            const int64_t* xk = X.data.data() + (size_t)k * n;
            for (int j = 0; j < n; ++j) ri[j] += a * xk[j];
        }
    }
    Matrix R((size_t)n);           // R = X^T * AX
    for (int i = 0; i < n; ++i) {
        int64_t* ri = R.data.data() + (size_t)i * n;
        for (int k = 0; k < n; ++k) {
            const int64_t x = X.at(k, i);   // (X^T)_{i,k}
            if (x == 0) continue;
            const int64_t* axk = AX.data.data() + (size_t)k * n;
            for (int j = 0; j < n; ++j) ri[j] += x * axk[j];
        }
    }
    return R;
}

inline Matrix transpose(const Matrix& M) {
    Matrix R(M.n);
    for (size_t i = 0; i < M.n; ++i)
        for (size_t j = 0; j < M.n; ++j)
            R.at((int)i, (int)j) = M.at((int)j, (int)i);
    return R;
}

// A lattice basis: m vectors (rows) living in dimension d (columns). For a
// genuine basis m <= d, and the Gram matrix A = L L^T is then m x m.
struct Lattice {
    int m = 0;                  // number of basis vectors (rows)
    int d = 0;                  // ambient dimension (columns)
    std::vector<int64_t> data;  // row-major, size m*d
    int64_t* row(int i) { return data.data() + (size_t)i * d; }
    const int64_t* row(int i) const { return data.data() + (size_t)i * d; }
};

// Replace a file atomically: write to a sibling temp file, flush it, then rename
// it over the target. A rename within a directory is atomic on Windows
// (MoveFileExW) and POSIX, so a reader or an interrupted run always sees either
// the previous complete file or the new complete one, never a half-written one.
// Throws on any I/O error. body writes the full contents into the stream.
template <typename Body>
inline void atomic_write(const std::string& filename, Body&& body) {
    const std::string tmp = filename + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) throw std::runtime_error("Failed to open output file: " + tmp);
        body(out);
        out.flush();
        if (!out) throw std::runtime_error("Failed while writing: " + tmp);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, filename, ec);
    if (ec) throw std::runtime_error("Failed to replace " + filename + ": " + ec.message());
}

inline void write_matrix_csv(const Matrix& M, const std::string& filename) {
    atomic_write(filename, [&](std::ofstream& out) {
        int n = static_cast<int>(M.n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                out << M.at(i, j);
                if (j + 1 < n) out << ",";
            }
            out << "\n";
        }
    });
}

// Write a double matrix at full round-trip precision. Used for the dual Q.
inline void write_matrixd_csv(const Matrixd& M, const std::string& filename) {
    atomic_write(filename, [&](std::ofstream& out) {
        out.precision(17);
        int n = static_cast<int>(M.n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                out << M.at(i, j);
                if (j + 1 < n) out << ",";
            }
            out << "\n";
        }
    });
}

inline Matrix read_matrix_csv(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) throw std::runtime_error("Cannot open " + filename);

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
        if (inner != cols) throw std::runtime_error("Bad CSV row length in " + filename);
    }
    if (rows == 0) throw std::runtime_error("Empty matrix file: " + filename);
    if (rows != cols) throw std::runtime_error("Matrix must be square in " + filename);

    Matrix A(rows);
    A.data = std::move(raw);
    return A;
}

// Read a lattice basis CSV (rows are vectors). Unlike read_matrix_csv this does
// not require the matrix to be square, so m x d bases with m != d are accepted.
inline Lattice read_lattice_csv(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) throw std::runtime_error("Cannot open " + filename);

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
        if (inner != cols) throw std::runtime_error("Bad CSV row length in " + filename);
    }
    if (rows == 0) throw std::runtime_error("Empty lattice file: " + filename);

    Lattice L;
    L.m = (int)rows;
    L.d = (int)cols;
    L.data = std::move(raw);
    return L;
}

inline void write_lattice_csv(const Lattice& L, const std::string& filename) {
    atomic_write(filename, [&](std::ofstream& out) {
        for (int i = 0; i < L.m; ++i) {
            const int64_t* ri = L.row(i);
            for (int k = 0; k < L.d; ++k) {
                out << ri[k];
                if (k + 1 < L.d) out << ',';
            }
            out << '\n';
        }
    });
}

// Write a vector-of-rows integer matrix as CSV, one vector per line.
inline void write_rows_csv(const std::vector<std::vector<int64_t>>& rows,
                           const std::string& filename) {
    atomic_write(filename, [&](std::ofstream& out) {
        for (const auto& r : rows) {
            for (size_t k = 0; k < r.size(); ++k) {
                out << r[k];
                if (k + 1 < r.size()) out << ',';
            }
            out << '\n';
        }
    });
}

// Write a flat row-major buffer (rows x cols) as CSV, one vector per line.
inline void write_flat_csv(const std::vector<int64_t>& data, int rows, int cols,
                           const std::string& filename) {
    atomic_write(filename, [&](std::ofstream& out) {
        for (int i = 0; i < rows; ++i) {
            const int64_t* row = data.data() + (size_t)i * cols;
            for (int k = 0; k < cols; ++k) {
                out << row[k];
                if (k + 1 < cols) out << ',';
            }
            out << '\n';
        }
    });
}

// Format a possibly large nonnegative squared norm carried as long double.
inline std::string norm2_str(long double v) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(0);
    ss << v;
    return ss.str();
}

// Gram matrix of a lattice whose rows are the m basis vectors in dimension d:
// A = L L^T (m x m), A(i,j) = <row_i, row_j>. Symmetric, and positive definite
// for a full-rank integer basis.
inline Matrix gram_of(const Lattice& L) {
    const int m = L.m, d = L.d;
    Matrix A((size_t)m);
    for (int i = 0; i < m; ++i) {
        const int64_t* ri = L.row(i);
        for (int j = i; j < m; ++j) {
            const int64_t* rj = L.row(j);
            int64_t s = 0;
            for (int k = 0; k < d; ++k) s += ri[k] * rj[k];
            A.at(i, j) = s;
            A.at(j, i) = s;
        }
    }
    return A;
}

// New basis L2 = X^T L (m x d). With the annealer producing D = X^T A X and
// A = L L^T, we have L2 L2^T = X^T A X = D, so L2 is a basis of the same lattice
// whose Gram matrix is exactly D.
inline Lattice xt_times(const Matrix& X, const Lattice& L) {
    const int m = L.m, d = L.d;
    Lattice R;
    R.m = m;
    R.d = d;
    R.data.assign((size_t)m * d, 0);
    for (int i = 0; i < m; ++i) {
        int64_t* ri = R.row(i);
        for (int p = 0; p < m; ++p) {
            const int64_t xpi = X.at(p, i);   // (X^T)_{i,p} = X_{p,i}
            if (xpi == 0) continue;
            const int64_t* lp = L.row(p);
            for (int k = 0; k < d; ++k) ri[k] += xpi * lp[k];
        }
    }
    return R;
}

// a^e mod p by fast exponentiation. Requires 0 < p < 2^31 so every product of
// two residues fits in int64.
inline int64_t powmod_small(int64_t a, int64_t e, int64_t p) {
    int64_t r = 1 % p;
    a %= p; if (a < 0) a += p;
    while (e > 0) {
        if (e & 1) r = (r * a) % p;
        a = (a * a) % p;
        e >>= 1;
    }
    return r;
}

// Determinant of A modulo a prime p (p < 2^31) by Gaussian elimination over Z/p.
// All intermediate products are below p^2 < 2^62, so plain int64 suffices.
inline int64_t det_mod_prime(const Matrix& A, int64_t p) {
    const int n = (int)A.n;
    std::vector<int64_t> m((size_t)n * n);
    for (size_t k = 0; k < m.size(); ++k) {
        int64_t v = A.data[k] % p;
        m[k] = v < 0 ? v + p : v;
    }
    auto at = [&](int i, int j) -> int64_t& { return m[(size_t)i * n + j]; };
    int64_t det = 1 % p;
    for (int col = 0; col < n; ++col) {
        int piv = -1;
        for (int r = col; r < n; ++r) if (at(r, col) != 0) { piv = r; break; }
        if (piv < 0) return 0;                 // singular -> det = 0
        if (piv != col) {
            for (int j = 0; j < n; ++j) std::swap(at(piv, j), at(col, j));
            det = (p - det) % p;               // a row swap flips the sign
        }
        const int64_t pv = at(col, col);
        det = (det * pv) % p;
        const int64_t inv = powmod_small(pv, p - 2, p);   // Fermat inverse
        for (int r = col + 1; r < n; ++r) {
            const int64_t f = (at(r, col) * inv) % p;
            if (f == 0) continue;
            for (int j = col; j < n; ++j) {
                int64_t v = at(r, j) - (f * at(col, j)) % p;
                v %= p; if (v < 0) v += p;
                at(r, j) = v;
            }
        }
    }
    return det % p;
}

// Probabilistic exact test that det(A) == +/-1 (i.e. A is unimodular). It
// computes det(A) modulo a couple of large primes. A unimodular matrix always
// passes (+/-1 is +/-1 mod every p), and a non-unimodular one passes only if its
// determinant is congruent to +1 (or to -1) modulo every prime, which for
// distinct ~10^9 primes has negligible probability ~ (1/p)^k.
inline bool is_unimodular(const Matrix& A) {
    static const int64_t primes[] = { 1000000007LL, 998244353LL };
    bool all_pos = true, all_neg = true;
    for (int64_t p : primes) {
        const int64_t d = det_mod_prime(A, p);
        if (d != 1) all_pos = false;
        if (d != p - 1) all_neg = false;
    }
    return all_pos || all_neg;
}
