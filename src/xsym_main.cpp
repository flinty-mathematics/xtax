// xsym: symmetry-directed lattice reduction and short-vector search.
//
// The idea. A unimodular integer matrix C acts on basis coordinates by
// x -> x C, which maps the lattice onto itself exactly, whatever C is. If in
// addition C preserves the metric (C P C^T = P for the working Gram P) then the
// map is an isometry, so it sends short vectors to short vectors of the same
// length: one short vector immediately yields its whole orbit for free. That is
// the mechanism that makes reduction on ideal / structured lattices cheaper than
// on generic ones, and it is normally available only when the lattice is handed
// to us with a known ring structure.
//
// xsym goes looking for the structure instead. For a fixed C of finite order it
// anneals over unimodular changes of basis P = X^T G X, minimizing
//
//     E(X) = ||C P C^T - P||_F^2   (+ a weighted basis-quality term)
//
// so the search asks: is there a basis of this lattice whose Gram is (nearly)
// C-invariant? E = 0 certifies an exact automorphism of the lattice conjugate to
// C, and the orbit map is then a genuine isometry. E small gives a near-isometry,
// whose orbits are still cheap and still short, which is all the sieve needs.
//
// Completeness of the parametrization. Over Z[i] (and Z[omega], and any PID
// order) every integer matrix T with T^2 = -I is GL(n,Z)-conjugate to the
// standard block form, because a torsion-free module over a PID is free. So
// searching over X with C fixed misses no integer complex structure: it covers
// the whole conjugacy class. The same argument covers the other cyclotomic
// targets whose order is a PID.
//
// What the discovered symmetry is used for:
//   - Orbit enrichment. Every vector the sieve touches contributes its orbit
//     x C^k at O(n) per image, so the database grows by up to the order of C for
//     the price of one vector. All images are exact lattice vectors regardless of
//     the distortion, so nothing here can produce a wrong answer.
//   - Orbit differences. v - v C^k is a lattice vector whose length is governed
//     by how far C moves v, so a near-symmetry that nearly fixes a direction
//     hands over a short vector directly.
//   - Basis quality. The discovery objective carries a basis-quality term, so the
//     annealer doubles as a reducer (this is the xtax / xdual congruence search
//     applied to a new objective).
//
// Everything else is a from-scratch reducer in the style of xbkz: double
// Gram-Schmidt, LLL, pruned Schnorr-Euchner enumeration for BKZ tours, int64
// basis entries with overflow checks. There is no bignum path, so very large raw
// entries are out of scope, exactly as in xbkz.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "CLI11.hpp"

// Shared helpers: Matrix / Lattice / CSV IO / ext_gcd / axpy_overflow
// (mat_io.hpp), the Ctrl-C plumbing (stop_signal.hpp), and the templated
// annealing engine the discovery search plugs into (congruence_anneal.hpp).
#include "congruence_anneal.hpp"
#include "mat_io.hpp"
#include "stop_signal.hpp"

// Floating-point type of the reduction core. Kept as one alias so the precision
// policy lives in one place, and fixed to double so results agree across the big
// three platforms (the x87 80-bit long double would differ).
using real_t = double;
using Clock = std::chrono::steady_clock;

// Thrown when an int64 basis entry would overflow during reduction. Callers roll
// back to a checkpoint and keep the basis they already had.
struct ReduceOverflow {};

// Thrown when a bounded reduction exceeds its work cap, so a speculative
// insertion can be abandoned without hanging the round.
struct ReduceAbort {};

static double elapsed_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Wall-clock budget for the whole run. Every stage that can take unbounded time
// polls this, so the budget is honoured to within one enumeration block rather
// than one round: a single BKZ tour on a bad profile can otherwise run for
// minutes past the deadline.
static Clock::time_point g_t0;
static double g_max_seconds = 0.0;

static bool out_of_time() {
    return g_max_seconds > 0.0 && elapsed_since(g_t0) >= g_max_seconds;
}

static bool should_stop() {
    return g_stop.load(std::memory_order_relaxed) || out_of_time();
}

// ---------------------------------------------------------------------------
// Symmetry targets
// ---------------------------------------------------------------------------
// Each mode below builds one fixed integer matrix C of known finite order, to be
// used as the target of the discovery search. Requirements on C: integer,
// unimodular (so x -> x C maps the lattice onto itself), of finite order (so the
// orbits are finite), and sparse with small entries (so the annealer's
// incremental updates stay O(n) and the exact int64 arithmetic cannot overflow).
//
// The modes are deliberately kept as independent builders rather than one
// parametrized routine: the algebra behind each family is different and mixing
// them would obscure all of them.

enum class SymMode { None, Gauss, Eisen3, Eisen6, Cyclic, NegaCyclic, Cyclo, Perm };

static const char* sym_mode_name(SymMode m) {
    switch (m) {
        case SymMode::None:       return "none";
        case SymMode::Gauss:      return "gauss";
        case SymMode::Eisen3:     return "eisen3";
        case SymMode::Eisen6:     return "eisen6";
        case SymMode::Cyclic:     return "cyclic";
        case SymMode::NegaCyclic: return "negacyclic";
        case SymMode::Cyclo:      return "cyclo";
        case SymMode::Perm:       return "perm";
    }
    return "?";
}

static bool parse_sym_mode(const std::string& s, SymMode& out) {
    if (s == "none")            { out = SymMode::None;       return true; }
    if (s == "gauss")           { out = SymMode::Gauss;      return true; }
    if (s == "eisen3")          { out = SymMode::Eisen3;     return true; }
    if (s == "eisen6")          { out = SymMode::Eisen6;     return true; }
    if (s == "cyclic")          { out = SymMode::Cyclic;     return true; }
    if (s == "negacyclic")      { out = SymMode::NegaCyclic; return true; }
    if (s == "cyclo")           { out = SymMode::Cyclo;      return true; }
    if (s == "perm")            { out = SymMode::Perm;       return true; }
    return false;
}

// Gaussian mode: the standard complex structure, block diagonal copies of
// [[0,-1],[1,0]] (multiplication by i on each coordinate pair). Order 4, and
// C^2 = -I, so a C-invariant Gram makes the lattice a Z[i]-lattice of half rank.
// Needs an even dimension.
static Matrix make_target_gauss(int n) {
    Matrix C((size_t)n);
    for (int b = 0; b + 1 < n; b += 2) {
        C.at(b, b + 1) = -1;
        C.at(b + 1, b) = 1;
    }
    return C;
}

// Eisenstein mode: block diagonal copies of the companion matrix of
// x^2 + x + 1, that is multiplication by a primitive cube root of unity on each
// coordinate pair. Order 3. order6 negates it, giving order 6 (multiplication by
// -omega). The invariant form is the hexagonal one, not the identity, which the
// objective finds on its own since it only asks for C-invariance.
// Needs an even dimension.
static Matrix make_target_eisen(int n, bool order6) {
    Matrix C((size_t)n);
    const int64_t s = order6 ? -1 : 1;
    for (int b = 0; b + 1 < n; b += 2) {
        C.at(b, b + 1)     = -s;
        C.at(b + 1, b)     = s;
        C.at(b + 1, b + 1) = -s;
    }
    return C;
}

// Cyclic mode: block diagonal copies of the companion matrix of x^k - 1, that is
// the cyclic coordinate shift on each block of k basis vectors. With k = n this
// is the plain circulant (cyclic ideal) hypothesis. Smaller k covers module
// lattices, whose basis splits into several blocks that rotate together. Order k.
static Matrix make_target_cyclic(int n, int k) {
    Matrix C((size_t)n);
    for (int b = 0; b + k <= n; b += k)
        for (int a = 0; a < k; ++a) C.at(b + a, b + (a + 1) % k) = 1;
    // Any leftover coordinates are fixed, which leaves them out of the hypothesis
    // rather than making a claim about them.
    for (int a = (n / k) * k; a < n; ++a) C.at(a, a) = 1;
    return C;
}

// Negacyclic mode: block diagonal copies of the companion matrix of x^k + 1, the
// signed cyclic shift. This is the anticirculant structure behind NTRU and
// ring-LWE lattices: with k equal to the ring degree, a module lattice of rank
// m over the ring has m blocks that rotate in step. It is a signed permutation,
// so orbit images need no arithmetic at all. Order 2k.
static Matrix make_target_negacyclic(int n, int k) {
    Matrix C((size_t)n);
    for (int b = 0; b + k <= n; b += k) {
        for (int a = 0; a + 1 < k; ++a) C.at(b + a, b + a + 1) = 1;
        C.at(b + k - 1, b) = -1;
    }
    for (int a = (n / k) * k; a < n; ++a) C.at(a, a) = 1;
    return C;
}

// Coefficients of the m-th cyclotomic polynomial Phi_m, lowest degree first,
// computed by dividing x^m - 1 by every Phi_d with d | m, d < m. Exact integer
// arithmetic: each division is exact by construction.
static std::vector<int64_t> cyclotomic_coeffs(int m) {
    std::vector<int64_t> num((size_t)m + 1, 0);
    num[0] = -1;
    num[(size_t)m] = 1;
    for (int d = 1; d < m; ++d) {
        if (m % d != 0) continue;
        const std::vector<int64_t> div = cyclotomic_coeffs(d);
        // Long division num /= div, exact.
        const int dn = (int)num.size() - 1;
        const int dd = (int)div.size() - 1;
        std::vector<int64_t> q((size_t)(dn - dd) + 1, 0);
        std::vector<int64_t> rem = num;
        for (int k = dn - dd; k >= 0; --k) {
            const int64_t c = rem[(size_t)(k + dd)] / div[(size_t)dd];
            q[(size_t)k] = c;
            if (c == 0) continue;
            for (int t = 0; t <= dd; ++t) rem[(size_t)(k + t)] -= c * div[(size_t)t];
        }
        num = q;
    }
    return num;
}

// Cyclotomic mode: block diagonal copies of the companion matrix of Phi_m, that
// is multiplication by a primitive m-th root of unity on each block of size
// phi(m). Order m, so orbits have m members. Covers gauss (m = 4) and eisen
// (m = 3) as special cases but is kept separate so those two stay simple and so
// this one can be pointed at any m whose degree divides the dimension.
static Matrix make_target_cyclo(int n, int m) {
    const std::vector<int64_t> phi = cyclotomic_coeffs(m);
    const int deg = (int)phi.size() - 1;
    Matrix C((size_t)n);
    for (int b = 0; b + deg <= n; b += deg) {
        // Companion matrix acting on the right: row a maps to row a+1, and the
        // last row folds back with the negated polynomial coefficients.
        for (int a = 0; a + 1 < deg; ++a) C.at(b + a, b + a + 1) = 1;
        for (int t = 0; t < deg; ++t) C.at(b + deg - 1, b + t) = -phi[(size_t)t];
    }
    return C;
}

// Signed permutation mode: a random signed permutation whose cycles all have
// length cyc, so its order is cyc (or 2 * cyc when a cycle carries an odd number
// of minus signs). Orbit images are a permutation plus sign flips, the cheapest
// orbits available. Coordinates left over when cyc does not divide n are fixed
// with a plus sign.
static Matrix make_target_perm(int n, int cyc, std::mt19937& rng) {
    std::vector<int> perm((size_t)n);
    for (int a = 0; a < n; ++a) perm[(size_t)a] = a;
    std::shuffle(perm.begin(), perm.end(), rng);
    Matrix C((size_t)n);
    int a = 0;
    for (; a + cyc <= n; a += cyc) {
        for (int t = 0; t < cyc; ++t) {
            const int from = perm[(size_t)(a + t)];
            const int to = perm[(size_t)(a + (t + 1) % cyc)];
            // One minus sign per cycle keeps the cycle fixed-point free in the
            // signed sense and doubles the orbit length.
            C.at(from, to) = (t + 1 == cyc) ? -1 : 1;
        }
    }
    for (; a < n; ++a) {
        const int idx = perm[(size_t)a];
        C.at(idx, idx) = 1;
    }
    return C;
}

// A symmetry target plus the sparse views the hot paths need: rows for x -> x C
// and for C u, columns for C e_j.
struct SymTarget {
    int n = 0;
    int order = 0;
    Matrix C;
    std::vector<std::vector<std::pair<int, int64_t>>> row;   // row[a] = {(b, C_ab)}
    std::vector<std::vector<std::pair<int, int64_t>>> col;    // col[b] = {(a, C_ab)}
    int64_t row_abs_max = 0;   // max_a sum_b |C_ab|, bounds the entry growth

    void build(const Matrix& Cin, int order_in) {
        C = Cin;
        n = (int)Cin.n;
        order = order_in;
        row.assign((size_t)n, {});
        col.assign((size_t)n, {});
        row_abs_max = 0;
        for (int a = 0; a < n; ++a) {
            int64_t s = 0;
            for (int b = 0; b < n; ++b) {
                const int64_t v = C.at(a, b);
                if (v == 0) continue;
                row[(size_t)a].push_back({ b, v });
                col[(size_t)b].push_back({ a, v });
                s += std::llabs(v);
            }
            row_abs_max = std::max(row_abs_max, s);
        }
    }

    // out = x * C for a row vector x. O(nonzeros of C touched).
    void apply_right(const std::vector<int64_t>& x, std::vector<int64_t>& out) const {
        out.assign((size_t)n, 0);
        for (int a = 0; a < n; ++a) {
            const int64_t xa = x[(size_t)a];
            if (xa == 0) continue;
            for (const auto& e : row[(size_t)a]) out[(size_t)e.first] += xa * e.second;
        }
    }
};

// Build the target matrix for a mode, or report why the mode does not fit this
// dimension. order receives the multiplicative order of the result.
static bool build_sym_target(SymMode mode, int n, int cyclo_order, int perm_cycle,
                             int sym_block, std::mt19937& rng, SymTarget& out,
                             std::string& err) {
    // Block size for the two shift families: 0 means one block spanning the whole
    // rank (the plain ideal-lattice hypothesis).
    const int k = (sym_block > 0) ? sym_block : n;
    if (mode == SymMode::Cyclic || mode == SymMode::NegaCyclic) {
        if (k < 2 || k > n) {
            err = "--sym-block must be in [2, rank]";
            return false;
        }
        if (n % k != 0)
            std::cout << "[xsym] warning: --sym-block " << k << " does not divide the "
                      << "rank " << n << ", the leftover coordinates are left fixed\n";
    }
    switch (mode) {
        case SymMode::None:
            err = "no symmetry mode selected";
            return false;
        case SymMode::Gauss:
            if (n % 2 != 0) { err = "gauss mode needs an even dimension"; return false; }
            out.build(make_target_gauss(n), 4);
            return true;
        case SymMode::Eisen3:
            if (n % 2 != 0) { err = "eisen3 mode needs an even dimension"; return false; }
            out.build(make_target_eisen(n, false), 3);
            return true;
        case SymMode::Eisen6:
            if (n % 2 != 0) { err = "eisen6 mode needs an even dimension"; return false; }
            out.build(make_target_eisen(n, true), 6);
            return true;
        case SymMode::Cyclic:
            out.build(make_target_cyclic(n, k), k);
            return true;
        case SymMode::NegaCyclic:
            out.build(make_target_negacyclic(n, k), 2 * k);
            return true;
        case SymMode::Cyclo: {
            if (cyclo_order < 3) { err = "--cyclo-order must be at least 3"; return false; }
            const int deg = (int)cyclotomic_coeffs(cyclo_order).size() - 1;
            if (deg <= 0 || n % deg != 0) {
                err = "cyclo mode needs deg(Phi_" + std::to_string(cyclo_order) +
                      ") = " + std::to_string(deg) + " to divide the dimension " +
                      std::to_string(n);
                return false;
            }
            out.build(make_target_cyclo(n, cyclo_order), cyclo_order);
            return true;
        }
        case SymMode::Perm: {
            if (perm_cycle < 2 || perm_cycle > n) {
                err = "--perm-cycle must be in [2, dimension]";
                return false;
            }
            out.build(make_target_perm(n, perm_cycle, rng), 2 * perm_cycle);
            return true;
        }
    }
    err = "unknown mode";
    return false;
}

// ---------------------------------------------------------------------------
// Ambient symmetries
// ---------------------------------------------------------------------------
// The coordinate matrix C above expresses a symmetry relative to one basis, and
// a change of basis conjugates it. An ambient symmetry is the same map written on
// the ambient space instead, so it does not depend on the basis at all, and
// checking whether the lattice is stable under it is a direct question: is J b_i
// in the lattice for every basis vector?
//
// Restricting to integer orthogonal maps makes this exact and cheap, and costs
// nothing in generality: an integer matrix is orthogonal exactly when it is a
// signed permutation. The rotations behind ideal, NTRU and module lattices are
// all of this kind, and they are the cases where an orbit is most valuable, so
// this is the fast path. It also sidesteps a trap: a q-ary basis carries the
// rotation only up to a multiple of q, which makes the coordinate matrix block
// triangular rather than block diagonal, so a basis-coordinate test on the given
// basis would miss a symmetry the lattice really has.
struct AmbientSym {
    int d = 0;
    int order = 0;
    std::vector<int> perm;      // coordinate k of the image comes from perm[k]
    std::vector<int64_t> sign;  // with this sign

    void apply(const std::vector<int64_t>& v, std::vector<int64_t>& out) const {
        out.assign((size_t)d, 0);
        for (int k = 0; k < d; ++k)
            out[(size_t)k] = sign[(size_t)k] * v[(size_t)perm[(size_t)k]];
    }
};

// The simultaneous negacyclic rotation of every block of `block` ambient
// coordinates: within a block, x_t -> x_{t+1} and x_{block-1} -> -x_0. This is
// multiplication by the ring generator in Z[x]/(x^block + 1) acting on a module
// lattice of rank d / block. Order 2 * block.
static AmbientSym make_ambient_negacyclic(int d, int block) {
    AmbientSym J;
    J.d = d;
    J.order = 2 * block;
    J.perm.assign((size_t)d, 0);
    J.sign.assign((size_t)d, 1);
    for (int b = 0; b + block <= d; b += block)
        for (int t = 0; t < block; ++t) {
            // Image coordinate b + t is the source coordinate one step back.
            const int src = (t == 0) ? (block - 1) : (t - 1);
            J.perm[(size_t)(b + t)] = b + src;
            J.sign[(size_t)(b + t)] = (t == 0) ? -1 : 1;
        }
    for (int k = (d / block) * block; k < d; ++k) J.perm[(size_t)k] = k;
    return J;
}

// The same for x^block - 1: a plain cyclic rotation of each block, order block.
static AmbientSym make_ambient_cyclic(int d, int block) {
    AmbientSym J;
    J.d = d;
    J.order = block;
    J.perm.assign((size_t)d, 0);
    J.sign.assign((size_t)d, 1);
    for (int b = 0; b + block <= d; b += block)
        for (int t = 0; t < block; ++t) {
            const int src = (t == 0) ? (block - 1) : (t - 1);
            J.perm[(size_t)(b + t)] = b + src;
        }
    for (int k = (d / block) * block; k < d; ++k) J.perm[(size_t)k] = k;
    return J;
}

// ---------------------------------------------------------------------------
// Discovery objective: anneal P = X^T G X toward C-invariance
// ---------------------------------------------------------------------------
// The objective policy for congruence_anneal.hpp. State is the working Gram P
// (exact int64), the accumulated transform X, and the exact residual
// M = C P C^T - P. The score is
//
//     score = sym_w * ||M||_F^2 + red_w * ||offdiag(P)||_F^2
//
// with both terms normalized by their initial values, so --sym-lambda 1 gives
// the basis-quality term the same starting weight as the symmetry term. The
// residual term alone is scale free only up to det(P), which a unimodular
// congruence leaves fixed, so no extra normalization is needed inside a run.
//
// An Add(i, j, s) move sends P to P + s(u e_j^T + e_j u^T) + s^2 P_ii e_j e_j^T
// with u the i-th column of P. Conjugating by C turns each outer product into
// one built from w = C u and p = C e_j, so
//
//     M' = M + s(w p^T + p w^T - u e_j^T - e_j u^T)
//            + s^2 P_ii (p p^T - e_j e_j^T).
//
// p is a single sparse column of C, so both the score delta and the commit are
// O(n): the delta needs a handful of dot products plus w^T M p and (M u)_j, and
// the commit touches only the rows and columns in the support of p together with
// row and column j.
struct SymObjective {
    using score_t = long double;

    SymTarget T;
    Matrix P;        // working Gram, exact
    Matrix Xt;       // accumulated transform, P = Xt^T G Xt
    Matrix M;        // C P C^T - P, exact

    long double sym_w = 1.0L;
    long double red_w = 0.0L;
    score_t score_ = 0;
    long double sym_raw = 0.0L;     // ||M||_F^2 at the last exact recompute
    long double red_raw = 0.0L;     // ||offdiag(P)||_F^2 at the last exact recompute
    long double p_frob2 = 0.0L;     // ||P||_F^2, the scale the residual is measured against
    int64_t sym_nz = 0;             // nonzero entries of M
    int recompute_period = 200000;

    std::vector<int64_t> mrow_abs;  // per-row sum |M_ab|, the targeting weight

    // Scratch reused by the O(n) delta and commit paths. Not part of the state.
    // The delta path works in long double (it only needs a good score estimate),
    // the commit path in int64 so the residual M stays exact.
    mutable std::vector<long double> u_buf, w_buf;
    std::vector<int64_t> ui_buf, wi_buf;

    int n() const { return (int)P.n; }
    score_t score() const { return score_; }
    int64_t offdiag_nonzero() const { return sym_nz; }
    bool solved() const { return sym_nz == 0; }
    int64_t row_weight(int r) const { return mrow_abs[(size_t)r]; }
    int64_t pivot_abs(int t, int c) const { return std::llabs(P.at(t, c)); }
    double suggest_t_init() const {
        const double s = (double)score_;
        const double d = (double)std::max(1, n());
        return s > 0.0 ? std::max(1e-6, s / (4.0 * d)) : 1.0;
    }
    void reorder_for_publish(bool) {}

    // Exact rebuild of M, the two raw score terms, the nonzero count and the
    // targeting weights from P. O(n^3) via the sparse C, so it is used at setup,
    // on a reseed, and periodically to clear incremental drift.
    void rebuild() {
        const int nn = n();
        // CP = C * P using the sparse rows of C.
        Matrix CP((size_t)nn);
        for (int a = 0; a < nn; ++a) {
            for (const auto& e : T.row[(size_t)a]) {
                const int64_t v = e.second;
                for (int b = 0; b < nn; ++b) CP.at(a, b) += v * P.at(e.first, b);
            }
        }
        // M = CP * C^T - P, where (CP * C^T)_{a,b} = sum_t CP_{a,t} C_{b,t}.
        M = Matrix((size_t)nn);
        for (int b = 0; b < nn; ++b) {
            for (const auto& e : T.row[(size_t)b]) {
                const int64_t v = e.second;
                for (int a = 0; a < nn; ++a) M.at(a, b) += v * CP.at(a, e.first);
            }
        }
        for (int a = 0; a < nn; ++a)
            for (int b = 0; b < nn; ++b) M.at(a, b) -= P.at(a, b);

        sym_raw = 0.0L;
        sym_nz = 0;
        mrow_abs.assign((size_t)nn, 0);
        for (int a = 0; a < nn; ++a) {
            int64_t abs_sum = 0;
            for (int b = 0; b < nn; ++b) {
                const int64_t v = M.at(a, b);
                if (v == 0) continue;
                const long double lv = (long double)v;
                sym_raw += lv * lv;
                ++sym_nz;
                abs_sum += std::llabs(v);
            }
            mrow_abs[(size_t)a] = abs_sum;
        }
        red_raw = P.offdiag_frob2();
        p_frob2 = 0.0L;
        for (const int64_t e : P.data) {
            const long double v = (long double)e;
            p_frob2 += v * v;
        }
        score_ = sym_w * sym_raw + red_w * red_raw;
        u_buf.assign((size_t)nn, 0.0L);
        w_buf.assign((size_t)nn, 0.0L);
        ui_buf.assign((size_t)nn, 0);
        wi_buf.assign((size_t)nn, 0);
    }

    void refresh_cache() { rebuild(); }
    score_t recompute_score() { rebuild(); return score_; }
    void periodic_maintenance(uint64_t moves) {
        if (recompute_period > 0 && moves % (uint64_t)recompute_period == 0) rebuild();
    }

    // Score change of Add(i, j, s), with no mutation. Returns false when the move
    // would push a Gram entry past the magnitude limit.
    bool delta_of(int i, int j, int64_t s, score_t& d_score) const {
        if (s == 0) return false;
        const int nn = n();
        const int64_t Pii = P.at(i, i);

        // Feasibility: the changed entries are row / column j and the diagonal.
        for (int b = 0; b < nn; ++b) {
            const long double v = (long double)P.at(j, b) + (long double)s * (long double)P.at(i, b);
            if (v > (long double)MAGNITUDE_LIMIT || v < -(long double)MAGNITUDE_LIMIT)
                return false;
        }
        {
            const long double vjj = (long double)P.at(j, j) +
                                    2.0L * (long double)s * (long double)P.at(i, j) +
                                    (long double)s * (long double)s * (long double)Pii;
            if (vjj > (long double)MAGNITUDE_LIMIT || vjj < -(long double)MAGNITUDE_LIMIT)
                return false;
        }

        // u = column i of P, w = C u, p = column j of C (sparse).
        std::vector<long double>& u = u_buf;
        std::vector<long double>& w = w_buf;
        for (int b = 0; b < nn; ++b) u[(size_t)b] = (long double)P.at(b, i);
        for (int a = 0; a < nn; ++a) {
            long double acc = 0.0L;
            for (const auto& e : T.row[(size_t)a])
                acc += (long double)e.second * u[(size_t)e.first];
            w[(size_t)a] = acc;
        }
        const auto& pcol = T.col[(size_t)j];

        // Dot products among {w, p, u, e_j}.
        long double ww = 0.0L, uu = 0.0L, wu = 0.0L;
        for (int a = 0; a < nn; ++a) {
            ww += w[(size_t)a] * w[(size_t)a];
            uu += u[(size_t)a] * u[(size_t)a];
            wu += w[(size_t)a] * u[(size_t)a];
        }
        long double pp = 0.0L, wp = 0.0L, pu = 0.0L, pj = 0.0L;
        for (const auto& e : pcol) {
            const long double pv = (long double)e.second;
            pp += pv * pv;
            wp += pv * w[(size_t)e.first];
            pu += pv * u[(size_t)e.first];
            if (e.first == j) pj += pv;
        }
        const long double wj = w[(size_t)j];
        const long double uj = u[(size_t)j];

        // The M-weighted terms: w^T M p, (M u)_j, p^T M p, M_jj.
        long double wMp = 0.0L, pMp = 0.0L;
        for (const auto& e : pcol) {
            const long double pv = (long double)e.second;
            const int b = e.first;
            long double col_dot = 0.0L;
            for (int a = 0; a < nn; ++a) col_dot += w[(size_t)a] * (long double)M.at(a, b);
            wMp += pv * col_dot;
            long double pcol_dot = 0.0L;
            for (const auto& f : pcol)
                pcol_dot += (long double)f.second * (long double)M.at(f.first, b);
            pMp += pv * pcol_dot;
        }
        long double Muj = 0.0L;
        for (int b = 0; b < nn; ++b) Muj += (long double)M.at(j, b) * u[(size_t)b];
        const long double Mjj = (long double)M.at(j, j);

        const long double ls = (long double)s;
        const long double q = ls * ls * (long double)Pii;

        // <M, Delta>
        const long double inner = 2.0L * ls * (wMp - Muj) + q * (pMp - Mjj);

        // ||Delta||^2 with Delta = s*A1 + q*A2, A1 = (w p^T + p w^T) - (u e_j^T +
        // e_j u^T), A2 = p p^T - e_j e_j^T.
        const long double n_S1 = 2.0L * (ww * pp + wp * wp);
        const long double n_S2 = 2.0L * (uu + uj * uj);
        const long double i_S1S2 = 2.0L * (wu * pj + wj * pu);
        const long double n_A1 = n_S1 - 2.0L * i_S1S2 + n_S2;
        const long double n_A2 = pp * pp - 2.0L * pj * pj + 1.0L;
        const long double i_A1A2 = 2.0L * (wp * pp - wj * pj - pu * pj + uj);
        const long double d_sym = 2.0L * inner + ls * ls * n_A1 +
                                  2.0L * ls * q * i_A1A2 + q * q * n_A2;

        // Basis-quality term: only row / column j of the off-diagonal changes.
        long double d_red = 0.0L;
        if (red_w != 0.0L) {
            long double cross = 0.0L, sq = 0.0L;
            for (int b = 0; b < nn; ++b) {
                if (b == j) continue;
                const long double pjb = (long double)P.at(j, b);
                const long double pib = (long double)P.at(i, b);
                cross += pjb * pib;
                sq += pib * pib;
            }
            d_red = 4.0L * ls * cross + 2.0L * ls * ls * sq;
        }

        d_score = sym_w * d_sym + red_w * d_red;
        return true;
    }

    bool evaluate(int i, int j, int64_t s, score_t& d_score, int64_t& d_nonzero) {
        d_nonzero = 0;   // the exact count is refreshed by rebuild()
        return delta_of(i, j, s, d_score);
    }

    // Apply Add(i, j, s) to P, M and Xt. Every touched buffer is recomputed here
    // rather than carried over from evaluate, which keeps best_shear const-safe.
    bool commit(int i, int j, int64_t s, score_t d_score, int64_t) {
        const int nn = n();
        // Transform first: it is the only part that can still fail.
        for (int a = 0; a < nn; ++a) {
            const int64_t xi = Xt.at(a, i);
            if (xi == 0) continue;
            const long double v = (long double)Xt.at(a, j) + (long double)s * (long double)xi;
            if (v > (long double)MAGNITUDE_LIMIT || v < -(long double)MAGNITUDE_LIMIT)
                return false;
        }

        std::vector<int64_t>& u = ui_buf;
        std::vector<int64_t>& w = wi_buf;
        for (int b = 0; b < nn; ++b) u[(size_t)b] = P.at(b, i);
        for (int a = 0; a < nn; ++a) {
            int64_t acc = 0;
            for (const auto& e : T.row[(size_t)a]) acc += e.second * u[(size_t)e.first];
            w[(size_t)a] = acc;
        }
        const auto& pcol = T.col[(size_t)j];
        const int64_t Pii = P.at(i, i);

        // M += s(w p^T + p w^T): touches the columns and rows in supp(p). When
        // a == b the entry legitimately receives both outer products.
        for (const auto& e : pcol) {
            const int b = e.first;
            const int64_t pv = e.second;
            for (int a = 0; a < nn; ++a) {
                const int64_t t = s * w[(size_t)a] * pv;
                if (t == 0) continue;
                M.at(a, b) += t;
                M.at(b, a) += t;
            }
        }
        // M -= s(u e_j^T + e_j u^T): touches row and column j. The (j,j) entry
        // legitimately receives the update twice.
        for (int a = 0; a < nn; ++a) {
            const int64_t t = -s * u[(size_t)a];
            if (t == 0) continue;
            M.at(a, j) += t;
            M.at(j, a) += t;
        }
        // M += s^2 P_ii (p p^T - e_j e_j^T).
        for (const auto& e : pcol) {
            for (const auto& f : pcol)
                M.at(e.first, f.first) += s * s * Pii * e.second * f.second;
        }
        M.at(j, j) -= s * s * Pii;

        // P update: row and column j gain s times row and column i.
        const int64_t Pij = P.at(i, j);
        for (int b = 0; b < nn; ++b) {
            if (b == j) continue;
            const int64_t v = P.at(j, b) + s * P.at(i, b);
            P.at(j, b) = v;
            P.at(b, j) = v;
        }
        P.at(j, j) = P.at(j, j) + 2 * s * Pij + s * s * Pii;

        // Transform update: column j gains s times column i.
        for (int a = 0; a < nn; ++a) {
            const int64_t xi = Xt.at(a, i);
            if (xi == 0) continue;
            Xt.at(a, j) += s * xi;
        }

        // Refresh the targeting weights of the rows that changed.
        auto refresh_row = [&](int a) {
            int64_t acc = 0;
            for (int b = 0; b < nn; ++b) acc += std::llabs(M.at(a, b));
            mrow_abs[(size_t)a] = acc;
        };
        refresh_row(j);
        for (const auto& e : pcol) refresh_row(e.first);

        score_ += d_score;
        return true;
    }

    // Best integer shear for the pair (i, j). The score is quartic in s, so
    // rather than solving it exactly this scans the neighbourhood of the shear
    // that is optimal for the basis-quality term, which is where the useful
    // moves sit, plus the two unit steps.
    int64_t best_shear(int i, int j) const {
        const int64_t Pii = P.at(i, i);
        int64_t s0 = 0;
        if (Pii > 0) s0 = -rounded_div(P.at(i, j), Pii);
        s0 = std::clamp(s0, -SHEAR_CAP, SHEAR_CAP);
        int64_t best_s = 0;
        score_t best_d = score_t(0);
        auto consider = [&](int64_t s) {
            if (s == 0 || s < -SHEAR_CAP || s > SHEAR_CAP) return;
            score_t d = score_t(0);
            if (!delta_of(i, j, s, d)) return;
            if (d < best_d) { best_d = d; best_s = s; }
        };
        for (int64_t off = -2; off <= 2; ++off) consider(s0 + off);
        consider(1);
        consider(-1);
        return best_s;
    }

    void publish_files() const {
        write_matrix_csv(Xt, "best_sym_X.csv");
        write_matrix_csv(P, "best_sym_P.csv");
    }

    // Symmetry mismatch relative to the scale of the Gram it is a mismatch of,
    // so it is comparable across bases and across instances. Zero means the
    // symmetry is exact.
    double residual() const {
        if (p_frob2 <= 0.0L) return 0.0;
        return (double)(sym_raw / p_frob2);
    }

    std::string best_line() const {
        std::ostringstream ss;
        ss.precision(6);
        ss << "score=" << (double)score_ << " residual=" << residual()
           << " sym_nz=" << sym_nz << " offdiag=" << (double)red_raw;
        return ss.str();
    }
};

// ---------------------------------------------------------------------------
// Reduction core
// ---------------------------------------------------------------------------
// One basis and its Gram-Schmidt data, plus the unimodular transform U with
// B = U * L0. Same design as the xbkz reducer: the Gram is cached and kept in
// sync incrementally, the GSO is recomputed from it, and all integer updates are
// overflow checked.
struct Red {
    int n = 0;
    int d = 0;
    double delta = 0.99;
    bool track_u = true;
    bool u_valid = true;

    std::vector<int64_t> B;
    std::vector<int64_t> U;
    std::vector<real_t> mu;
    std::vector<real_t> r;
    std::vector<real_t> G;

    std::vector<int64_t> snap_B, snap_U;
    std::vector<real_t> snap_mu, snap_r, snap_G;
    bool snap_u_valid = true;

    int64_t* b_row(int i) { return B.data() + (size_t)i * d; }
    const int64_t* b_row(int i) const { return B.data() + (size_t)i * d; }
    int64_t* u_row(int i) { return U.data() + (size_t)i * n; }
    const int64_t* u_row(int i) const { return U.data() + (size_t)i * n; }

    real_t& M(int i, int j) { return mu[(size_t)i * n + j]; }
    real_t  M(int i, int j) const { return mu[(size_t)i * n + j]; }
    real_t& Gx(int i, int j) { return G[(size_t)i * n + j]; }
    real_t  Gx(int i, int j) const { return G[(size_t)i * n + j]; }

    real_t dot(int i, int j) const {
        const int64_t* a = b_row(i);
        const int64_t* b = b_row(j);
        real_t s = 0;
        for (int k = 0; k < d; ++k) s += (real_t)a[k] * (real_t)b[k];
        return s;
    }

    // sum_{t < cnt} mu(i,t) mu(j,t) r[t], the Gram-Schmidt projection term.
    real_t gso_sum(int i, int j, int cnt) const {
        const real_t* mi = mu.data() + (size_t)i * n;
        const real_t* mj = mu.data() + (size_t)j * n;
        real_t s = 0;
        for (int t = 0; t < cnt; ++t) s += mi[t] * mj[t] * r[(size_t)t];
        return s;
    }

    void init(const Lattice& L, bool track) {
        n = L.m;
        d = L.d;
        track_u = track;
        u_valid = track;
        B.assign((size_t)n * d, 0);
        for (int i = 0; i < n; ++i)
            std::copy_n(L.row(i), d, b_row(i));
        if (track_u) {
            U.assign((size_t)n * n, 0);
            for (int i = 0; i < n; ++i) u_row(i)[i] = 1;
        }
        mu.assign((size_t)n * n, 0.0);
        r.assign((size_t)n, 0.0);
        G.assign((size_t)n * n, 0.0);
        build_gram();
        compute_gso();
    }

    void build_gram() {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                const real_t s = dot(i, j);
                Gx(i, j) = s;
                Gx(j, i) = s;
            }
    }

    void compute_gso_from(int from) {
        for (int i = from; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                const real_t s = Gx(i, j) - gso_sum(i, j, j);
                M(i, j) = (r[(size_t)j] > 0) ? s / r[(size_t)j] : 0.0;
            }
            r[(size_t)i] = Gx(i, i) - gso_sum(i, i, i);
            M(i, i) = 1.0;
        }
    }
    void compute_gso() { compute_gso_from(0); }

    void update_gso_after_block(int k0, int h) {
        const int kend = k0 + h;
        for (int i = k0; i < kend; ++i) {
            for (int j = 0; j < i; ++j) {
                const real_t s = Gx(i, j) - gso_sum(i, j, j);
                M(i, j) = (r[(size_t)j] > 0) ? s / r[(size_t)j] : 0.0;
            }
            r[(size_t)i] = Gx(i, i) - gso_sum(i, i, i);
            M(i, i) = 1.0;
        }
        for (int i = kend; i < n; ++i)
            for (int j = k0; j < kend; ++j) {
                const real_t s = Gx(i, j) - gso_sum(i, j, j);
                M(i, j) = (r[(size_t)j] > 0) ? s / r[(size_t)j] : 0.0;
            }
    }

    // B[dst] += c * B[src], keeping the Gram in sync. Throws on overflow with the
    // Gram untouched, so a caller can roll back.
    void axpy(int dst, int src, int64_t c) {
        if (c == 0) return;
        int64_t* bd = b_row(dst);
        const int64_t* bs = b_row(src);
        for (int k = 0; k < d; ++k)
            if (axpy_overflow(bd[k], c, bs[k])) throw ReduceOverflow{};

        const real_t cc = (real_t)c;
        const real_t old_ds = Gx(dst, src);
        const real_t old_ss = Gx(src, src);
        const real_t old_dd = Gx(dst, dst);
        for (int i = 0; i < n; ++i) {
            if (i == dst) continue;
            const real_t v = Gx(dst, i) + cc * Gx(src, i);
            Gx(dst, i) = v;
            Gx(i, dst) = v;
        }
        Gx(dst, dst) = old_dd + 2.0 * cc * old_ds + cc * cc * old_ss;

        if (track_u && u_valid) {
            int64_t* ud = u_row(dst);
            const int64_t* us = u_row(src);
            for (int k = 0; k < n; ++k)
                if (axpy_overflow(ud[k], c, us[k])) { u_valid = false; break; }
        }
    }

    void size_reduce(int k, int j) {
        const real_t m = M(k, j);
        if (m > -0.5 && m < 0.5) return;
        const long long q = std::llrint(m);
        if (q == 0) return;
        axpy(k, j, -(int64_t)q);
        for (int t = 0; t <= j; ++t) M(k, t) -= (real_t)q * M(j, t);
    }

    void swap_with_prev(int k) {
        std::swap_ranges(b_row(k), b_row(k) + d, b_row(k - 1));
        if (track_u) std::swap_ranges(u_row(k), u_row(k) + n, u_row(k - 1));
        for (int i = 0; i < n; ++i) std::swap(Gx(k, i), Gx(k - 1, i));
        for (int i = 0; i < n; ++i) std::swap(Gx(i, k), Gx(i, k - 1));

        const real_t m = M(k, k - 1);
        const real_t Bk = r[(size_t)k] + m * m * r[(size_t)k - 1];
        if (Bk < 1e-30) {
            update_gso_after_block(k - 1, 2);
            return;
        }
        M(k, k - 1) = m * r[(size_t)k - 1] / Bk;
        r[(size_t)k] = r[(size_t)k - 1] * r[(size_t)k] / Bk;
        r[(size_t)k - 1] = Bk;
        for (int j = 0; j < k - 1; ++j) std::swap(M(k - 1, j), M(k, j));
        for (int i = k + 1; i < n; ++i) {
            const real_t t = M(i, k);
            M(i, k) = M(i, k - 1) - m * t;
            M(i, k - 1) = t + M(k, k - 1) * M(i, k);
        }
    }

    void lll(int start = 1, long long max_steps = 0) {
        int k = std::max(1, start);
        long long steps = 0;
        while (k < n) {
            ++steps;
            if ((steps & 0xFF) == 0 && should_stop()) return;
            if (max_steps > 0 && steps > max_steps) throw ReduceAbort{};
            for (int j = k - 1; j >= 0; --j) size_reduce(k, j);
            const real_t lhs = r[(size_t)k];
            const real_t m = M(k, k - 1);
            const real_t rhs = ((real_t)delta - m * m) * r[(size_t)k - 1];
            if (lhs >= rhs) {
                ++k;
            } else {
                swap_with_prev(k);
                k = std::max(k - 1, 1);
            }
        }
    }

    void save_state() {
        snap_B = B;
        if (track_u) snap_U = U;
        snap_mu = mu;
        snap_r = r;
        snap_G = G;
        snap_u_valid = u_valid;
    }
    void restore_state() {
        B = snap_B;
        if (track_u) U = snap_U;
        mu = snap_mu;
        r = snap_r;
        G = snap_G;
        u_valid = snap_u_valid;
    }

    real_t row_norm2(int i) const {
        const int64_t* a = b_row(i);
        real_t s = 0;
        for (int k = 0; k < d; ++k) s += (real_t)a[k] * (real_t)a[k];
        return s;
    }

    real_t shortest_norm2(int* idx = nullptr) const {
        real_t best = row_norm2(0);
        int bi = 0;
        for (int i = 1; i < n; ++i) {
            const real_t v = row_norm2(i);
            if (v < best) { best = v; bi = i; }
        }
        if (idx) *idx = bi;
        return best;
    }

    // log of the basis determinant, from the Gram-Schmidt norms.
    double log_det() const {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += 0.5 * std::log(std::max(r[(size_t)i], 1e-300));
        return s;
    }

    // The LLL potential sum_i (n - i) log r_i. Every genuine improvement to the
    // basis lowers it, so it is the acceptance test for a speculative insertion
    // whose benefit is spread over the profile rather than sitting in b_0.
    double log_potential() const {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += (double)(n - i) * std::log(std::max(r[(size_t)i], 1e-300));
        return s;
    }

    // Root Hermite factor of the current basis: (|b_0| / det^(1/n))^(1/n).
    double root_hermite() const {
        const real_t n0 = row_norm2(0);
        if (n0 <= 0) return 0.0;
        const double ln = 0.5 * std::log((double)n0) - log_det() / (double)n;
        return std::exp(ln / (double)n);
    }

    Matrix gram_matrix() const {
        Matrix P((size_t)n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                int64_t s = 0;
                const int64_t* bi = b_row(i);
                const int64_t* bj = b_row(j);
                for (int k = 0; k < d; ++k) s += bi[k] * bj[k];
                P.at(i, j) = s;
            }
        return P;
    }

    // Recover the integer coordinates of a lattice vector v over the current
    // basis, using the Gram-Schmidt data and back substitution, then verify the
    // result exactly. Returns false if v is not in the lattice spanned by the
    // basis or the recovery is not exact (which can happen if the GSO has
    // drifted badly).
    bool coords_of(const int64_t* v, std::vector<int64_t>& x) const {
        std::vector<real_t> t((size_t)n, 0.0), sv((size_t)n, 0.0), c((size_t)n, 0.0);
        for (int i = 0; i < n; ++i) {
            const int64_t* bi = b_row(i);
            real_t s = 0;
            for (int k = 0; k < d; ++k) s += (real_t)bi[k] * (real_t)v[k];
            t[(size_t)i] = s;
        }
        for (int i = 0; i < n; ++i) {
            real_t s = t[(size_t)i];
            for (int j = 0; j < i; ++j) s -= M(i, j) * sv[(size_t)j];
            sv[(size_t)i] = s;
            c[(size_t)i] = (r[(size_t)i] > 0) ? s / r[(size_t)i] : 0.0;
        }
        std::vector<real_t> xr((size_t)n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            real_t s = c[(size_t)i];
            for (int j = i + 1; j < n; ++j) s -= xr[(size_t)j] * M(j, i);
            xr[(size_t)i] = s;
        }
        x.assign((size_t)n, 0);
        for (int i = 0; i < n; ++i) {
            const real_t rv = xr[(size_t)i];
            if (!(rv > -9.0e17 && rv < 9.0e17)) return false;
            x[(size_t)i] = (int64_t)std::llrint(rv);
        }
        // Exact verification.
        std::vector<int64_t> chk((size_t)d, 0);
        for (int i = 0; i < n; ++i) {
            const int64_t xi = x[(size_t)i];
            if (xi == 0) continue;
            const int64_t* bi = b_row(i);
            for (int k = 0; k < d; ++k)
                if (axpy_overflow(chk[(size_t)k], xi, bi[k])) return false;
        }
        for (int k = 0; k < d; ++k)
            if (chk[(size_t)k] != v[k]) return false;
        return true;
    }
};

// Divide a coefficient vector by the gcd of its entries and normalize its sign.
// Returns false when the vector is zero or its entries are implausibly large.
static bool coeff_make_primitive(std::vector<int64_t>& c) {
    int64_t g = 0;
    for (int64_t v : c) g = std::gcd(g, (int64_t)std::llabs((long long)v));
    if (g == 0) return false;
    if (g > 1) for (int64_t& v : c) v /= g;
    for (int64_t v : c) {
        if (v != 0) {
            if (v < 0) for (int64_t& w : c) w = -w;
            break;
        }
    }
    for (int64_t v : c)
        if (std::llabs((long long)v) > (1LL << 22)) return false;
    return true;
}

// Build a unimodular h x h matrix whose first row is the primitive vector x.
static std::vector<std::vector<int64_t>> complete_unimodular(std::vector<int64_t> x) {
    const int h = (int)x.size();

    // Fast path: a +/-1 coordinate lets the completion be x plus unit rows,
    // which keeps the transform tiny.
    int unit = -1;
    for (int i = 0; i < h; ++i)
        if (x[(size_t)i] == 1 || x[(size_t)i] == -1) { unit = i; break; }
    if (unit >= 0) {
        std::vector<std::vector<int64_t>> H((size_t)h, std::vector<int64_t>((size_t)h, 0));
        H[0] = x;
        int rr = 1;
        for (int j = 0; j < h; ++j) {
            if (j == unit) continue;
            H[(size_t)rr][(size_t)j] = 1;
            ++rr;
        }
        return H;
    }

    std::vector<std::vector<int64_t>> Cinv((size_t)h, std::vector<int64_t>((size_t)h, 0));
    for (int i = 0; i < h; ++i) Cinv[(size_t)i][(size_t)i] = 1;
    std::vector<int64_t> y = x;
    for (int j = 1; j < h; ++j) {
        if (y[(size_t)j] == 0) continue;
        const int64_t p = y[0], q = y[(size_t)j];
        int64_t a = 0, b = 0;
        const int64_t g = ext_gcd(p, q, a, b);
        const int64_t pg = p / g, qg = q / g;
        y[0] = g;
        y[(size_t)j] = 0;
        for (int t = 0; t < h; ++t) {
            const int64_t r0 = Cinv[0][(size_t)t], rj = Cinv[(size_t)j][(size_t)t];
            Cinv[0][(size_t)t] = pg * r0 + qg * rj;
            Cinv[(size_t)j][(size_t)t] = -b * r0 + a * rj;
        }
    }
    if (y[0] < 0)
        for (int t = 0; t < h; ++t) Cinv[0][(size_t)t] = -Cinv[0][(size_t)t];
    return Cinv;
}

// Replace rows [k0, k0+h) of the basis (and of U) by H times the old rows, and
// bring the cached Gram along without recomputing dot products.
static void apply_block_transform(Red& red, int k0,
                                  const std::vector<std::vector<int64_t>>& H) {
    const int h = (int)H.size();
    const int d = red.d;
    const int n = red.n;

    {
        std::vector<int64_t> nb((size_t)h * d, 0);
        for (int a = 0; a < h; ++a) {
            int64_t* dst = nb.data() + (size_t)a * d;
            for (int c = 0; c < h; ++c) {
                const int64_t coef = H[(size_t)a][(size_t)c];
                if (coef == 0) continue;
                const int64_t* src = red.b_row(k0 + c);
                for (int k = 0; k < d; ++k)
                    if (axpy_overflow(dst[(size_t)k], coef, src[k])) throw ReduceOverflow{};
            }
        }
        for (int a = 0; a < h; ++a)
            std::copy_n(nb.data() + (size_t)a * d, d, red.b_row(k0 + a));
    }

    {
        std::vector<std::vector<real_t>> oldblk((size_t)h, std::vector<real_t>((size_t)n));
        for (int a = 0; a < h; ++a)
            for (int j = 0; j < n; ++j) oldblk[(size_t)a][(size_t)j] = red.Gx(k0 + a, j);

        std::vector<std::vector<real_t>> T((size_t)h, std::vector<real_t>((size_t)n, 0.0));
        for (int a = 0; a < h; ++a)
            for (int c = 0; c < h; ++c) {
                const real_t hac = (real_t)H[(size_t)a][(size_t)c];
                if (hac == 0) continue;
                const std::vector<real_t>& ob = oldblk[(size_t)c];
                std::vector<real_t>& ta = T[(size_t)a];
                for (int j = 0; j < n; ++j) ta[(size_t)j] += hac * ob[(size_t)j];
            }

        std::vector<std::vector<real_t>> bb((size_t)h, std::vector<real_t>((size_t)h, 0.0));
        for (int a = 0; a < h; ++a)
            for (int b = 0; b < h; ++b) {
                real_t s = 0;
                for (int e = 0; e < h; ++e)
                    s += (real_t)H[(size_t)b][(size_t)e] * T[(size_t)a][(size_t)(k0 + e)];
                bb[(size_t)a][(size_t)b] = s;
            }

        for (int a = 0; a < h; ++a)
            for (int j = 0; j < n; ++j) {
                if (j >= k0 && j < k0 + h) continue;
                red.Gx(k0 + a, j) = T[(size_t)a][(size_t)j];
                red.Gx(j, k0 + a) = T[(size_t)a][(size_t)j];
            }
        for (int a = 0; a < h; ++a)
            for (int b = 0; b < h; ++b) red.Gx(k0 + a, k0 + b) = bb[(size_t)a][(size_t)b];
    }

    if (red.track_u && red.u_valid) {
        std::vector<int64_t> nu((size_t)h * n, 0);
        bool ok = true;
        for (int a = 0; a < h && ok; ++a) {
            int64_t* dst = nu.data() + (size_t)a * n;
            for (int c = 0; c < h && ok; ++c) {
                const int64_t coef = H[(size_t)a][(size_t)c];
                if (coef == 0) continue;
                const int64_t* src = red.u_row(k0 + c);
                for (int k = 0; k < n; ++k)
                    if (axpy_overflow(dst[(size_t)k], coef, src[k])) { ok = false; break; }
            }
        }
        if (ok) {
            for (int a = 0; a < h; ++a)
                std::copy_n(nu.data() + (size_t)a * n, n, red.u_row(k0 + a));
        } else {
            red.u_valid = false;
        }
    }
}

// Apply a full-rank unimodular transform to the basis, rebuilding the Gram from
// the integer basis afterwards. apply_block_transform propagates the Gram through
// the transform in floating point, which is accurate for the small coefficient
// vectors an enumeration produces inside a block, but a full-rank transform with
// arbitrary coefficients loses too many bits that way and leaves the
// Gram-Schmidt data (and every radius derived from it) corrupted.
static void apply_full_transform(Red& red,
                                 const std::vector<std::vector<int64_t>>& H) {
    const int n = red.n, d = red.d;
    std::vector<int64_t> nb((size_t)n * d, 0);
    for (int a = 0; a < n; ++a) {
        int64_t* dst = nb.data() + (size_t)a * d;
        for (int c = 0; c < n; ++c) {
            const int64_t coef = H[(size_t)a][(size_t)c];
            if (coef == 0) continue;
            const int64_t* src = red.b_row(c);
            for (int k = 0; k < d; ++k)
                if (axpy_overflow(dst[(size_t)k], coef, src[k])) throw ReduceOverflow{};
        }
    }
    std::vector<int64_t> nu;
    if (red.track_u && red.u_valid) {
        nu.assign((size_t)n * n, 0);
        bool ok = true;
        for (int a = 0; a < n && ok; ++a) {
            int64_t* dst = nu.data() + (size_t)a * n;
            for (int c = 0; c < n && ok; ++c) {
                const int64_t coef = H[(size_t)a][(size_t)c];
                if (coef == 0) continue;
                const int64_t* src = red.u_row(c);
                for (int k = 0; k < n; ++k)
                    if (axpy_overflow(dst[(size_t)k], coef, src[k])) { ok = false; break; }
            }
        }
        if (!ok) { red.u_valid = false; nu.clear(); }
    }
    red.B = std::move(nb);
    if (!nu.empty()) red.U = std::move(nu);
    red.build_gram();
    red.compute_gso();
}

// Insert a shorter block combination via a unimodular block transform, then
// re-reduce the affected suffix. Rolls back on overflow or on a blown step cap.
static bool try_insert_block_vector(Red& red, int kappa, int h,
                                    std::vector<int64_t> coeff) {
    if (should_stop()) return false;
    if (!coeff_make_primitive(coeff)) return false;
    const long long cap = 20000LL + 2000LL * (long long)h + 20LL * (long long)red.n;
    const std::vector<std::vector<int64_t>> H = complete_unimodular(coeff);
    red.save_state();
    try {
        if (h == red.n && kappa == 0) {
            apply_full_transform(red, H);
        } else {
            apply_block_transform(red, kappa, H);
            red.update_gso_after_block(kappa, h);
        }
        red.lll(std::max(1, kappa), cap);
        return true;
    } catch (const ReduceOverflow&) {
        red.restore_state();
        return false;
    } catch (const ReduceAbort&) {
        red.restore_state();
        return false;
    }
}

// Pruned Schnorr-Euchner enumeration of the projected block [k0, k0+h) for a
// combination shorter than the block's leading Gram-Schmidt norm.
struct Enumerator {
    int h = 0;
    real_t prune = 0.0;
    real_t bound = 0;
    real_t deg_eps = 0.0;
    std::vector<real_t> rr;
    std::vector<real_t> mu_row;   // mu_row[k*h + j] = mu(k0+k, k0+j), j < k
    std::vector<real_t> tsum;     // running center partial sums, one row per level
    std::vector<int64_t> x, best;
    bool found = false;
    bool aborted = false;
    long long nodes = 0;
    long long node_limit = 0;

    static constexpr real_t k_degenerate_rr_rel = 1e-12;

    real_t prune_bound(int cnt) const {
        if (prune <= 0.0) return bound;
        const real_t frac = 1.0 - prune * (1.0 - (real_t)cnt / (real_t)h);
        return bound * frac;
    }

    void dfs(int k, real_t partdist) {
        if (aborted) return;
        ++nodes;
        if (node_limit > 0 && nodes >= node_limit) { aborted = true; return; }
        if ((nodes & 0xFFFF) == 0 && should_stop()) {
            aborted = true;
            return;
        }
        const int cnt = h - 1 - k;
        if (partdist >= prune_bound(cnt)) return;
        if (k < 0) {
            bool nz = false;
            for (int i = 0; i < h; ++i) if (x[(size_t)i] != 0) { nz = true; break; }
            if (nz && partdist < bound) { bound = partdist; best = x; found = true; }
            return;
        }
        const real_t center = -tsum[(size_t)k * h + k];
        const long long base = std::llrint(center);
        const real_t child_cap = prune_bound(cnt + 1);
        const real_t* mrow = mu_row.data() + (size_t)k * h;
        real_t* child = (k > 0) ? tsum.data() + (size_t)(k - 1) * h : nullptr;
        const real_t* self = tsum.data() + (size_t)k * h;
        for (int radius = 0;; ++radius) {
            bool any = false;
            const int signs = (radius == 0) ? 1 : 2;
            for (int s = 0; s < signs; ++s) {
                const long long xk = base + (s == 0 ? radius : -radius);
                const real_t ck = (real_t)xk - center;
                const real_t nd = partdist + ck * ck * rr[(size_t)k];
                if (nd < child_cap) {
                    any = true;
                    x[(size_t)k] = (int64_t)xk;
                    if (child) {
                        const real_t v = (real_t)xk;
                        for (int j = 0; j < k; ++j)
                            child[j] = v * mrow[j] + self[j];
                    }
                    dfs(k - 1, nd);
                }
            }
            if (radius > 0 && !any) break;
            // A Gram-Schmidt norm that has drifted to a spurious near-zero would
            // make this radius scan explode, and it does not change the projected
            // distance anyway, so only its nearest integer matters.
            if (rr[(size_t)k] <= deg_eps) break;
            if (radius > (1 << 20)) break;
        }
    }

    bool run(const Red& red, int k0, int block, real_t prune_amt,
             long long node_limit_, std::vector<int64_t>& coeff) {
        h = block;
        prune = prune_amt;
        node_limit = node_limit_;
        rr.assign((size_t)h, 0.0);
        for (int i = 0; i < h; ++i) rr[(size_t)i] = red.r[(size_t)(k0 + i)];
        mu_row.assign((size_t)h * h, 0.0);
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < i; ++j)
                mu_row[(size_t)i * h + j] = red.M(k0 + i, k0 + j);
        tsum.assign((size_t)h * h, 0.0);
        bound = rr[0];
        deg_eps = (rr[0] > 0.0) ? rr[0] * k_degenerate_rr_rel : 0.0;
        x.assign((size_t)h, 0);
        best.clear();
        found = false;
        aborted = false;
        nodes = 0;
        dfs(h - 1, 0.0);
        if (found && bound < rr[0] * (1.0 - 1e-9)) {
            coeff = best;
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Orbit oracle
// ---------------------------------------------------------------------------
// A discovered symmetry is a coordinate matrix C, meaningful only relative to the
// basis it was discovered in. The underlying map is an ambient one, so it
// survives every later change of basis; what has to be remembered is the frame
// that expresses it. The oracle keeps a copy of that frame and turns the symmetry
// into a map on ambient lattice vectors:
//
//     v  ->  coordinates in the frame  ->  times C  ->  back to ambient.
//
// That leaves the working basis free to change (insertions, LLL, perturbations)
// without invalidating the symmetry, at the cost of one coordinate recovery per
// image. Every image is an exact lattice vector: C is unimodular, so the whole
// path stays inside the lattice however large the residual of the symmetry is.
struct OrbitOracle {
    bool valid = false;
    SymTarget sym;
    Red frame;
    // When the symmetry is an integer orthogonal map of the ambient space and the
    // lattice has been verified stable under it, images cost one pass over the
    // coordinates and are exactly length preserving. This path needs no frame and
    // cannot fail.
    bool ambient_valid = false;
    AmbientSym ambient;

    int order() const { return ambient_valid ? ambient.order : sym.order; }
    bool exact() const { return ambient_valid; }

    // One symmetry step on an ambient lattice vector. Returns false if v is not
    // recognized as a lattice vector of the frame, or the image overflows.
    bool step(const std::vector<int64_t>& v, std::vector<int64_t>& out) const {
        if (ambient_valid) {
            ambient.apply(v, out);
            return true;
        }
        if (!valid) return false;
        std::vector<int64_t> x;
        if (!frame.coords_of(v.data(), x)) return false;
        std::vector<int64_t> y;
        sym.apply_right(x, y);
        out.assign((size_t)frame.d, 0);
        for (int i = 0; i < frame.n; ++i) {
            const int64_t yi = y[(size_t)i];
            if (yi == 0) continue;
            const int64_t* bi = frame.b_row(i);
            for (int k = 0; k < frame.d; ++k)
                if (axpy_overflow(out[(size_t)k], yi, bi[k])) return false;
        }
        return true;
    }
};

// Is the lattice stable under the ambient map J? Exact test: J b_i has to be a
// lattice vector for every basis vector b_i, and coords_of verifies membership in
// exact integer arithmetic. J is a signed permutation, so it preserves lengths,
// and J L subset L together with equal covolume gives J L = L.
static bool ambient_stabilizes(const Red& red, const AmbientSym& J) {
    if (J.d != red.d) return false;
    std::vector<int64_t> row((size_t)red.d), img, x;
    for (int i = 0; i < red.n; ++i) {
        std::copy_n(red.b_row(i), red.d, row.begin());
        J.apply(row, img);
        if (!red.coords_of(img.data(), x)) return false;
    }
    return true;
}

// Module descent
// ---------------------------------------------------------------------------
// A symmetry that is an exact automorphism is an isometry, so every orbit image
// v C^k has exactly the norm of v. Orbits alone therefore never produce a
// shorter vector, and the gain has to come from integer combinations inside the
// module the orbit generates.
//
// Those combinations are a lattice in their own right: M = <v, vC, vC^2, ...> is
// a C-invariant sublattice of L. Every vector of M is a vector of L, so anything
// short found in M lifts back directly, but M has the rank of the orbit rather
// than the rank of L. On a negacyclic module lattice of rank 2n the orbit spans
// rank n, which halves the rank of the search space, and reduction cost grows
// steeply with rank. That is the one use of the symmetry the isometry property
// does not defeat.
//
// The images are collected greedily by rank so the descent basis is independent
// and the Gram-Schmidt stays well conditioned. For a negacyclic C the n images
// are generically independent, so this keeps the whole module rather than a
// proper sublattice of it; where an image is dependent, dropping it only shrinks
// the search space and never makes a returned vector invalid.
static int bkz_tour(Red& red, int beta, real_t prune, long long node_limit);

struct DescentReport {
    int rank = 0;               // rank of the orbit sublattice that was searched
    real_t found_n2 = 0;        // best norm^2 inside it, 0 when nothing was built
    bool inserted = false;      // whether it improved the working basis
};

// Collect orbit images that increase the rank, drawing from every seed in turn.
//
// Interleaving the seeds is what makes the descent worth doing. The orbit of a
// single v generates the principal ideal (v), and every element of it is v times
// a ring element, so its norm is N(v) times an integer: nothing in (v) is
// generically shorter than v, and descending into it cannot win. Taking images
// from several short seeds instead builds a sublattice of (v) + (w) + ..., which
// has the same rank but a smaller determinant, and that is where shorter vectors
// live. Filling the rank from one orbit before touching the next would throw the
// later seeds away, hence the round-robin.
//
// Rank is tracked with a running Gram-Schmidt in double: a candidate is kept when
// its residual after projecting out the span so far is a non-trivial fraction of
// its own norm, a conservative test that errs towards dropping a vector rather
// than admitting a near-dependent one and wrecking the conditioning.
static void collect_independent_orbit(const OrbitOracle& oracle,
                                      const std::vector<std::vector<int64_t>>& seeds,
                                      int depth, int d,
                                      std::vector<std::vector<int64_t>>& rows) {
    rows.clear();
    std::vector<std::vector<real_t>> basis;   // orthogonalised span so far
    std::vector<real_t> basis_n2;

    auto try_keep = [&](const std::vector<int64_t>& v) {
        std::vector<real_t> w((size_t)d);
        real_t n2 = 0;
        for (int k = 0; k < d; ++k) {
            w[(size_t)k] = (real_t)v[(size_t)k];
            n2 += w[(size_t)k] * w[(size_t)k];
        }
        if (n2 <= 0) return;
        for (size_t b = 0; b < basis.size(); ++b) {
            real_t dot = 0;
            for (int k = 0; k < d; ++k) dot += w[(size_t)k] * basis[b][(size_t)k];
            const real_t f = dot / basis_n2[b];
            for (int k = 0; k < d; ++k) w[(size_t)k] -= f * basis[b][(size_t)k];
        }
        real_t res = 0;
        for (int k = 0; k < d; ++k) res += w[(size_t)k] * w[(size_t)k];
        if (res <= n2 * 1e-12) return;          // dependent on what we already have
        basis_n2.push_back(res);
        basis.push_back(std::move(w));
        rows.push_back(v);
    };

    std::vector<std::vector<int64_t>> cur = seeds;
    std::vector<bool> live(seeds.size(), true);
    for (size_t s = 0; s < seeds.size(); ++s) try_keep(seeds[s]);

    std::vector<int64_t> nxt;
    for (int k = 0; k < depth && (int)rows.size() < d; ++k) {
        bool any_live = false;
        for (size_t s = 0; s < cur.size() && (int)rows.size() < d; ++s) {
            if (!live[s]) continue;
            if (should_stop()) return;
            if (!oracle.step(cur[s], nxt)) { live[s] = false; continue; }
            cur[s] = nxt;
            bool zero = true;
            for (int64_t e : cur[s]) if (e != 0) { zero = false; break; }
            if (zero) { live[s] = false; continue; }
            any_live = true;
            try_keep(cur[s]);
        }
        if (!any_live) break;
    }
}

// Reduce a set of generators of a sublattice of red and lift the best vector it
// contains back into red. The generators must be exact lattice vectors; the
// sublattice they span is searched in its own frame, where the rank can be far
// below that of red and a much larger block size is therefore affordable.
static DescentReport reduce_and_lift(Red& red,
                                     const std::vector<std::vector<int64_t>>& rows,
                                     int beta, real_t prune, long long node_limit) {
    DescentReport rep;
    if (rows.size() < 2) return rep;
    rep.rank = (int)rows.size();

    Lattice sub;
    sub.m = (int)rows.size();
    sub.d = red.d;
    sub.data.assign((size_t)sub.m * sub.d, 0);
    for (int i = 0; i < sub.m; ++i)
        std::copy_n(rows[(size_t)i].data(), sub.d, sub.row(i));

    Red dr;
    std::vector<int64_t> best_vec;
    try {
        dr.init(sub, false);
        dr.delta = red.delta;
        dr.lll(1);
        const int bmax = std::min(beta, dr.n);
        for (int b = 2; b <= bmax && !should_stop(); ++b)
            bkz_tour(dr, b, prune, node_limit);
        int idx = 0;
        rep.found_n2 = dr.shortest_norm2(&idx);
        best_vec.assign(dr.b_row(idx), dr.b_row(idx) + dr.d);
    } catch (const ReduceOverflow&) {
        return rep;
    } catch (const ReduceAbort&) {
        return rep;
    }

    if (rep.found_n2 <= 0 || rep.found_n2 >= red.shortest_norm2()) return rep;
    std::vector<int64_t> x;
    if (!red.coords_of(best_vec.data(), x)) return rep;
    rep.inserted = try_insert_block_vector(red, 0, red.n, x);
    return rep;
}

// Isotypic descent
// ---------------------------------------------------------------------------
// A module of rank two has no useful proper submodules: a single orbit spans a
// principal ideal, whose minimum is its own generator, and two orbits already
// span everything. The room comes instead from the ring being decomposable.
//
// For the negacyclic ring Z[x]/(x^n+1), whenever k divides n with n/k odd,
// x^k + 1 divides x^n + 1 with cofactor g(x) = sum_j (-1)^j x^{kj}. The image
// g(C)L is then annihilated by C^k + 1, so it is a module over Z[x]/(x^k+1) and
// its rank is (rank of L) * k / n. For rank 80 with n = 40 and k = 8 that is a
// rank 16 sublattice of an 80-dimensional lattice, small enough to search to the
// bottom, and every vector in it is a vector of L by construction.
//
// Applying g(C) row by row costs n/k orbit steps per row and no arithmetic
// beyond addition, since C is an isometry of the ambient coordinates.
// Replace a set of generators by a basis of the lattice they generate.
//
// Picking a maximal independent subset instead would be much cheaper but wrong
// for this purpose: it yields some sublattice of finite index rather than the
// lattice itself, and that index is exactly the density the descent is after.
// Hermite reduction gives the right answer but its entries grow out of int64
// range across this many columns.
//
// So the generators are LLL reduced with a scaled copy of the identity attached:
// row i becomes [c*g_i | e_i], which is full rank even when the g_i are not. A
// reduced row whose leading block vanishes is a dependency among the generators
// and carries no lattice vector; the rest have leading blocks that are exactly c
// times a basis of the generated lattice. Scaling by c is what makes LLL treat
// clearing the leading block as the priority, and LLL keeps every entry small on
// the way, which is the property Hermite reduction lacks.
static bool basis_from_generators(std::vector<std::vector<int64_t>>& rows, int d) {
    const int m = (int)rows.size();
    if (m == 0) return false;

    int64_t maxabs = 1;
    for (const std::vector<int64_t>& row : rows)
        for (int64_t e : row)
            maxabs = std::max(maxabs, (int64_t)std::llabs((long long)e));
    const int64_t c = std::max<int64_t>(1, std::min<int64_t>(1 << 16,
                                                             1000000000LL / maxabs));
    if (c < 2) return false;

    Lattice aug;
    aug.m = m;
    aug.d = d + m;
    aug.data.assign((size_t)aug.m * aug.d, 0);
    for (int i = 0; i < m; ++i) {
        int64_t* dst = aug.row(i);
        for (int k = 0; k < d; ++k) dst[k] = c * rows[(size_t)i][(size_t)k];
        dst[d + i] = 1;
    }

    Red aux;
    try {
        aux.init(aug, false);
        aux.lll(1);
    } catch (const ReduceOverflow&) {
        return false;
    } catch (const ReduceAbort&) {
        return false;
    }

    std::vector<std::vector<int64_t>> out;
    for (int i = 0; i < aux.n; ++i) {
        const int64_t* src = aux.b_row(i);
        bool zero = true;
        for (int k = 0; k < d; ++k) if (src[k] != 0) { zero = false; break; }
        if (zero) continue;                 // a dependency, not a lattice vector
        std::vector<int64_t> v((size_t)d);
        for (int k = 0; k < d; ++k) {
            if (src[k] % c != 0) return false;   // separation failed, do not guess
            v[(size_t)k] = src[k] / c;
        }
        out.push_back(std::move(v));
    }
    if (out.size() < 2) return false;
    rows.swap(out);
    return true;
}

static bool apply_negacyclic_cofactor(const OrbitOracle& oracle,
                                      const std::vector<int64_t>& v, int k,
                                      int terms, std::vector<int64_t>& out) {
    out = v;
    std::vector<int64_t> cur = v, nxt;
    for (int j = 1; j < terms; ++j) {
        for (int s = 0; s < k; ++s) {
            if (!oracle.step(cur, nxt)) return false;
            cur = nxt;
        }
        const int64_t sign = ((j % 2) == 0) ? 1 : -1;
        for (size_t t = 0; t < out.size(); ++t)
            if (axpy_overflow(out[t], sign, cur[t])) return false;
    }
    return true;
}

// The smallest k dividing n with n/k odd and k < n, which gives the lowest rank
// isotypic component available. Returns 0 when the ring does not split this way.
static int negacyclic_split_degree(int n) {
    for (int k = 1; k < n; ++k) {
        if (n % k != 0) continue;
        if (((n / k) % 2) == 1) return k;
    }
    return 0;
}

static DescentReport isotypic_descent(Red& red, const OrbitOracle& oracle, int n,
                                      int beta, real_t prune,
                                      long long node_limit) {
    DescentReport rep;
    const int k = negacyclic_split_degree(n);
    if (k <= 0) return rep;
    const int terms = n / k;

    // Every row's image is kept, dependent or not: the whole point is to obtain
    // g(C)L itself rather than a sparse sublattice of it, and it is the Hermite
    // step that turns the generating set into a basis.
    std::vector<std::vector<int64_t>> rows;
    std::vector<int64_t> img;
    for (int i = 0; i < red.n && !should_stop(); ++i) {
        const std::vector<int64_t> row(red.b_row(i), red.b_row(i) + red.d);
        if (!apply_negacyclic_cofactor(oracle, row, k, terms, img)) continue;
        bool zero = true;
        for (int64_t e : img) if (e != 0) { zero = false; break; }
        if (!zero) rows.push_back(img);
    }
    if (!basis_from_generators(rows, red.d)) return rep;
    return reduce_and_lift(red, rows, beta, prune, node_limit);
}

// Build the orbit sublattice of seed_vec, reduce it in its own frame, and lift
// the best vector found back into red. The descent lattice has its own Red, so
// the block size there is free to be much larger than the one used on the full
// rank without costing more.
static DescentReport module_descent(Red& red, const OrbitOracle& oracle,
                                    const std::vector<std::vector<int64_t>>& seeds,
                                    int depth, int beta, real_t prune,
                                    long long node_limit) {
    DescentReport rep;
    if (!oracle.valid || depth <= 0 || seeds.empty()) return rep;

    std::vector<std::vector<int64_t>> rows;
    collect_independent_orbit(oracle, seeds, depth, red.d, rows);
    return reduce_and_lift(red, rows, beta, prune, node_limit);
}

// Random unimodular shears followed by a bounded LLL, used to break out of a
// stalled round. Strength scales the number of shears. Rolls back on overflow.
static void perturb_basis(Red& red, double strength, std::mt19937_64& rng) {
    const int n = red.n;
    const int moves = std::max(2, (int)(strength * (double)n));
    red.save_state();
    try {
        for (int t = 0; t < moves; ++t) {
            const int i = (int)(rng() % (unsigned)n);
            int j = (int)(rng() % (unsigned)n);
            if (i == j) j = (j + 1) % n;
            const int64_t s = ((rng() & 1u) ? 1 : -1) * (int64_t)(1 + rng() % 2);
            red.axpy(j, i, s);
        }
        red.compute_gso();
        red.lll(1, 200000LL + 1000LL * (long long)n);
    } catch (const ReduceOverflow&) {
        red.restore_state();
    } catch (const ReduceAbort&) {
        red.restore_state();
    }
}

// One BKZ tour at the given block size: visit every window, enumerate it, and
// insert any shorter projected vector found. Returns the number of insertions.
static int bkz_tour(Red& red, int beta, real_t prune, long long node_limit) {
    int inserted = 0;
    Enumerator en;
    std::vector<int64_t> coeff;
    for (int kappa = 0; kappa + 1 < red.n; ++kappa) {
        if (should_stop()) break;
        const int h = std::min(beta, red.n - kappa);
        if (h < 2) continue;
        if (en.run(red, kappa, h, prune, node_limit, coeff)) {
            if (try_insert_block_vector(red, kappa, h, coeff)) ++inserted;
        }
    }
    return inserted;
}

// ---------------------------------------------------------------------------
// Orbit-enriched sieve over the whole lattice
// ---------------------------------------------------------------------------
// A list-based Gauss sieve on ambient lattice vectors.
//
// This deliberately does not use the symmetry. Feeding orbit images into the list
// and reducing candidates against whole orbits were both tried and measured: an
// exact automorphism is an isometry, so an image has precisely its parent's norm
// and adds no reach to the list, while costing an orbit walk per pair. The
// symmetry earns its place in the descent stage instead, where combinations
// inside an invariant sublattice can be shorter than anything the orbit contains.
//
// The sieve itself is off unless asked for: on the instances measured here a call
// costs hundreds of BKZ tours and the tours find records faster.
struct OrbitSieve {
    struct Vec {
        std::vector<int64_t> v;
        real_t n2 = 0;
    };

    int d = 0;
    std::vector<Vec> list;
    std::vector<Vec> queue;

    static real_t norm2_of(const std::vector<int64_t>& v) {
        real_t s = 0;
        for (int64_t e : v) s += (real_t)e * (real_t)e;
        return s;
    }
    static real_t dot_of(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        real_t s = 0;
        for (size_t k = 0; k < a.size(); ++k) s += (real_t)a[k] * (real_t)b[k];
        return s;
    }
    static bool is_zero(const Vec& a) {
        for (int64_t e : a.v) if (e != 0) return false;
        return true;
    }
    static Vec wrap(std::vector<int64_t> v) {
        Vec out;
        out.n2 = norm2_of(v);
        out.v = std::move(v);
        return out;
    }

    // Ambient vector of a coordinate vector over the given basis.
    static bool combine(const Red& red, const std::vector<int64_t>& x,
                        std::vector<int64_t>& out) {
        out.assign((size_t)red.d, 0);
        for (int i = 0; i < red.n; ++i) {
            const int64_t xi = x[(size_t)i];
            if (xi == 0) continue;
            const int64_t* bi = red.b_row(i);
            for (int k = 0; k < red.d; ++k)
                if (axpy_overflow(out[(size_t)k], xi, bi[k])) return false;
        }
        return true;
    }

    // a -= q b with q the nearest integer to <a,b>/|b|^2. Returns true when a got
    // strictly shorter.
    bool reduce_by(Vec& a, const Vec& b) const {
        if (b.n2 <= 0) return false;
        const real_t dot = dot_of(a.v, b.v);
        const long long q = std::llrint(dot / b.n2);
        if (q == 0) return false;
        if (std::llabs(q) > (1LL << 30)) return false;
        const real_t m = (real_t)q;
        const real_t nn = a.n2 - 2.0 * m * dot + m * m * b.n2;
        if (nn >= a.n2 * (1.0 - 1e-12)) return false;
        for (int k = 0; k < d; ++k) {
            const real_t t = (real_t)a.v[(size_t)k] - m * (real_t)b.v[(size_t)k];
            if (t > 9.0e17 || t < -9.0e17) return false;
        }
        for (int k = 0; k < d; ++k) a.v[(size_t)k] -= (int64_t)q * b.v[(size_t)k];
        a.n2 = norm2_of(a.v);
        return true;
    }

    void run(const Red& red, int pool_cap, int iters, real_t target_n2,
             std::mt19937_64& rng, std::vector<std::vector<int64_t>>& out,
             size_t out_max) {
        d = red.d;
        list.clear();
        queue.clear();
        list.reserve((size_t)pool_cap);
        const size_t queue_cap = (size_t)pool_cap * 4;

        std::vector<Vec> hits;
        auto consider = [&](const Vec& v) {
            if (v.n2 <= 0 || v.n2 >= target_n2 * (1.0 - 1e-12)) return;
            hits.push_back(v);
        };

        for (int i = 0; i < red.n; ++i)
            queue.push_back(wrap(std::vector<int64_t>(red.b_row(i), red.b_row(i) + red.d)));

        auto sample_one = [&](Vec& out_v) -> bool {
            std::vector<int64_t> x((size_t)red.n, 0);
            const int nz = 2 + (int)(rng() % (unsigned)std::max(1, red.n / 4));
            for (int t = 0; t < nz; ++t) {
                const int idx = (int)(rng() % (unsigned)red.n);
                x[(size_t)idx] += (rng() & 1u) ? 1 : -1;
            }
            std::vector<int64_t> amb;
            if (!combine(red, x, amb)) return false;
            out_v = wrap(std::move(amb));
            return true;
        };

        long long budget = (long long)pool_cap * (long long)iters + (long long)queue.size();
        while (budget-- > 0) {
            if (should_stop()) break;
            Vec v;
            if (!queue.empty()) {
                v = std::move(queue.back());
                queue.pop_back();
            } else if (!sample_one(v)) {
                continue;
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (const Vec& w : list)
                    if (reduce_by(v, w)) changed = true;
                if (should_stop()) break;
            }
            if (is_zero(v)) continue;   // collision
            consider(v);

            // Pull out list members the newcomer can shorten and re-queue them,
            // which keeps the list pairwise reduced.
            for (size_t i = 0; i < list.size();) {
                if (reduce_by(list[i], v)) {
                    Vec moved = std::move(list[i]);
                    list[i] = std::move(list.back());
                    list.pop_back();
                    if (!is_zero(moved) && queue.size() < queue_cap) {
                        consider(moved);
                        queue.push_back(std::move(moved));
                    }
                } else {
                    ++i;
                }
            }

            if ((int)list.size() < pool_cap) {
                list.push_back(std::move(v));
            } else {
                size_t worst = 0;
                for (size_t i = 1; i < list.size(); ++i)
                    if (list[i].n2 > list[worst].n2) worst = i;
                if (v.n2 < list[worst].n2) list[worst] = std::move(v);
            }
        }

        std::sort(hits.begin(), hits.end(),
                  [](const Vec& a, const Vec& b) { return a.n2 < b.n2; });
        out.clear();
        for (const Vec& hv : hits) {
            if (out.size() >= out_max) break;
            out.push_back(hv.v);
        }
    }
};

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct Params {
    std::string lattice;
    std::string out = "reduced.csv";
    std::string transform_out = "U.csv";
    std::string shortest_out = "shortest.csv";
    bool no_transform = false;

    std::string mode = "gauss";
    int cyclo_order = 8;
    int perm_cycle = 4;
    int sym_block = 0;
    double sym_lambda = 0.25;
    double sym_seconds = 3.0;
    int sym_threads = 0;
    int orbit_depth = 0;        // 0 = full order of C, capped
    int descent_block = 24;     // BKZ block size inside the orbit sublattice
    int descent_seeds = 4;      // short rows whose orbits generate that sublattice

    int block = 20;
    int block_start = 2;
    double delta = 0.99;
    double prune = 0.0;
    long long enum_node_limit = 2000000;

    int pool = 256;
    int sieve_iters = 8;
    bool use_sieve = false;
    int sieve_after = 8;        // rounds without a record before the sieve runs
    int insert_max = 8;

    int rounds = 0;
    double max_seconds = 60.0;
    int stall_rounds = 2;
    double perturb_strength = 0.5;
    bool no_init_lll = false;
    uint64_t seed = 0;
    uint64_t modulus = 0;
    int64_t saturate = 0;
    bool verbose = false;
};

// The single global best: the reduced basis plus the shortest vector ever seen,
// tracked separately so it can never get longer.
struct BestState {
    Lattice basis;
    Matrix transform;
    bool have_transform = false;
    std::vector<int64_t> shortest;
    real_t shortest_n2 = 0.0;
};

static void snapshot_best(const Red& red, BestState& best) {
    best.basis.m = red.n;
    best.basis.d = red.d;
    best.basis.data = red.B;
    if (red.track_u && red.u_valid) {
        best.transform = Matrix((size_t)red.n);
        best.transform.data = red.U;
        best.have_transform = true;
    }
}

static void note_shortest(const Red& red, BestState& best) {
    int idx = 0;
    const real_t n2 = red.shortest_norm2(&idx);
    if (best.shortest.empty() || n2 < best.shortest_n2) {
        best.shortest_n2 = n2;
        best.shortest.assign(red.b_row(idx), red.b_row(idx) + red.d);
    }
}

static void write_outputs(const Params& p, const BestState& best) {
    write_lattice_csv(best.basis, p.out);
    if (!p.no_transform && best.have_transform)
        write_matrix_csv(best.transform, p.transform_out);
    if (!best.shortest.empty())
        write_flat_csv(best.shortest, 1, (int)best.shortest.size(), p.shortest_out);
}

// Build the discovery objective for the current basis, with both terms
// normalized by their initial values so --sym-lambda 1 gives the basis-quality
// term the same starting weight as the symmetry residual.
static SymObjective make_objective(const Red& red, const SymTarget& target,
                                   double sym_lambda) {
    SymObjective start;
    start.T = target;
    start.P = red.gram_matrix();
    start.Xt = Matrix(start.P.n);
    start.Xt.fill_identity();
    start.sym_w = 1.0L;
    start.red_w = 0.0L;
    start.rebuild();
    const long double s0 = start.sym_raw > 0.0L ? start.sym_raw : 1.0L;
    const long double r0 = start.red_raw > 0.0L ? start.red_raw : 1.0L;
    start.sym_w = 1.0L / s0;
    start.red_w = (long double)sym_lambda / r0;
    start.rebuild();
    return start;
}

// Run the discovery annealer on the current basis Gram and return the best
// objective state found. The engine runs quiet: xsym prints its own summary.
static SymObjective discover_symmetry(const Red& red, const SymTarget& target,
                                      const Params& p, double seconds,
                                      uint64_t seed) {
    SymObjective start = make_objective(red, target, p.sym_lambda);
    // Nothing to search for: the current basis already carries the symmetry.
    if (start.solved()) return start;
    canneal::EngineParams ep;
    ep.threads = p.sym_threads > 0 ? p.sym_threads : canneal::physical_core_count();
    ep.max_seconds = seconds;
    ep.seed = seed;
    ep.quiet = true;
    ep.verbose = p.verbose;
    return canneal::run_annealer(start, ep);
}

// Apply a unimodular transform X to the basis: the new basis is X^T B, so the
// new Gram is X^T G X. Rebuilds the Gram and the Gram-Schmidt data.
static bool apply_transform(Red& red, const Matrix& X) {
    const int n = red.n, d = red.d;
    std::vector<int64_t> nb((size_t)n * d, 0);
    for (int i = 0; i < n; ++i) {
        int64_t* dst = nb.data() + (size_t)i * d;
        for (int a = 0; a < n; ++a) {
            const int64_t xai = X.at(a, i);   // (X^T)_{i,a}
            if (xai == 0) continue;
            const int64_t* src = red.b_row(a);
            for (int k = 0; k < d; ++k)
                if (axpy_overflow(dst[(size_t)k], xai, src[k])) return false;
        }
    }
    std::vector<int64_t> nu;
    if (red.track_u && red.u_valid) {
        nu.assign((size_t)n * n, 0);
        for (int i = 0; i < n; ++i) {
            int64_t* dst = nu.data() + (size_t)i * n;
            for (int a = 0; a < n; ++a) {
                const int64_t xai = X.at(a, i);
                if (xai == 0) continue;
                const int64_t* src = red.u_row(a);
                for (int k = 0; k < n; ++k)
                    if (axpy_overflow(dst[(size_t)k], xai, src[k])) return false;
            }
        }
    }
    red.B = std::move(nb);
    if (!nu.empty()) red.U = std::move(nu);
    red.build_gram();
    red.compute_gso();
    return true;
}

int main(int argc, char** argv) {
    install_signal_handlers();

    Params p;
    CLI::App app{ "xsym symmetry-directed lattice reduction and short-vector search" };
    app.add_option("-L,--lattice", p.lattice, "CSV lattice basis (rows are vectors)")
        ->required();
    app.add_option("-o,--out", p.out, "Output CSV for the reduced basis");
    app.add_option("--transform-out", p.transform_out,
                   "Output CSV for the unimodular transform U (reduced = U * L)");
    app.add_option("--shortest-out", p.shortest_out,
                   "Output CSV for the shortest vector found");
    app.add_flag("--no-transform", p.no_transform, "Do not track or write U");

    app.add_option("--mode", p.mode,
                   "Symmetry family to search for: none, gauss, eisen3, eisen6, "
                   "cyclic, negacyclic, cyclo, perm");
    app.add_option("--cyclo-order", p.cyclo_order,
                   "Order m for --mode cyclo (uses the companion of Phi_m)");
    app.add_option("--perm-cycle", p.perm_cycle, "Cycle length for --mode perm");
    app.add_option("--sym-block", p.sym_block,
                   "Block size for --mode cyclic / negacyclic (0 = one block over "
                   "the whole rank; set it to the ring degree for a module lattice)");
    app.add_option("--sym-lambda", p.sym_lambda,
                   "Weight of the basis-quality term in the discovery objective");
    app.add_option("--sym-seconds", p.sym_seconds,
                   "Discovery annealing seconds per round (0 = skip discovery)");
    app.add_option("--sym-threads", p.sym_threads,
                   "Worker threads for the discovery annealer (default: physical cores)");
    app.add_option("--orbit-depth", p.orbit_depth,
                   "Orbit powers used for enrichment (0 = the full order of C, capped)");

    app.add_option("-b,--block", p.block, "Maximum BKZ block size");
    app.add_option("--block-start", p.block_start, "Minimum BKZ block size");
    app.add_option("--delta", p.delta, "LLL delta in (0.25, 1.0)");
    app.add_option("--prune", p.prune, "Enumeration pruning in [0, 1] (0 = exact)");
    app.add_option("--enum-node-limit", p.enum_node_limit,
                   "Maximum enumeration nodes per block (0 = unlimited)");

    app.add_option("--pool", p.pool, "Orbit sieve pool size");
    app.add_option("--sieve-iters", p.sieve_iters, "Orbit sieve work multiplier");
    app.add_flag("--sieve", p.use_sieve,
                 "Enable the orbit sieve stage (costs many tours per call; pays "
                 "only when the tours have stopped finding records)");
    app.add_option("--descent-block", p.descent_block,
                   "BKZ block size used inside the orbit sublattice, which has "
                   "lower rank than the full lattice and so affords a larger one");
    app.add_option("--descent-seeds", p.descent_seeds,
                   "Short rows whose orbits generate the descent sublattice; one "
                   "seed gives a principal ideal, which cannot beat the seed");
    app.add_option("--sieve-after", p.sieve_after,
                   "Rounds without a new record before the sieve runs (0 = every round)");
    app.add_option("--insert-max", p.insert_max,
                   "Maximum sieve candidates inserted per round");

    app.add_option("--rounds", p.rounds, "Number of rounds (0 = until the time budget)");
    app.add_option("--max-seconds", p.max_seconds, "Wall-clock budget (0 = until Ctrl-C)");
    app.add_option("--stall-rounds", p.stall_rounds,
                   "Rounds without improvement before perturbing (0 = never)");
    app.add_option("--perturb", p.perturb_strength,
                   "Perturbation strength on a stall, as a fraction of the rank");
    app.add_flag("--no-init-lll", p.no_init_lll,
                 "Skip the initial LLL, so an already-aligned input keeps its basis");
    app.add_option("--seed", p.seed, "Base RNG seed (0 = random)");
    app.add_option("--modulus", p.modulus, "Reduce loaded entries modulo this value");
    app.add_option("--saturate", p.saturate, "Clamp loaded entries into [-n, n]");
    app.add_flag("--verbose", p.verbose, "Print the discovery annealer's progress");

    CLI11_PARSE(app, argc, argv);

    if (p.delta <= 0.25 || p.delta >= 1.0) {
        std::cerr << "--delta must be in (0.25, 1.0)\n";
        return 1;
    }
    if (p.prune < 0.0 || p.prune > 1.0) {
        std::cerr << "--prune must be in [0, 1]\n";
        return 1;
    }
    SymMode mode = SymMode::Gauss;
    if (!parse_sym_mode(p.mode, mode)) {
        std::cerr << "unknown --mode '" << p.mode << "'\n";
        return 1;
    }

    Lattice L;
    try {
        L = read_lattice_csv(p.lattice);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    if (p.modulus > 0) {
        reduce_entries_mod(L.data, p.modulus);
        std::cout << "[xsym] " << modulus_note(p.modulus) << "\n";
    }
    if (p.saturate > 0) {
        saturate_entries(L.data, p.saturate);
        std::cout << "[xsym] " << saturate_note(p.saturate) << "\n";
    }
    if (L.m < 2 || L.d < 2) {
        std::cerr << "the basis needs at least 2 vectors in dimension 2\n";
        return 1;
    }
    if (L.m > L.d)
        std::cout << "[xsym] warning: " << L.m << " rows in dimension " << L.d
                  << ", the rows cannot be independent\n";

    const uint64_t seed = p.seed ? p.seed : (uint64_t)std::random_device{}();
    std::mt19937 target_rng((uint32_t)seed);
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);

    Red red;
    red.delta = p.delta;
    red.init(L, !p.no_transform);

    SymTarget target;
    bool have_target = false;
    if (mode != SymMode::None && p.sym_seconds > 0.0) {
        std::string err;
        if (build_sym_target(mode, red.n, p.cyclo_order, p.perm_cycle, p.sym_block,
                             target_rng, target, err)) {
            // The exact int64 residual C P C^T - P must stay inside the range the
            // score arithmetic assumes, which bounds how dense C may be.
            if (target.row_abs_max > 64) {
                std::cout << "[xsym] target for mode " << sym_mode_name(mode)
                          << " is too dense (row weight " << target.row_abs_max
                          << "), skipping discovery\n";
            } else {
                have_target = true;
            }
        } else {
            std::cout << "[xsym] " << err << ", running without discovery\n";
        }
    }
    // Orbit depth is fixed after the oracle is settled, since the ambient path can
    // have a different order than the coordinate target.
    int orbit_depth = 0;

    const auto t0 = Clock::now();
    g_t0 = t0;
    g_max_seconds = p.max_seconds;
    BestState best;

    std::cout << "[xsym] " << red.n << " vectors in dimension " << red.d
              << ", mode=" << sym_mode_name(mode);
    if (have_target) std::cout << " order=" << target.order;
    std::cout << " seed=" << seed << "\n";

    // The symmetry the sieve and the orbit stage use, kept as an ambient map so it
    // survives every later change of basis. Probed on the input basis before any
    // reduction: a lattice handed over in its natural structured basis carries the
    // symmetry in exactly those coordinates, and the initial LLL would hide it.
    OrbitOracle oracle;
    double best_resid = 0.0;
    if (have_target) {
        const SymObjective probe = make_objective(red, target, p.sym_lambda);
        best_resid = probe.residual();
        oracle.valid = true;
        oracle.sym = target;
        oracle.frame = red;
        std::cout << "[xsym] input basis: sym_nz=" << probe.sym_nz << " residual="
                  << best_resid << (probe.solved() ? " (already symmetric)" : "") << "\n";
    }

    // Ambient test for the two shift families: if the lattice really is stable
    // under the rotation, that is an exact automorphism, it is basis independent,
    // and its orbits are exactly length preserving. Nothing is left to search for,
    // so this supersedes the annealing discovery.
    if (mode == SymMode::Cyclic || mode == SymMode::NegaCyclic) {
        const int blk = p.sym_block > 0 ? p.sym_block : red.d;
        if (blk >= 2 && blk <= red.d && red.d % blk == 0) {
            const AmbientSym J = (mode == SymMode::NegaCyclic)
                ? make_ambient_negacyclic(red.d, blk)
                : make_ambient_cyclic(red.d, blk);
            if (ambient_stabilizes(red, J)) {
                oracle.ambient_valid = true;
                oracle.ambient = J;
                oracle.valid = true;
                best_resid = 0.0;
                std::cout << "[xsym] the lattice is stable under the ambient "
                          << sym_mode_name(mode) << " rotation of block " << blk
                          << ": exact automorphism of order " << J.order
                          << ", orbits are free and length preserving\n";
            }
        }
    }

    if (oracle.valid) {
        const int ord = oracle.order();
        // The descent walks the orbit only until the rank stops growing, so the
        // whole orbit is affordable and truncating it would just hand the descent
        // a proper sublattice of the module.
        orbit_depth = p.orbit_depth > 0 ? std::min(p.orbit_depth, ord - 1) : ord - 1;
        orbit_depth = std::max(0, orbit_depth);
    }

    // The negacyclic ring Z[x]/(x^n+1) splits whenever n has a divisor k with n/k
    // odd, and the isotypic component for the smallest such k is the lowest rank
    // invariant sublattice available.
    int iso_block = 0;
    if (oracle.valid && mode == SymMode::NegaCyclic) {
        const int n_ring = p.sym_block > 0 ? p.sym_block : L.m;
        if (negacyclic_split_degree(n_ring) > 0) {
            iso_block = n_ring;
            std::cout << "[xsym] the ring Z[x]/(x^" << n_ring
                      << "+1) splits at x^" << negacyclic_split_degree(n_ring)
                      << "+1: descent has a rank "
                      << (L.m / n_ring) * negacyclic_split_degree(n_ring)
                      << " invariant sublattice to search\n";
        }
    }

    if (!p.no_init_lll) {
        try {
            red.lll(1);
        } catch (const ReduceOverflow&) {
            std::cerr << "[xsym] the initial LLL overflowed int64, the input entries "
                         "are too large for this reducer\n";
            return 1;
        } catch (const ReduceAbort&) {
            // No cap is set here, so this cannot fire; kept for uniformity.
        }
    }
    snapshot_best(red, best);
    note_shortest(red, best);
    std::cout << "[t=" << elapsed_since(t0) << "s] start: |b0|^2="
              << norm2_str(red.row_norm2(0)) << " shortest^2="
              << norm2_str(best.shortest_n2) << " rhf=" << red.root_hermite();
    std::cout << "\n";

    Red best_red = red;
    real_t best_b0 = red.row_norm2(0);
    int round = 0;
    int stall = 0;
    double last_print = -1e18;
    double sym_budget = p.sym_seconds;
    int since_gain = 0;
    bool descent_done = false;
    while (!should_stop()) {
        if (p.rounds > 0 && round >= p.rounds) break;
        ++round;
        const real_t record_at_start = best.shortest_n2;

        // Stage 1: progressive BKZ tours.
        for (int beta = std::max(2, p.block_start); beta <= p.block; ++beta) {
            if (should_stop()) break;
            try {
                bkz_tour(red, beta, (real_t)p.prune, p.enum_node_limit);
            } catch (const ReduceOverflow&) {
                break;
            } catch (const ReduceAbort&) {
                break;
            }
        }
        note_shortest(red, best);

        // Stage 2: symmetry discovery, then adopt the basis it produced if it did
        // not make the reduction worse. Skipped once an exact frame is in hand,
        // since there is nothing left to find, and given a shrinking budget when
        // it keeps failing to improve, so it cannot starve the reduction.
        if (have_target && best_resid > 0.0 && !should_stop()) {
            double budget = sym_budget;
            if (p.max_seconds > 0.0)
                budget = std::min(budget, p.max_seconds - elapsed_since(t0));
            if (budget > 0.05) {
                SymObjective found = discover_symmetry(red, target, p, budget,
                                                       seed + (uint64_t)round);
                const double resid = found.residual();
                // The transform is a change of basis into the frame where the
                // symmetry lives. Keeping that frame is what makes the orbit
                // oracle work, so it is recorded whether or not the reduced basis
                // is adopted as the working one. Only a frame that beats the best
                // residual so far replaces the oracle's: a worse frame would spend
                // the enrichment budget on long orbit images.
                Red aligned = red;
                bool have_frame = false;
                bool better_frame = false;
                if (apply_transform(aligned, found.Xt)) {
                    have_frame = true;
                    if (!oracle.valid || resid < best_resid) {
                        best_resid = resid;
                        oracle.valid = true;
                        oracle.sym = target;
                        oracle.frame = aligned;
                        better_frame = true;
                    }
                }
                // Spend less on a search that is not paying off, and go back to
                // the full budget as soon as it does.
                sym_budget = better_frame ? p.sym_seconds
                                          : std::max(0.25, sym_budget * 0.5);
                bool adopted = false;
                if (have_frame) {
                    Red cand = aligned;
                    try {
                        cand.lll(1);
                        // Adopt only if the leading vector did not get worse. The
                        // symmetric frame is a means to cheap orbits, never an
                        // excuse to give up reduction quality.
                        if (cand.row_norm2(0) <= red.row_norm2(0) * 1.02) {
                            red = std::move(cand);
                            adopted = true;
                        }
                    } catch (const ReduceOverflow&) {
                    } catch (const ReduceAbort&) {
                    }
                }
                note_shortest(red, best);
                if (p.verbose || found.solved())
                    std::cout << "[t=" << elapsed_since(t0) << "s] round " << round
                              << " discovery: sym_nz=" << found.sym_nz
                              << " residual=" << resid
                              << (found.solved() ? " (exact symmetry)" : "")
                              << (adopted ? " basis adopted" : " basis kept") << "\n";
            }
        }

        // Stage 3: module descent. Search the orbit sublattice of the current
        // record, which is C-invariant and of lower rank than L, and lift back
        // anything shorter. It only fires after a new record, since the sublattice
        // is a function of the record and re-searching an unchanged one repeats
        // work exactly.
        // The isotypic component is a fixed sublattice, so its minimum is a fixed
        // number: once it has been searched there is nothing further to learn from
        // it, and repeating the search only spends time and pulls the basis back
        // towards the same vectors. The orbit variant does depend on the current
        // record, so it stays tied to one.
        const bool new_record = best.shortest_n2 < record_at_start || round == 1;
        const bool descent_due = iso_block > 0 ? !descent_done : new_record;
        if (oracle.valid && orbit_depth > 0 && descent_due && !should_stop()) {
            // Seed with the shortest rows: the descent lattice is the sum of the
            // ideals they generate, so more seeds means a denser sublattice.
            std::vector<int> order(red.n);
            for (int i = 0; i < red.n; ++i) order[(size_t)i] = i;
            const int want = std::min(std::max(1, p.descent_seeds), red.n);
            std::partial_sort(order.begin(), order.begin() + want, order.end(),
                              [&](int a, int b) {
                                  return red.row_norm2(a) < red.row_norm2(b);
                              });
            std::vector<std::vector<int64_t>> seeds;
            for (int i = 0; i < want; ++i) {
                const int row = order[(size_t)i];
                seeds.emplace_back(red.b_row(row), red.b_row(row) + red.d);
            }
            const real_t before = red.shortest_norm2();
            // Prefer the isotypic component when the ring splits: it has far
            // lower rank than any span of orbits, so it can be searched to the
            // bottom rather than merely reduced.
            DescentReport rep;
            if (iso_block > 0) {
                rep = isotypic_descent(red, oracle, iso_block, p.descent_block,
                                       (real_t)p.prune, p.enum_node_limit);
                if (rep.rank > 0) descent_done = true;
            }
            if (!rep.inserted && rep.rank == 0)
                rep = module_descent(red, oracle, seeds, orbit_depth,
                                     p.descent_block, (real_t)p.prune,
                                     p.enum_node_limit);
            note_shortest(red, best);
            if (p.verbose && rep.rank > 0)
                std::cout << "[t=" << elapsed_since(t0) << "s] round " << round
                          << " descent: rank " << rep.rank << " of " << red.n
                          << ", best^2=" << norm2_str(rep.found_n2) << " vs "
                          << norm2_str(before)
                          << (rep.inserted ? " -> lifted in" : "") << "\n";
        }

        // Stage 4: orbit-enriched sieve, then insert what it found. The sieve is
        // far more expensive than a tour, and on a basis the tours are still
        // improving it only repeats what they already do, so it is held back as a
        // diversification move for when the tours stop producing records.
        const bool sieve_due = p.sieve_after <= 0 || since_gain >= p.sieve_after;
        if (p.use_sieve && sieve_due && !should_stop()) {
            since_gain = 0;
            OrbitSieve sieve;
            std::vector<std::vector<int64_t>> cands;
            const real_t target_n2 = red.row_norm2(0);
            sieve.run(red, p.pool, p.sieve_iters, target_n2, rng, cands,
                      (size_t)p.insert_max);
            int inserted = 0;
            for (const std::vector<int64_t>& amb : cands) {
                if (should_stop()) break;
                real_t n2 = 0;
                for (int k = 0; k < red.d; ++k)
                    n2 += (real_t)amb[(size_t)k] * (real_t)amb[(size_t)k];
                if (n2 <= 0 || n2 >= red.row_norm2(0)) continue;
                // Coordinates are recovered here rather than carried from the
                // sieve, because an earlier insertion in this loop may already
                // have changed the basis.
                std::vector<int64_t> x;
                if (!red.coords_of(amb.data(), x)) continue;
                if (try_insert_block_vector(red, 0, red.n, x)) ++inserted;
            }
            note_shortest(red, best);
            if (inserted > 0 || p.verbose)
                std::cout << "[t=" << elapsed_since(t0) << "s] round " << round
                          << " sieve: " << cands.size() << " candidates, "
                          << inserted << " inserted\n";
        }

        note_shortest(red, best);
        if (best.shortest_n2 < record_at_start) since_gain = 0;
        else ++since_gain;

        // Keep the best basis seen, and escalate when a round changed nothing:
        // fall back to the best basis if this one has drifted worse, then kick it
        // so the next round explores a different neighbourhood.
        const real_t b0 = red.row_norm2(0);
        const bool improved = b0 < best_b0 * (1.0 - 1e-12);
        if (improved) {
            best_b0 = b0;
            best_red = red;
            snapshot_best(red, best);
            stall = 0;
        } else {
            ++stall;
            if (p.stall_rounds > 0 && stall >= p.stall_rounds) {
                if (b0 > best_b0) red = best_red;
                perturb_basis(red, p.perturb_strength, rng);
                note_shortest(red, best);
                stall = 0;
            }
        }

        const double now = elapsed_since(t0);
        if (improved || p.verbose || now - last_print >= 1.0) {
            last_print = now;
            std::cout << "[t=" << now << "s] round " << round << ": |b0|^2="
                      << norm2_str(b0) << " best^2=" << norm2_str(best_b0)
                      << " shortest^2=" << norm2_str(best.shortest_n2)
                      << " rhf=" << red.root_hermite() << "\n";
        }
        if (improved) {
            try {
                write_outputs(p, best);
            } catch (const std::exception& e) {
                std::cerr << "[xsym] write failed: " << e.what() << "\n";
            }
        }
    }

    note_shortest(red, best);
    // Report the best basis seen, not whatever the last perturbation left behind.
    if (red.row_norm2(0) < best_b0) {
        best_b0 = red.row_norm2(0);
        best_red = red;
    }
    snapshot_best(best_red, best);
    note_shortest(best_red, best);
    try {
        write_outputs(p, best);
    } catch (const std::exception& e) {
        std::cerr << "[xsym] write failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "done rounds=" << round << " seconds=" << elapsed_since(t0)
              << " shortest^2=" << norm2_str(best.shortest_n2)
              << (g_interrupted.load(std::memory_order_relaxed) ? " (interrupted)" : "")
              << "\n";
    if (!best.have_transform && !p.no_transform)
        std::cout << "[xsym] the transform overflowed int64 and was not written\n";
    return 0;
}
