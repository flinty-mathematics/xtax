# XTAX

Heavily improved and slop coded with AI assistance.
Three related C++20 command-line tools for integer lattice and matrix
reduction, sharing a common CSV-based I/O layer and (for the two annealers) a
common multithreaded simulated-annealing engine:

- **xtax**: a random-congruence annealer that drives a symmetric integer
  matrix $A$ toward a diagonal form via unimodular congruences $X^\top A X$,
  minimizing an $L_1$ sparsity score.
- **xdual**: the same congruence search, but annealing a working Gram $P$ and
  its dual $Q = P^{-1}$ simultaneously, minimizing a combined off-diagonal
  Frobenius score so both the lattice and its dual end up close to orthogonal.
- **xbkz**: a standalone multithreaded BKZ lattice reducer built from scratch
  (Gram-Schmidt, LLL, pruned Schnorr-Euchner enumeration, and an optional block
  sieve), independent of the two annealers above.

Background on the congruence-annealing idea:
https://mathematica.stackexchange.com/a/314866/72682

## Building

The project is C++20 and vendors its CLI parser (CLI11), so no external
dependencies are needed beyond a compiler and CMake.

```
cmake -B build_dir -DCMAKE_BUILD_TYPE=Release
cmake --build build_dir --config Release
```

This same pair of commands works on both single-config generators (Makefiles,
Ninja) and multi-config generators (Visual Studio): each generator simply
ignores the configuration flag it does not use. The three executables land in
`build_dir/xtax`, `build_dir/xbkz`, `build_dir/xdual` on Linux and macOS, or
`build_dir/Release/*.exe` on Windows. Release builds enable AVX2 where
supported.

Two opt-in CMake options tune the floating-point-heavy paths (`xbkz`'s
reducer and `xdual`'s dual maintenance):

- `-DXTAX_AVX512=ON` compiles the SIMD reductions for AVX-512 instead of AVX2.
  Its wider (8-lane) summation order can differ from AVX2/scalar in the last
  ULP, so this is off by default.
- `-DXTAX_FAST_MATH=ON` compiles `xbkz` and `xdual` with fast/reassociating
  floating point (`/fp:fast` or `-ffast-math`). Faster, but no longer
  bit-for-bit reproducible across builds, so this is off by default.

## xtax: congruence annealer (L1 diagonalization)

A multithreaded "random congruence annealer" for the integer matrix problem
$X^\top A X = B$ with $B$ diagonal, where $A$ is a symmetric integer matrix
(not necessarily positive definite). The solver does not take a target $B$. It
searches for an integer $X$ that drives $X^\top A X$ to *some* diagonal form by
repeatedly applying integer congruences.

It has two input modes:

- **Matrix mode** (`-A`): diagonalize a given symmetric integer matrix $A$.
- **Lattice mode** (`-L`): given a lattice basis $L$ (rows are vectors), anneal
  its Gram matrix $A = L L^\top$ and report the corresponding basis of the same
  lattice.

### What it does

Given an $n \times n$ symmetric integer matrix $A$ (as a CSV file), the solver:

1. Loads $A_0$ from `-A` and starts from $X_0 = I$ (or a supplied initial $X$ via `-X`).
2. Runs a pool of simulated-annealing worker threads. Each worker keeps its own
   copy of the working matrix and the accumulating transform and proposes
   unimodular **Add** (integer shear) congruences $P$: row/column $j$ gains $s$
   times row/column $i$. With probability `--greedy-fraction` the shear uses
   the **exact best integer $s$** for that pair (the $|A_{ik}|$-weighted median
   of the off-diagonal breakpoints, refined against the diagonal term).
   Otherwise it nudges by $\pm 1$ for exploration.
3. Scores configurations by a sparsity measure $2\sum_{i,j}|A_{ij}| - \sum_i |A_{ii}|$
   (lower is more diagonal) and accepts moves by the Metropolis rule: every move
   that does not raise the score is taken, and uphill moves are taken with
   probability $\exp(-\Delta/T)$. After each move, $A \leftarrow P^\top A P$ and
   $X \leftarrow X P$.
4. With two or more threads the pool runs **parallel tempering** by default: each
   worker sits on a rung of a geometric temperature ladder and a coordinator
   periodically swaps rungs between neighbouring workers by the replica-exchange
   rule, so good configurations diffuse toward the cold end while hot rungs keep
   exploring. `--no-tempering` restores the classic per-worker geometric cooling
   with reheat-on-stall and reseed-from-best. Add moves can be biased toward
   "hot" rows (those with the most off-diagonal mass) via a small tournament,
   and `--sweep-fraction` optionally fires a greedy exact-shear reduction sweep
   as a plateau breaker when a worker stalls.
5. Whenever a worker beats the shared global best it reports the new score and
   (throttled) writes the result to disk.
6. The run succeeds when some worker reaches a diagonal $X^\top A X$.

Entry magnitudes are bounded (by $2^{48}$) so the search stays numerically sane
and the score arithmetic cannot overflow. Moves that would exceed the bound are
rejected.

### Usage

Matrix mode:

```
xtax -A <matrix.csv> [options]
```

`-A` is a CSV of the square integer matrix $A$ (one row per line,
comma-separated). The solver writes two CSV files to the current directory:

- `best_X.csv`: the transform $X$ found so far.
- `best_A.csv`: the corresponding $X^\top A X$ (the diagonalized matrix).

For small matrices ($n \le 20$) the final $A$ and $X$ are also printed to stdout.
Progress is logged as `[t=...s] new best score=... (thread k)` lines, and the run
ends with a `done moves=... seconds=... moves_per_sec=...` summary.

Lattice mode:

```
xtax -L <basis.csv> [options]
```

`-L` is a CSV of a lattice basis whose **rows** are the basis vectors (so an
$m \times d$ file is $m$ vectors in dimension $d$, and it need not be square). The
solver forms the Gram matrix $A = L L^\top$ and anneals it exactly as in matrix
mode. Because $X^\top A X = X^\top L L^\top X = (X^\top L)(X^\top L)^\top$, the
annealed Gram is the Gram of the new basis $X^\top L$, a basis of the *same*
lattice that is more orthogonal.

Lattice mode writes to the current directory:

- `final_L.csv`: the final basis $X^\top L$ of the same lattice.
- `best_A.csv`: its Gram matrix (the annealed $X^\top A X$).
- `best_X.csv`: the transform $X$ relating the input basis to `final_L`.

### Options

Exactly one of `-A` or `-L` must be given.

| Option | Default | Description |
|---|---|---|
| `-A <file>` | (one required) | CSV file for $A$ ($n \times n$ integers). |
| `-L, --lattice <file>` | (one required) | CSV lattice basis (rows are vectors). Anneals the Gram $A = L L^\top$. |
| `-X <file>` | identity | Initial $X$ to continue from (matrix mode). |
| `-t, --threads <int>` | physical cores | Number of worker threads (see `--use-hyperthreads`). |
| `--use-hyperthreads` | off | Default the worker count to all logical processors instead of physical cores (ignored if `--threads` is given). |
| `--no-pin` | off | Do not pin worker threads to physical cores (Windows; pinning is on by default). |
| `--seed <int>` | `0` | Base RNG seed for reproducible worker seeding (`0` = random). |
| `--max-seconds <float>` | `0` | Wall-clock stop. `<= 0` runs until a diagonal is found. |
| `--greedy-fraction <float>` | `0.5` | Probability an Add uses the exact best-integer shear. |
| `--target-fraction <float>` | `0.5` | Probability an Add targets a hot row (`0` = uniform). |
| `--target-samples <int>` | `8` | Tournament size for hot-row / large-pivot selection. |
| `--tempering / --no-tempering` | on | Parallel-tempering ladder with replica exchange (needs 2+ threads; single-thread runs always use the cooling schedule). |
| `--exchange-interval <int>` | `2000` | Moves between replica-exchange sweeps (tempering mode). |
| `--worker-diversity <float>` | `0.4` | Spread of per-worker greedy/target fraction offsets (`0` = identical workers). |
| `--sweep-fraction <float>` | `0` | Probability of a greedy exact-shear reduction sweep when a worker stalls (`0` = off). |
| `--t-init <float>` | `0` (auto) | Initial SA temperature (ladder top under tempering). `<= 0` auto-calibrates from the start score. |
| `--t-min <float>` | `1e-3` | Temperature floor (ladder bottom under tempering). |
| `--cooling <float>` | `0.999` | Geometric cooling factor per cooling step (`--no-tempering`). |
| `--moves-per-cool <int>` | `200` | Moves between cooling steps (`--no-tempering`). |
| `--adaptive-cooling` | off | Nudge the cooling rate toward a target acceptance ratio (`--no-tempering`). |
| `--stuck-threshold <int>` | `20000` | Moves without improvement before reheating / sweeping. |
| `--reheat <float>` | `1.0` | Fraction of the initial temperature restored when stuck (`--no-tempering`). |
| `--reseed-factor <float>` | `1.25` | Reseed from the global best when stuck and this far behind it (`--no-tempering`). |
| `--save-interval <float>` | `2.0` | Minimum seconds between `best_*.csv` disk writes. |
| `--rcm` | off | Reorder the working matrix toward a band (Reverse Cuthill-McKee) on each new best. |
| `--centroid` | off | Reorder by iterative row centre-of-mass on each new best. |
| `--deflate` | off | Strict deflation outer loop (see below). Requires a unimodular matrix. Starts from the identity transform. |
| `--deflate-blocks` | off | Relaxed deflation: peel off orthogonal summands. Works on any Gram matrix. Starts from the identity transform. |
| `--deflate-slice <float>` | `0.5` | Deflation: annealing seconds per slice before checking for pivots. |
| `--verbose` | off | Also print the inner annealer's per-slice progress inside `--deflate` / `--deflate-blocks` (purely console output, does not write per-slice CSVs). |

### Deflation

Both deflation flags wrap the annealer in an outer loop that **locks in pivots and
shrinks the problem**: a coordinate that has been "solved" is frozen and dropped
from the active set, so it is never disturbed again. Because frozen coordinates
cannot be undone, progress is **monotone**. Where the plain annealer can churn away
a good pivot, deflation cannot. The loop reports progress periodically as
`[deflate] ...` lines, writes `best_X.csv` and `best_A.csv` as it goes (throttled by
`--save-interval`), and always starts from the identity transform (it ignores `-X`).

The two flags differ in *what* counts as a lockable pivot:

**`--deflate` (strict).** Diagonalizing $A$ reduces to repeatedly finding a
coordinate whose diagonal entry is $\pm 1$ and peeling it off. A $\pm 1$ pivot
lets you zero its whole row/column with *exact* integer shears, so that coordinate
splits off as a $\langle \pm 1 \rangle$ summand and the remaining problem is one
dimension smaller. Every full pivot of a unimodular form is forced to $\pm 1$, so
this drives such a form straight to a $\pm 1$ diagonal. Because of this, `--deflate`
first verifies the matrix is unimodular ($\det = \pm 1$, checked modulo a couple of
large primes) and **fails with an error if it is not**. On a non-unimodular matrix
no $\pm 1$ pivot need ever exist, so the loop would spin without locking anything.

**`--deflate-blocks` (relaxed).** For a general (non-unimodular) Gram matrix you
cannot expect $\pm 1$ pivots, so this mode instead locks any coordinate that has
become **orthogonal to all other active coordinates** (its off-diagonal row is
already zero), splitting off a $\langle c \rangle$ summand of any norm $c$. This
needs no shears, so it never overflows and works on any Gram. It can only peel off
*orthogonal summands*. It helps a lattice that decomposes as an orthogonal direct
sum (where it can fully diagonalize), and simply never fires on an irreducible
lattice, where it degrades gracefully to the plain sliced annealer. It cannot
manufacture $\pm 1$ pivots, so it is not a substitute for the strict mode on
unimodular forms. (Note: a positive-definite irreducible lattice has no orthogonal
basis at all, so no deflation-style trick can diagonalize it.)

### Examples

Diagonalize a matrix, running until a diagonal is found:

```
xtax -A A.csv
```

Run for 30 seconds with 16 threads and report the best result found:

```
xtax -A A.csv -t 16 --max-seconds 30
```

Continue from a previously found transform:

```
xtax -A A.csv -X best_X.csv --max-seconds 30
```

Anneal a lattice's Gram matrix for 30 seconds and report the resulting basis:

```
xtax -L basis.csv --max-seconds 30
```

Diagonalize a unimodular form with strict deflation (locks in $\pm 1$ pivots as
they appear, and errors out if the matrix is not unimodular):

```
xtax -A unimodular.csv --deflate
```

Peel orthogonal summands off a lattice Gram with relaxed deflation:

```
xtax -L basis.csv --deflate-blocks --max-seconds 30
```

## xdual: simultaneous primal/dual congruence annealer

`xdual` runs the same unimodular-congruence search as `xtax`, but with a
different objective: instead of driving a single working Gram to a diagonal by
$L_1$ sparsity, it anneals the lattice *and its dual* at the same time. Under
the same congruence move it keeps:

- $P = X^\top G X$: the primal working Gram, exact integer, starting at $G$.
- $Q = P^{-1}$: the true dual lattice Gram, double precision, starting at $G^{-1}$.

and minimizes the combined squared-Frobenius off-diagonal score

$$F(X) = \lVert \operatorname{offdiag}(P) \rVert_F^2 + c \cdot \lVert \operatorname{offdiag}(Q) \rVert_F^2,$$

so a low score means both the basis and its dual basis are close to
orthogonal. The move $P \to E^\top P E$ (for a unimodular $E$) sends
$P^{-1} \to E^{-1} P^{-1} E^{-\top}$, which is the same shear machinery applied
to $Q$ with the pivot/target swapped and the sign of the shear flipped, so the
dual is maintained incrementally with no per-move inversion.

The dual is floating-point on purpose: it is only a search-guidance penalty,
not part of the exact output (the basis $X^\top L$ is recovered from the exact
integer $P$/$X$). $Q$ is re-inverted from the exact $P$ periodically
(`--dual-refresh`), and additionally whenever a sampled residual of $PQ - I$
exceeds a tolerance (`--dual-check` / `--dual-tol`), to bound floating-point
drift. `--lambda` sets the dual weight $c$ (normalized so `--lambda 1.0` gives
the dual equal initial weight to the primal); `--lambda-ramp` can phase the
dual weight in gradually so the early search optimizes the primal freely.

Deflation is intentionally not offered here (it is `xtax`-specific): a
diagonal primal already forces a diagonal dual, so there is no separate
pivot-locking scheme for the dual.

### Usage

Matrix mode:

```
xdual -A <matrix.csv> [options]
```

`-A` is a CSV of the symmetric integer matrix used as the primal working Gram.
The solver writes to the current directory:

- `best_X.csv`: the transform $X$ found so far.
- `best_P.csv`: the primal Gram $X^\top A X$.
- `best_Q.csv`: the dual $Q \approx P^{-1}$ (double precision, full round-trip precision).

Lattice mode:

```
xdual -L <basis.csv> [options]
```

Builds the Gram $G = L L^\top$ and anneals it (and its dual) exactly as in
matrix mode. Writes `final_L.csv` (the final basis $X^\top L$) in addition to
`best_P.csv`, `best_Q.csv`, and `best_X.csv`.

For small matrices ($n \le 20$) the final $P$, $Q$, and $X$ (or $L$) are also
printed to stdout. Progress is logged as `[t=...s] new best score=... primal=...
dual=... (thread k)` lines, where `primal` is the number of nonzero primal
off-diagonal pairs and `dual` is $\lVert \operatorname{offdiag}(Q) \rVert_F$.

### Options

Exactly one of `-A` or `-L` must be given. `xdual` shares its annealing engine
options with `xtax` (see the table above for `--seed` through `--save-interval`);
the options specific to the primal/dual objective are:

| Option | Default | Description |
|---|---|---|
| `-A <file>` | (one required) | CSV file for a symmetric matrix $A$ ($n \times n$ integers), used as the primal working Gram. |
| `-L, --lattice <file>` | (one required) | CSV lattice basis (rows are vectors). Anneals the Gram $G = L L^\top$ and its dual. |
| `-X <file>` | identity | Initial $X$ to continue from (matrix mode). |
| `-t, --threads <int>` | physical cores | Number of worker threads (see `--use-hyperthreads`). |
| `--use-hyperthreads` | off | Default the worker count to all logical processors instead of physical cores (ignored if `--threads` is given). |
| `--no-pin` | off | Do not pin worker threads to physical cores (Windows; pinning is on by default). |
| `--lambda <float>` | `1.0` | Dual weight $c$ in $F = \lVert \operatorname{offdiag}(P) \rVert_F^2 + c \lVert \operatorname{offdiag}(Q) \rVert_F^2$, where `1.0` gives equal initial primal/dual weight. |
| `--dual-refresh <int>` | `200000` | Moves between unconditional exact re-inversions of the dual (`0` = never). |
| `--dual-check <int>` | `25000` | Moves between sampled dual residual checks that can trigger an early re-inversion (`0` = never). |
| `--dual-tol <float>` | `1e-6` | Sampled residual of $PQ - I$ that triggers an early dual re-inversion. |
| `--lambda-ramp <float>` | `0` (off) | Ramp the dual weight linearly from `0` to its full value over this many seconds (`0` runs at full weight from the start). |
| `--seed <int>` | `0` | Base RNG seed for reproducible worker seeding (`0` = random). |
| `--max-seconds <float>` | `0` | Wall-clock stop. `<= 0` runs until interrupted or fully diagonal. |
| `--greedy-fraction <float>` | `0.5` | Probability an Add uses the exact best-integer shear. |
| `--target-fraction <float>` | `0.5` | Probability an Add targets a hot row (`0` = uniform). |
| `--target-samples <int>` | `8` | Tournament size for hot-row / large-pivot selection. |
| `--tempering / --no-tempering` | on | Parallel-tempering ladder with replica exchange (needs 2+ threads; single-thread runs always use the cooling schedule). |
| `--exchange-interval <int>` | `2000` | Moves between replica-exchange sweeps (tempering mode). |
| `--worker-diversity <float>` | `0.4` | Spread of per-worker greedy/target fraction offsets (`0` = identical workers). |
| `--sweep-fraction <float>` | `0` | Probability of a greedy exact-shear reduction sweep when a worker stalls (`0` = off). |
| `--t-init <float>` | `0` (auto) | Initial SA temperature (ladder top under tempering). `<= 0` auto-calibrates from the start score. |
| `--t-min <float>` | `1e-3` | Temperature floor (ladder bottom under tempering). |
| `--cooling <float>` | `0.999` | Geometric cooling factor per cooling step (`--no-tempering`). |
| `--moves-per-cool <int>` | `200` | Moves between cooling steps (`--no-tempering`). |
| `--adaptive-cooling` | off | Nudge the cooling rate toward a target acceptance ratio (`--no-tempering`). |
| `--stuck-threshold <int>` | `20000` | Moves without improvement before reheating / sweeping. |
| `--reheat <float>` | `1.0` | Fraction of the initial temperature restored when stuck (`--no-tempering`). |
| `--reseed-factor <float>` | `1.25` | Reseed from the global best when stuck and this far behind it (`--no-tempering`). |
| `--save-interval <float>` | `2.0` | Minimum seconds between `best_*.csv` disk writes. |

## xbkz: multithreaded BKZ lattice reducer

A standalone BKZ (Block Korkine-Zolotarev) lattice reducer, independent of the
congruence annealers above. It reduces a lattice basis to one with shorter,
more orthogonal vectors using its own from-scratch reduction core: a
double-precision Gram-Schmidt process, an LLL inner reducer, and a pruned
Schnorr-Euchner enumeration as the per-block SVP (shortest-vector) oracle, with
an optional list-based Gauss sieve as a faster oracle for large blocks. Basis
entries are `int64` with overflow checks, which suits the modest-entry bases
the annealers above tend to produce; very large raw entries are out of scope
(there is no bignum path).

### How it works

- **Tours.** Each worker thread runs independent BKZ tours on its own copy of
  the basis: a tour visits every window $[\kappa, \kappa+\beta)$ of the basis
  in some order and, for each window, searches for a shorter vector in the
  projected sublattice (the SVP oracle for that block), inserting it via a
  unimodular block transform plus a local LLL pass if one is found.
- **Progressive schedule.** By default (`--no-progressive` to disable) each
  worker ramps its block size $\beta$ up tour by tour from `--block-start` to
  `--block`, diversified per worker so the pool covers a range of block sizes
  at once; smaller-$\beta$ tours precondition the basis for the larger, more
  expensive ones that follow.
- **SVP oracle.** Below `--sieve-beta` (or when sieving is disabled) every
  block uses pruned Schnorr-Euchner enumeration (`--prune`, `--enum-node-limit`,
  optional `--gh-factor` Gaussian-heuristic radius cap, optional
  `--preprocess-beta` BKZ-2.0-style local preprocessing). Above `--sieve-beta`
  a tour instead uses a list-based Gauss sieve (`--sieve-pool`,
  `--sieve-iters`) as the oracle; a single tour never mixes the two.
- **Worker coordination.** Workers share one global best basis under a mutex.
  When a worker's tour improves on its own frontier it nudges the basis with a
  small unimodular perturbation and keeps exploring nearby. When a worker
  stalls for `--reseed-every-k` tours without local improvement, it either
  perturbs more strongly in place (if it still holds the frontier) or jumps
  back to the global best and diversifies (if it has fallen behind), which
  balances intensifying the best basin against broad exploration.
- **Transform tracking.** With transform tracking on (default; `--no-transform`
  to disable) each worker also carries the unimodular $U$ such that the
  reduced basis equals $U L_0$, written to `--transform-out` on completion.

### Usage

```
xbkz -L <basis.csv> [options]
```

`-L` is a CSV of the lattice basis to reduce (rows are vectors). The reducer
writes to the current directory (or wherever `-o` / `--transform-out` /
`--shortest-out` point):

- `reduced.csv`: the reduced basis.
- `U.csv`: the unimodular transform ($\text{reduced} = U \cdot L$), unless
  `--no-transform` is given.
- `shortest.csv`: the single shortest row of the reduced basis.

On Windows a live progress window shows each worker's phase, tour, current
block size, and enumeration/sieve progress; `--no-gui` disables it. Elsewhere
(and with `--no-gui`) progress is reported only via the final summary lines.

### Options

| Option | Default | Description |
|---|---|---|
| `-L, --lattice <file>` | (required) | CSV lattice basis to reduce (rows are vectors). |
| `-o, --out <file>` | `reduced.csv` | Output CSV for the reduced basis. |
| `--transform-out <file>` | `U.csv` | Output CSV for the unimodular transform $U$ (reduced = $U \cdot L$). |
| `--shortest-out <file>` | `shortest.csv` | Output CSV for the shortest vector (first reduced row). |
| `--no-transform` | off | Do not track or write the transform $U$ (saves memory and time; recommended for large $n$, where each worker otherwise holds a full $n \times n$ transform in addition to its basis and Gram data). |
| `-t, --threads <int>` | physical cores | Number of worker threads (default: physical core count). |
| `--use-hyperthreads` | off | Default the worker count to all logical processors instead of physical cores (ignored if `--threads` is given). |
| `--no-pin` | off | Do not pin worker threads to physical cores (Windows; by default each worker is pinned to its own core). |
| `-b, --block <int>` | `20` | Maximum BKZ block size. |
| `--block-start <int>` | `2` | Minimum BKZ block size (each tour picks a size in `[block-start, block]`, progressively or at random; see `--no-progressive`). |
| `--delta <float>` | `0.99` | LLL delta in `(0.25, 1.0)`. |
| `--prune <float>` | `0` | Enumeration pruning in `[0, 1]`. `0` is exact, higher is faster but may miss vectors. |
| `--enum-node-limit <int>` | `10000000` | Maximum Schnorr-Euchner nodes per block (`0` = unlimited). |
| `--no-progressive` | off | Disable the per-worker progressive beta schedule and pick a random block size per tour instead. |
| `--preprocess-beta <int>` | `0` | Local block preprocessing block size before each full enumeration (`0` = off); a cheap smaller-beta pass that shrinks the enumeration tree (BKZ 2.0 style). |
| `--gh-factor <float>` | `0` (off) | Gaussian-heuristic enumeration radius cap: the search radius is capped at `gh-factor * GH(block)`. Around `1.1` prunes hard blocks; vectors missed by the cap are recovered by re-randomization across tours and workers. |
| `--sieve-beta <int>` | `0` | Use the block sieve instead of enumeration for tours whose block size exceeds this (`0` = enumeration only). A tour uses a single oracle, never a mix. |
| `--sieve-pool <int>` | `64` | Block sieve pool size (used only by sieve tours). |
| `--sieve-iters <int>` | `16` | Sieve work budget per block is `sieve-pool * sieve-iters + seeds`. |
| `--max-seconds <float>` | `0` | Wall-clock budget. `0` runs until Ctrl-C. |
| `--reseed-every-k <int>` | `10` | Reseed a worker from the global best after this many consecutive tours without local improvement (`0` disables reseeding). |
| `--seed <int>` | random | Base RNG seed. |
| `--no-init-lll` | off | Skip the shared initial LLL pass (use when the input is already reduced). Perturbation shears still run, but their follow-up LLL is skipped so workers diversify without re-reducing. |
| `--no-gui` | off | Run without the Win32 progress window (Windows only). |

## Shared machinery

All three tools share a small set of header-only C++20 pieces under `src/`:

- `mat_io.hpp`: the dense integer `Matrix` (and floating-point `Matrixd`) types,
  the `Lattice` basis type, atomic CSV read/write, and small integer helpers
  (nearest-integer division, extended gcd, overflow-checked axpy, a
  mod-prime unimodularity test).
- `stop_signal.hpp`: the Ctrl-C / stop-flag plumbing shared by all worker
  pools, so every tool shuts down gracefully and writes its best result so far.
- `congruence_anneal.hpp`: the templated simulated-annealing engine shared by
  `xtax` and `xdual` (worker threads and affinity, the temperature schedule,
  the move loop, the plateau-breaking reduction sweep, and the throttled
  global-best publish). Each tool supplies its own objective policy (`L1Objective`
  for `xtax`, `DualObjective` for `xdual`); `xbkz` has its own independent
  reduction core and does not use this engine.

`bench/benchmark.py` (xtax) and `bench/xdual_benchmark.py` (xdual) measure how
quickly the score drops for one or more configurations on a given matrix.
