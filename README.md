# XTAX

Heavily improved and slop coded with AI assistance.
C++20 command-line tools for integer lattice, matrix reduction, and
combinatorial matrix search:

- **xtax**: a random-congruence annealer that drives a symmetric integer
  matrix $A$ toward a diagonal form via unimodular congruences $X^\top A X$,
  minimizing an $L_1$ sparsity score.
- **xdual**: the same congruence search, but annealing a working Gram $P$ and
  its dual $Q = P^{-1}$ simultaneously, minimizing a combined off-diagonal
  Frobenius score so both the lattice and its dual end up close to orthogonal.
- **xbkz**: a standalone multithreaded BKZ lattice reducer built from scratch
  (Gram-Schmidt, LLL, pruned Schnorr-Euchner enumeration, and an optional block
  sieve), independent of the two annealers above.
- **xweigh**: a standalone fixed-degree ternary annealer that searches for a
  weighing matrix $W(n,w)$ satisfying $W W^\top = wI$.
- **xweigh_cuda**: a GPU population annealer for the same weighing-matrix
  problem on orders that fit in device shared memory.

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
ignores the configuration flag it does not use. The executables land under
`build_dir/` on Linux and macOS, or `build_dir/Release/*.exe` on Windows.
Release builds enable AVX2 where supported.

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

### Restricting the Gram metric

In lattice mode, `--lattice-dims <spec>` changes how the initial Gram is
computed: only the listed 0-based columns (dimensions) of $L$ enter the dot
products, as if all other columns had been discarded. The spec uses the same
syntax as `--gram-rows` (comma-separated indices and inclusive `lo..hi`
ranges). The Gram stays $m \times m$ and symmetric; only its numbers change,
so this is a change of metric, not a restriction of the annealer's moves
(that is what `--gram-rows` is for, and the two compose). The lattice itself
is never modified: the restricted copy exists only for the one-time Gram
calculation, and when the run ends the accumulated unimodular transform is
applied to the full original lattice as usual.

This lets you improve orthogonality or sparsity in a few chosen dimensions,
at the accepted risk of blowing up the basis in the unselected ones. Note
that with fewer selected dimensions than basis vectors the restricted Gram
is singular, and `best_A.csv` holds the annealed *restricted* Gram; the full
Gram is recoverable from `final_L.csv`.

### Options

Exactly one of `-A` or `-L` must be given.

| Option | Default | Description |
|---|---|---|
| `-A <file>` | (one required) | CSV file for $A$ ($n \times n$ integers). |
| `-L, --lattice <file>` | (one required) | CSV lattice basis (rows are vectors). Anneals the Gram $A = L L^\top$. |
| `-X <file>` | identity | Initial $X$ to continue from (matrix mode). |
| `--modulus <uint>` | off | Reduce the loaded input entries ($A$ or $L$) modulo this value once at load (see [Input modulus](#input-modulus)). |
| `--gram-rows <spec>` | all rows | Restrict moves to these 0-based row indices of the working matrix: a comma-separated list of indices and inclusive `lo..hi` ranges, e.g. `0,3..6,9`. Rows outside the set are never used as pivot or target. Not combinable with `--deflate`, `--deflate-blocks`, `--rcm`, or `--centroid`. |
| `--lattice-dims <spec>` | all dims | Lattice mode only: build the initial Gram from these 0-based lattice columns (dimensions) only, same list syntax as `--gram-rows` (see [Restricting the Gram metric](#restricting-the-gram-metric)). |
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

$$F(X) = \lVert \text{offdiag}(P) \rVert_F^2 + c \cdot \lVert \text{offdiag}(Q) \rVert_F^2,$$

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
off-diagonal pairs and `dual` is $\lVert \text{offdiag}(Q) \rVert_F$.

### Options

Exactly one of `-A` or `-L` must be given. `xdual` shares its annealing engine
options with `xtax` (see the table above for `--seed` through `--save-interval`);
the options specific to the primal/dual objective are:

| Option | Default | Description |
|---|---|---|
| `-A <file>` | (one required) | CSV file for a symmetric matrix $A$ ($n \times n$ integers), used as the primal working Gram. |
| `-L, --lattice <file>` | (one required) | CSV lattice basis (rows are vectors). Anneals the Gram $G = L L^\top$ and its dual. |
| `-X <file>` | identity | Initial $X$ to continue from (matrix mode). |
| `--modulus <uint>` | off | Reduce the loaded input entries ($A$ or $L$) modulo this value once at load (see [Input modulus](#input-modulus)). |
| `--gram-rows <spec>` | all rows | Restrict moves to these 0-based row indices of the working matrix: a comma-separated list of indices and inclusive `lo..hi` ranges, e.g. `0,3..6,9`. Rows outside the set are never used as pivot or target. |
| `-t, --threads <int>` | physical cores | Number of worker threads (see `--use-hyperthreads`). |
| `--use-hyperthreads` | off | Default the worker count to all logical processors instead of physical cores (ignored if `--threads` is given). |
| `--no-pin` | off | Do not pin worker threads to physical cores (Windows; pinning is on by default). |
| `--lambda <float>` | `1.0` | Dual weight $c$ in $F = \lVert \text{offdiag}(P) \rVert_F^2 + c \lVert \text{offdiag}(Q) \rVert_F^2$, where `1.0` gives equal initial primal/dual weight. |
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
- `shortest.csv`: the single shortest vector found. This is normally the
  shortest row of the reduced basis, but it is tracked separately and never
  gets longer: if the input basis contained a shorter row than any reduced
  basis achieved (the initial LLL orders rows by projected norm and can
  legitimately lose a short row), that row is preserved and written instead.

On Windows a live progress window shows each worker's phase, tour, current
block size, and enumeration/sieve progress; `--no-gui` disables it. Elsewhere
(and with `--no-gui`) progress is reported only via the final summary lines.

### Options

| Option | Default | Description |
|---|---|---|
| `-L, --lattice <file>` | (required) | CSV lattice basis to reduce (rows are vectors). |
| `--modulus <uint>` | off | Reduce the loaded basis entries modulo this value once at load (see [Input modulus](#input-modulus)). Note the reduced basis spans a different lattice unless the original is $m$-ary; all outputs refer to the reduced input. |
| `-o, --out <file>` | `reduced.csv` | Output CSV for the reduced basis. |
| `--transform-out <file>` | `U.csv` | Output CSV for the unimodular transform $U$ (reduced = $U \cdot L$). |
| `--shortest-out <file>` | `shortest.csv` | Output CSV for the shortest vector found (usually a row of the reduced basis, kept separately when it is shorter than every reduced row). |
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

## Input modulus

`xtax`, `xdual`, and `xbkz` accept `--modulus <uint>` (> 0). It reduces the
entries of the loaded input (the `-A` matrix or the `-L` basis) exactly once,
at load time, so it costs a single pass over the input and adds nothing to the
hot paths.

- **`--modulus m` with `m > 1`**: every entry is replaced by its balanced
  (least absolute) residue in $(-m/2,\, m/2]$, which keeps magnitudes as small
  as possible.
- **`--modulus 1` (saturation mode)**: plain mod 1 would zero everything, so a
  modulus of 1 instead saturates the entries: every nonzero entry (positive or
  negative) becomes `1`, and `0` stays `0`.

The reduction changes the input itself: the tools then operate on, and their
output invariants (`reduced = U * L`, $X^\top A X$ = best working matrix) hold
against, the *reduced* input. For `xbkz` in particular the reduced basis spans
a different lattice than the raw input unless the lattice is $m$-ary. A row
that becomes all zeros under the modulus is rejected by `xbkz` (the basis is no
longer full rank) and warned about by `xtax` / `xdual` (the Gram turns
singular).

## Shared machinery

The congruence annealers and `xbkz` share common C++20 infrastructure: dense
matrix and lattice I/O, graceful Ctrl-C shutdown with a final best-result write,
and (for `xtax` and `xdual`) a templated simulated-annealing engine with
pluggable objectives. `xbkz` has its own independent reduction core and does
not use that engine.

## xweigh: unrestricted weighing-matrix annealer

`xweigh` searches for an $n \times n$ matrix with entries in
$\{-1,0,1\}$ such that

$$W W^\top = w I.$$

Every candidate always has exactly $w$ nonzeros in every row and every column.
The annealer minimizes the exact integer residual

$$\sum_{i \lt j} |\langle W_i,W_j\rangle|,$$

which is zero exactly when all distinct rows are orthogonal. Because the
candidate is square and has nonzero weight, a solution also satisfies
$W^\top W = wI$.

This is a direct combinatorial search, not a unimodular congruence search.
Starting from $I$, a congruence $X^\top X=wI$ with unimodular $X$ would force
$w=1$ by determinants, so xtax's shear moves cannot generate nontrivial
weighing matrices.

### Search

The initial support is a randomized $w$-regular bipartite graph. Two move
types keep the row and column weights exact:

- flip the sign of one nonzero entry;
- switch the support of a legal $2 \times 2$ submatrix, moving two nonzeros
  across its diagonal.

A dense cached Gram matrix makes evaluating either move $O(n)$. The hot path
stores `W` column-major as signed bytes so the changed columns are streamed
contiguously, stores the Gram as signed 16-bit integers, and keeps only the
smaller of each row's support or zero set. Multiple workers use parallel
tempering by default. Each worker owns its cache-aligned state, so memory use
is proportional to `threads * n^2`; startup reports the estimated memory per
worker and in total.

When $w$ is a power of two and divides $n$, xweigh constructs a direct sum of
Sylvester Hadamard blocks as the start state and skips annealing when that
start is already a weighing matrix. This is a valid (possibly decomposable)
$W(n,w)$. Other parameters use the annealer.

### Usage

```
xweigh <n> <w> [options]
```

The best candidate is written atomically to `best_W.csv` by default. A run
that reaches score zero performs an independent exact verification before
reporting success. On a time limit or Ctrl-C, the best fixed-weight candidate
is still written, but it need not yet be a weighing matrix.

| Option | Default | Description |
|---|---|---|
| `n` | required | Matrix order, in `1..32767`. |
| `w` | required | Weight, in `1..n`. |
| `-o, --out <file>` | `best_W.csv` | Output CSV for the best matrix. |
| `--start <file>` | none | Start from a complete ternary CSV with weight `w` in every row and column. |
| `-t, --threads <int>` | physical cores | Number of worker states. |
| `--use-hyperthreads` | off | Default to all logical processors. |
| `--no-pin` | off | Disable Windows physical-core affinity. |
| `--seed <uint>` | `0` | Base seed (`0` uses `random_device`). |
| `--max-seconds <float>` | `0` | Search budget (`0` runs until solved or interrupted). |
| `--save-interval <float>` | `2` | Minimum seconds between atomic best writes. |
| `--sign-fraction <float>` | `0.5` | Probability of proposing a sign flip. |
| `--greedy-fraction <float>` | `0.5` | Probability of best-of-samples move selection. |
| `--candidate-samples <int>` | `4` | Candidates scored by a sampled greedy move. |
| `--target-fraction <float>` | `0.7` | Probability of targeting a high-residual row. |
| `--target-samples <int>` | `8` | Hot-row tournament size. |
| `--tempering / --no-tempering` | on | Parallel tempering for two or more workers. |
| `--exchange-interval <int>` | `2000` | Approximate moves per worker between exchanges. |
| `--t-init <float>` | `0` | Initial temperature (`0` auto-calibrates). |
| `--t-min <float>` | `0.25` | Minimum temperature. |
| `--cooling <float>` | `0.999` | Single-worker geometric cooling factor. |
| `--moves-per-cool <int>` | `500` | Single-worker moves between cooling steps. |
| `--stuck-threshold <int>` | `50000` | Moves without improvement before reheating. |
| `--reheat <float>` | `1` | Fraction of initial temperature restored. |
| `--reseed-factor <float>` | `1.25` | Reseed a lagging worker from the global best. |

Examples:

```
xweigh 7 4
xweigh 512 25 -t 16 --max-seconds 60
xweigh 1000 64 --seed 1234 -o W-1000-64.csv
xweigh 35 25 --start previous-best.csv --max-seconds 300
```

xweigh rejects only elementary proven impossibilities before searching:

- $w = n$ when $n \gt 2$ and $n$ is not a valid Hadamard order (1, 2, or a
  multiple of 4);
- odd $n$ when $w$ is not a perfect square;
- odd $n$ when $n \lt w + \sqrt{w} + 1$;
- $n \equiv 2 \pmod 4$ when $w$ is not a sum of two integer squares.

Failure to reach score zero is not evidence of nonexistence. Unrestricted
search remains exponential in difficult cases, and large dense instances can
require substantial memory and time.

### CUDA population annealer

When a CUDA compiler is available, CMake also builds `xweigh_cuda`. This is a
separate small-order solver which runs many independent annealing replicas on
the GPU. Each CUDA block owns one replica, keeps its matrix and exact Gram
cache in shared memory, and uses separate warps to score sampled candidate
moves concurrently. The CPU `xweigh` target and its search are unchanged.

```
cmake -B build_dir -DCMAKE_BUILD_TYPE=Release -DXTAX_BUILD_CUDA=ON
cmake --build build_dir --config Release
build_dir/Release/xweigh_cuda 35 25 --max-seconds 60
```

`--replicas 0` selects a population from the device occupancy; an explicit
positive value overrides it. `--moves-per-launch` controls how often the GPU
returns to the host for stopping, progress, and checkpointing. The annealing
options shared with `xweigh` have the same meaning, but this first CUDA path
uses a cooling/reheating population rather than CPU parallel tempering.
`--start <file>` initializes every replica from the same validated candidate;
their independent random streams and temperature ladder then make them
diverge. In both executables the CSV must contain all $n^2$ ternary entries
and already have exactly $w$ nonzeros in every row and column. Unknown or
blank cells are not accepted.

For difficult resumed candidates, the CUDA population also uses three larger
search mechanisms:

- atomic double-sign moves and support switches which can choose new signs;
- replicas guided by squared Gram residual or by the number of odd support
  intersections, alongside the usual $L_1$ replicas;
- elite restarts after a global-best plateau. These preserve the best replica
  and distribute the others across low-cost uphill support/sign exits,
  row-sign perturbations, support perturbations, and mixed random kicks.

The relevant controls are `--double-sign-fraction`,
`--switch-sign-fraction`, `--squared-objective-fraction`,
`--parity-objective-fraction`, `--restart-interval`,
`--restart-fraction`, and `--restart-kick-min/max`. Setting
`--restart-interval 0`, both compound fractions to zero, and both alternate
objective fractions to zero recovers the previous independent-replica move
set.

```
build_dir/Release/xweigh_cuda 35 25 \
  --start previous-best.csv --max-seconds 300
```

The complete state of a replica must fit in one block's shared memory.
`xweigh_cuda` checks that limit for the selected device and rejects larger
orders with the required and available byte counts. Use CPU `xweigh` for those
instances. CUDA support is optional so systems without the CUDA toolkit,
including macOS, continue to configure and build the CPU tools normally.
