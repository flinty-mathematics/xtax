# XTAX

A multithreaded "random congruence annealer" for the integer matrix problem
$X^\top A X = B$ with $B$ diagonal, where $A$ is a symmetric integer matrix (not
necessarily positive definite). The solver does not take a target $B$. It
searches for an integer $X$ that drives $X^\top A X$ to *some* diagonal form by
repeatedly applying integer congruences.

It has two input modes:

- **Matrix mode** (`-A`): diagonalize a given symmetric integer matrix $A$.
- **Lattice mode** (`-L`): given a lattice basis $L$ (rows are vectors), anneal
  its Gram matrix $A = L L^\top$ and report the corresponding basis of the same
  lattice.

Background: https://mathematica.stackexchange.com/a/314866/72682

## What it does

Given an $n \times n$ symmetric integer matrix $A$ (as a CSV file), the solver:

1. Loads $A_0$ from `-A` and starts from $X_0 = I$ (or a supplied initial $X$ via `-X`).
2. Runs a pool of **independent** simulated-annealing workers. Each worker keeps
   its own copy of the working matrix and the accumulating transform and proposes
   random unimodular congruences $P$:
   - **Add** (integer shear): the only score-changing move. With probability
     `--greedy-fraction` it uses the reducing quotient $s = -\mathrm{round}(A_{ji}/A_{ii})$
     (clamped) to knock down an entry. Otherwise it nudges by $\pm 1$.
   - **Swap** (permutation) and **Neg** (sign flip): these only permute / flip the
     signs of absolute values, so they leave the score unchanged but help escape
     local optima.
3. Scores configurations by a sparsity measure $2\sum_{i,j}|A_{ij}| - \sum_i |A_{ii}|$
   (lower is more diagonal) and accepts moves by the Metropolis rule: every move
   that does not raise the score is taken, and uphill moves are taken with
   probability $\exp(-\Delta/T)$. After each $P$, $A \leftarrow P^\top A P$ and
   $X \leftarrow X P$.
4. Cools $T$ geometrically. When a worker stalls it **reheats**, and if it has
   fallen far behind the global best it **reseeds** from that best. Add moves can
   be biased toward "hot" rows (those with the most off-diagonal mass) via a small
   tournament, which makes each move count on dense matrices.
5. Whenever a worker beats the shared global best it reports the new score and
   (throttled) writes the result to disk.
6. The run succeeds when some worker reaches a diagonal $X^\top A X$.

Entry magnitudes are bounded (by $2^{48}$) so the search stays numerically sane
and the score arithmetic cannot overflow. Moves that would exceed the bound are
rejected.

## Building

The project is C++20 and vendors its CLI parser, so no external dependencies are
needed beyond a compiler and CMake.

```
cmake -B build_dir -DCMAKE_BUILD_TYPE=Release
cmake --build build_dir --config Release
```

This same pair of commands works on both single-config generators (Makefiles,
Ninja) and multi-config generators (Visual Studio): each generator simply
ignores the configuration flag it does not use. The executable lands in
`build_dir/xtax` on Linux and macOS, or `build_dir/Release/xtax.exe` on Windows.
Release builds enable AVX2 where supported.

## Usage

### Matrix mode

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

### Lattice mode

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
| `--verbose` | off | Also print the inner annealer's per-slice progress inside `--deflate` / `--deflate-blocks` (purely console output, does not write per-slice CSVs). |
| `-X <file>` | identity | Initial $X$ to continue from (matrix mode). |
| `-w, --workers <int>` | hardware threads | Number of worker threads. |
| `--max-seconds <float>` | `0` | Wall-clock stop. `<= 0` runs until a diagonal is found. |
| `--greedy-fraction <float>` | `0.5` | Probability an Add uses the reducing quotient shear. |
| `--target-fraction <float>` | `0.5` | Probability an Add targets a hot row (`0` = uniform). |
| `--target-samples <int>` | `8` | Tournament size for hot-row / large-pivot selection. |
| `--add-weight <float>` | `0.8` | Relative weight of Add (shear) moves. |
| `--swap-weight <float>` | `0.1` | Relative weight of Swap moves. |
| `--neg-weight <float>` | `0.1` | Relative weight of Neg moves. |
| `--t-init <float>` | `0` (auto) | Initial SA temperature. `<= 0` auto-calibrates from the start score. |
| `--t-min <float>` | `1e-3` | Temperature floor. |
| `--cooling <float>` | `0.999` | Geometric cooling factor per cooling step. |
| `--moves-per-cool <int>` | `200` | Moves between cooling steps. |
| `--stuck-threshold <int>` | `20000` | Moves without improvement before reheating. |
| `--reheat <float>` | `1.0` | Fraction of the initial temperature restored when stuck. |
| `--reseed-factor <float>` | `1.25` | Reseed from the global best when stuck and this far behind it. |
| `--save-interval <float>` | `2.0` | Minimum seconds between `best_*.csv` disk writes. |
| `--deflate` | off | Strict deflation outer loop (see below). Requires a unimodular matrix. Starts from the identity transform. |
| `--deflate-blocks` | off | Relaxed deflation: peel off orthogonal summands. Works on any Gram matrix. Starts from the identity transform. |
| `--deflate-slice <float>` | `0.5` | Deflation: annealing seconds per slice before checking for pivots. |

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

Run for 30 seconds with 16 workers and report the best result found:

```
xtax -A A.csv -w 16 --max-seconds 30
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

## Testing

`tests/run_tests.py` is a dependency-free regression harness (Python 3 standard
library only). It runs the compiled solver on small known inputs and checks
correctness invariants with exact integer arithmetic: that $X$ is unimodular and
$X^\top A X$ matches the reported matrix, that the 10x10 form diagonalizes to its
expected signature (with and without `--deflate`), and that lattice mode preserves
the lattice (exact volume) while reducing the score (with and without
`--deflate-blocks`).

Build first, then run:

```
python tests/run_tests.py
```

The harness auto-locates `build_dir/Release/xtax.exe` (or `build_dir/xtax`). Pass
`--exe <path>` or set `XTAX_EXE` to override. It exits non-zero if any test fails.
