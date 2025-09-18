# XTAX
A CPU parallel 'random congruence annealer' for solving the integer matrix problem $X^\top A X = B$ with $B$ diagonal. 

See here: https://mathematica.stackexchange.com/a/314866/72682

## Building
- Use CMake to generate the Visual Studio project on Windows.
- On Linux, it's as simple as: `cmake -B build_dir && cmake --build build_dir`
  For release builds add `-DCMAKE_BUILD_TYPE=Release`

## What does this do?
Given an $n\times n$ square (symmetric) integer matrix $A$ (not necessarially definite) supplied as a CSV file, this code:
1) Initializes $A_0$ to your CSV file supplied in the `-A` argument and initializes $X_0$ to the identity matrix (or your initial X if you want to continue from a previous result).
2) Spawns a number of workers which go away and try out moves (random unimodular congruence matrices $P_n$). i.e:
  - swaps
  - unimodular shears (add)
  - negations
3) At each cooling step, a worker accepts a move $P_n$ if it improves a sparsity score $2\Sigma|A_{ij}|-\Sigma|A_{ii}|$, i.e tries to reduce off diagonal entries.
   New 'best' (lower) scores cause all workers to stop and receive the updated matrices to work on.
   $A_{n+1}=P_n^\top A_n P_n$ and $X_{n+1}=X_n P_n$
4) If a worker becomes stuck, it tries to 'warm' the system accepts any moves provided their score doesn't exceed the cooling fraction.
5) We have succeeded if we have found an $X$ such that matrix $X^\top A X$ is diagonal.
