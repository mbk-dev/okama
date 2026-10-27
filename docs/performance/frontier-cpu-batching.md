# Single-period frontier CPU batching — issue 127

Measured 2026-10-26 for [issue 127](https://github.com/mbk-dev/okama/issues/127).

`EfficientFrontierSingle.get_monte_carlo()` and `get_grid_portfolios()` now prepare pandas covariance and column means once per call, then calculate risk, annualized mean return and approximate CAGR in float64 NumPy chunks. The public API and weight generators are unchanged. No dependency was added.

## Selection from actual-method profiling

The baseline profiler for 128 Monte Carlo points spent 0.144 s in the actual single-period method: 0.069 s in the scalar mean helper and 0.068 s in the scalar risk helper. It called `DataFrame.cov()` 128 times and `DataFrame.mean()` 256 times. The grid profile showed the same repetition (66 covariance and 132 mean calls for 66 points). Candidate profiling shows one covariance and one mean calculation per call; tests pin covariance work independently of portfolio count. Profile timings include profiler overhead and are diagnostic, not benchmark timings.

The actual multi-period class in this revision is named `EfficientFrontier` (the issue calls it `EfficientFrontierMulti`). Profiling its annual-rebalancing Monte Carlo method took 0.886 s, with 0.821 s inside `Rebalance.return_ror_ts_ef()`; its grid call took 0.423 s, with 0.389 s in that function. It is a different numerical path, so this change selects the single-period hot path, as permitted by the issue's at-least-one-path acceptance criterion. Multi-period methods, calendar semantics and scalar optimization are unchanged; no speedup is claimed for them.

## Method and environment

- Baseline library revision: `b33ca12bdd81b4107c2de484fff581eda1023ce1`.
- Candidate: uncommitted implementation on that base. Exact library diff SHA-256: `249f881c664972091b41b2d6a1f6f822a21041e3f850b64f17cd29e2dfa11e3c`. Source file SHA-256: `f2f2ac83fe594a006613130e3fc25fe4ea2665ea57b257b4ce8176f84d70f8ae`.
- AMD Ryzen 7 8845H, Ubuntu/Linux, Python 3.11.17, okama 4.0.0, NumPy 2.4.6, pandas 3.1.0. The complete BLAS build configuration and platform are in the JSON files.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` were set only for benchmark processes. Machine-wide settings were not changed.
- Three offline assets, 252 monthly observations starting in March 2000, NumPy data seed 127 and weight seed 127. Real frontier constructors use synthetic asset objects; only currency lookup is replaced with an offline fixture. No API calls or network data.
- One warmup followed by five timed repetitions. The full public call is timed, including weight generation, statistics preparation and returned DataFrame construction. Input data/constructor setup is excluded. All individual timings are retained.
- Peak memory is Python/NumPy allocation tracking via `tracemalloc` during a separate warm call, including the returned table and weights. It is not total process RSS and excludes resident imports/input data.
- Monte Carlo sizes are exactly 128, 2048 and 8192. Natural three-asset grids at steps 1/15, 1/63 and 1/100 contain 136, 2080 and 5151 portfolios. A three-asset 8256-point grid would require step 1/127, which the existing minimum-step guard rejects; that guard remains intact.
- This is a shared development machine: OS scheduling, CPU frequency and other processes are uncontrolled. Absolute times and ratios vary. A separate alternating baseline/candidate experiment below limits drift between versions.

## Full-call timing and memory

Medians; milliseconds and MiB. The mean and CAGR variants share the same batched implementation.

| Method / return | Portfolios | Baseline ms | Candidate ms | Ratio | Peak MiB baseline / candidate |
|---|---:|---:|---:|---:|---:|
| Monte Carlo / mean | 128 | 48.623 | 1.610 | 30.2x | 0.117 / 0.058 |
| Grid / mean | 136 | 50.649 | 1.143 | 44.3x | 0.124 / 0.053 |
| Monte Carlo / mean | 2048 | 722.928 | 6.110 | 118.3x | 1.184 / 0.983 |
| Grid / mean | 2080 | 917.315 | 3.678 | 249.4x | 1.210 / 0.676 |
| Monte Carlo / mean | 8192 | 3347.399 | 18.535 | 180.6x | 4.542 / 3.082 |
| Grid / mean | 5151 | 1898.721 | 8.251 | 230.1x | 2.897 / 1.471 |
| Monte Carlo / cagr | 128 | 40.220 | 1.860 | 21.6x | 0.103 / 0.058 |
| Grid / cagr | 136 | 44.676 | 1.235 | 36.2x | 0.128 / 0.054 |
| Monte Carlo / cagr | 2048 | 966.706 | 7.023 | 137.7x | 1.217 / 0.983 |
| Grid / cagr | 2080 | 918.008 | 4.787 | 191.8x | 1.200 / 0.676 |
| Monte Carlo / cagr | 8192 | 4371.255 | 20.481 | 213.4x | 4.537 / 3.082 |
| Grid / cagr | 5151 | 2099.149 | 8.206 | 255.8x | 2.891 / 1.471 |

## Alternating baseline/candidate confirmation

The benchmark loads the actual single-period class from the baseline Git revision into a separate module, initializes identical offline inputs, and alternates the order of old/new calls. Each call resets the same weight seed. Financial results and complete output frames are compared at `rtol=atol=1e-12`. One warm pair is excluded; five measured pairs follow.

| Monte Carlo portfolios | Baseline ms | Candidate ms | Ratio |
|---|---:|---:|---:|
| 1 | 1.769 | 1.792 | 0.99x |
| 8 | 6.603 | 1.310 | 5.04x |
| 128 | 61.903 | 1.723 | 35.94x |
| 2048 | 973.308 | 10.792 | 90.19x |
| 8192 | 3191.417 | 20.696 | 154.21x |

For one portfolio the measured difference is about 0.02 ms, within timing noise; no scalar fallback is justified by this experiment. Eight portfolios already benefit. For 8192 portfolios, the alternating confirmation improves about 154x, from 3.191 s to 20.696 ms. This is a claim about these single-period calls on this input/environment, not the whole library or an API request. It is unrelated to the issue's synthetic 20x proxy target.

## Phase costs and chunking

For 8192 weights, separately timed candidate primitives have these medians (not an additive decomposition of the full call; cache/warmup and code shape differ):

- weight_generation: 16.015 ms.
- statistics_preparation: 0.206 ms.
- batch_calculation: 0.335 ms.
- output_construction: 0.399 ms.

A chunk contains at most 4096 weight vectors. The largest additional matrix products have shape `chunk_size × assets`, not `periods × portfolios`; statistics require `assets × assets` covariance storage. Generated weights, retained chunk tables, concatenation and final output still cost O(portfolios × assets). Smaller chunks reduce temporary matrix storage but add stacking/DataFrame/concatenation overhead. Tests compare chunk sizes 1, 7 and 4096, including partial final chunks; chunk size is internal and not a new public tuning parameter.

## Compatibility and checks

- Both public methods and both return modes are compared to the original scalar financial helpers. Tests cover 2, 3 and 32 assets, constrained/default bounds, ticker/name/local-name labels, duplicate labels, pairwise missing-data covariance, constant returns, and output column/row order.
- Baseline/candidate full-call benchmark outputs have maximum absolute difference 1.78e-14; all comparisons pass `rtol=atol=1e-12`, including unchanged weights. Floating-point operation ordering can change the last few digits.
- Existing validation is preserved, including the grid guard, invalid `kind`, empty Monte Carlo requests and the existing mixed-case `MEAN` KeyError. Uppercase `CAGR` still works. Negative variance from an indefinite pairwise covariance retains the scalar `ValueError` instead of silently emitting NaN.
- TDD RED demonstrated 19 covariance calls for 19 Monte Carlo points and 24 calls for a 24-point bounded grid, against the required single preparation. The new implementation passes that requirement and the numerical checks.
- Targeted frontier tests complete without warnings. The full-suite baseline passed 518 tests with 3 skips and 371 pre-existing pandas warnings; final full-suite/Ruff results are recorded in the delivery notes below.

## Reproduction

Run from the repository root using the project uv environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m analytics.benchmark_frontier_batching --output tmp/frontier-candidate.json --repeats 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m analytics.benchmark_frontier_batching --output tmp/frontier-paired.json --paired-base b33ca12bdd81b4107c2de484fff581eda1023ce1 --repeats 5
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The first command profiles both frontier classes and writes timing/allocation JSON plus output arrays. The second directly reproduces the old/new comparison from the Git baseline without switching the working tree. For the full historical baseline table, run the first command with baseline library files and the same benchmark script. The accompanying evidence retains [baseline distributions/profiles](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/frontier-cpu-batching/baseline.json), [candidate distributions/profiles](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/frontier-cpu-batching/candidate.json) and [alternating comparison](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/frontier-cpu-batching/paired.json). The benchmark is at [analytics/benchmark_frontier_batching.py](https://github.com/mbk-dev/okama/blob/v4.0.1/analytics/benchmark_frontier_batching.py).

## Delivery notes

Final verification: `poetry run pytest -q` completed with **547 passed, 3 skipped**, 371 existing pandas deprecation warnings (the clean baseline also had 371); the new frontier test modules produce no warnings. Both `poetry run ruff check .` and `poetry run ruff format --check .` pass, as does `git diff --check`. Tests run in the Poetry Python 3.11.17 environment, the project minimum version.

Independent read-only whole-change review completed with no significant findings. The coordinator verified the final source/diff digests and benchmark equivalence; the reviewer did not rerun tests. Earlier external exploration/review attempts ended at a turn cap/timeout without a final verdict and were not counted as accepted reviews. Their logs are retained privately.

The implementation and evidence were subsequently merged in [PR #134](https://github.com/mbk-dev/okama/pull/134) and released in [okama 4.0.1](https://github.com/mbk-dev/okama/releases/tag/v4.0.1). The measurements above describe the original development-stage runs.
