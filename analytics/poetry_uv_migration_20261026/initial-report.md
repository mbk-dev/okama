# Poetry versus uv: new okama environment

Python 3.14.4; Poetry 2.5.1; uv 0.12.23. Full main/test/docs/jupyter/dev dependencies, including editable okama. Measured sequentially on Valinor.

| Scenario | Poetry seconds | uv seconds | Speedup |
|---|---:|---:|---:|
| Warm cache, fresh-path-no-lock | 10.855 | 0.942 | 11.5x |
| Warm cache, no-lock | 19.964 | 0.442 | 45.2x |
| Warm cache, locked | 8.951 | 0.216 | 41.5x |

First no-lock runs (existing host caches, not controlled cold caches):
- poetry: 74.896 s
- uv: 52.989 s

All run timings:
- poetry no-lock trial 1, bytecode=False: 74.896 s
- uv no-lock trial 1, bytecode=False: 52.989 s
- uv no-lock trial 2, bytecode=False: 0.467 s
- poetry no-lock trial 2, bytecode=False: 24.214 s
- poetry no-lock trial 3, bytecode=False: 15.714 s
- uv no-lock trial 3, bytecode=False: 0.417 s
- uv locked trial 1, bytecode=False: 0.216 s
- poetry locked trial 1, bytecode=False: 8.951 s
- poetry locked trial 2, bytecode=False: 8.800 s
- uv locked trial 2, bytecode=False: 0.317 s
- uv locked trial 3, bytecode=False: 0.215 s
- poetry locked trial 3, bytecode=False: 9.817 s
- uv locked trial 4, bytecode=True: 5.053 s
- uv fresh-path-no-lock trial 4, bytecode=False: 1.068 s
- poetry fresh-path-no-lock trial 4, bytecode=False: 11.712 s
- uv fresh-path-no-lock trial 5, bytecode=False: 0.816 s
- poetry fresh-path-no-lock trial 5, bytecode=False: 9.997 s

All 17 installations succeeded with identical 167 package names and versions, excluding pip (Poetry-only environment seed). Import smoke passed for okama, pandas, scipy, pyarrow, arch, statsmodels, pytest, sphinx and notebook in every environment.

Every run started without .venv. Timings include environment creation, resolution when no lock, dependency install and editable project install. Inventory and import smoke are outside the timed section. Warm no-lock medians use trials 2 and 3; locked medians use 3 trials. Fresh-path runs use two additional uniquely named project folders and no lock, so editable project build caching cannot carry over between folders.

uv used an experimental PEP 621 manifest and standard dependency-groups, preserving every current dependency constraint and the poetry-core build backend. The production manifest and setup were not migrated.

Existing independent package caches were retained. Other host installations were observed, and load was elevated. Initial runs had unequal pre-existing cache coverage; they do not establish controlled cold-download performance. Results measure Python environment setup, excluding Git worktree creation, CodeGraph and agent startup. No full library test suite was run.

Disposable benchmark environments were deleted after inspection. Reproduction script, exact inventories and logs are retained beside this report.

References:
- https://docs.astral.sh/uv/concepts/projects/dependencies/
- https://docs.astral.sh/uv/concepts/cache/
- https://python-poetry.org/docs/basic-usage/#installing-without-poetrylock
