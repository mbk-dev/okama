# Poetry to uv migration validation

The library uses standard `[project]` metadata, four dependency groups and
`uv_build`. The local contributor interpreter is Python 3.14; the supported
library range remains Python 3.11–3.14. No release has been uploaded.

The release credential wrapper is covered by nine synthetic-token tests and never
prints the token. A fresh-shell integration probe verified the release notebook
command selects the current worktree kernel. No real token was read or changed
during this migration.

## Checks completed locally

- Full library suite on Python 3.11, 3.12, 3.13 and 3.14: 527 passed and 3
  benchmark tests skipped on each version. Existing pandas warnings remain;
  numerical library code and the 30 packaged source files are unchanged.
- All 11 example notebooks passed on Python 3.14 (305 s) and 3.12 (214 s).
  Kernels were created under temporary prefixes; user-global kernels were preserved.
- Notebook bootstrap: RED reproduced 11 local environments invoking an installer;
  GREEN passed all 22 local/Colab-mode cases without making real pip calls.
- Sphinx HTML build succeeded with 256 warnings, predominantly existing duplicate
  API descriptions. The updated installation text also built successfully in a subsequent incremental
  run (510 warnings after cached object re-registration). Neither build failed.
- Direct wheel and sdist builds passed Twine checks. Standard pip on Python 3.11
  installed both artifacts, built a wheel from the sdist, and imported the installed
  library outside the source checkout. Runtime requirements and Python constraints
  match the Poetry baseline. License metadata uses SPDX MIT and includes LICENSE.txt.
- Fresh no-lock installation, import, pytest and Sphinx command launch passed with
  Poetry absent from PATH. RTD's explicit-interpreter docs-group install resolves.
- Infra setup: 51 contract tests passed, migrated and legacy smoke worktrees both
  installed on Python 3.14, and repeated uv setup reused the environment. The
  installed release is `52fe640fd40bd337`; only the okama selector and runner changed
  relative to the previous installed configuration. Other project settings were preserved.

The infra source changes are maintained in the separate `okama-uv-setup` worktree.
These results were collected before repository publication; delivery is recorded
in Git history and the final task report. Hosted CI/RTD had not run at validation
time. An actual Colab session
has not been checked; local Python 3.12 execution and the simulated Colab bootstrap
are separate evidence, not a claim of hosted Colab acceptance.

## Final fresh-environment timing

`benchmark.py` uses three unique directories per tool, no lockfiles, all groups,
and existing independent host caches. Environment creation, dependency resolution,
installation and the editable package build are timed; import/inventory probes are
outside the timer. The legacy Poetry manifest is retained as `legacy-pyproject.toml`.

| Tool | Median |
|---|---:|
| Poetry 2.5.1 | 9.024 s |
| uv 0.12.23 with uv_build | 0.216 s |

The ratio is 41.8x (about 8.8 seconds saved per environment). All 167 package
versions match, excluding Poetry's pip seed. See `final-benchmark-results.json`
for every timing and inventory. This is a warm-cache result on a busy Valinor,
not a controlled empty-cache download benchmark or a total Orca startup estimate.
The earlier experiment used poetry-core for both tools and is retained separately
in `initial-*` files; its original scratch scripts/logs remain under
`tmp/poetry-uv-bench/`. The final benchmark is reproducible with:

```bash
python analytics/poetry_uv_migration_20261026/benchmark.py
uv run pytest analytics/poetry_uv_migration_20261026/test_notebook_bootstrap.py -q
```

Raw integration logs, XML and artifacts are under `tmp/uv-migration/`; compact
results, hashes and inventories are retained here. Generated environments and
caches are not versioned.

## Independent review

Grok 4.6 was admitted with a fresh 66% usage reading. Its partial review identified
release-shell kernel state and token handoff concerns; both were verified and
fixed, with targeted tests and full compatibility-suite reruns. The review was
bounded and timed out before a complete final assessment, so this is not a claim
of a clean independent whole-branch review.

## Release-skill instruction check

The lexical checker reports false conflicts on both the original and migrated
release skill: for example, limiting a tag push overlaps the rule about naming
classes in release prose. `reviewed-skill-overlaps.json` records the exact skill,
rule and directive hashes and a reason for each manual review. No global rules
or checker implementation were modified. To reproduce the scoped PASS, copy the
unchanged canonical checker and these records into an isolated temporary directory,
then run the normal checker against the real AGENTS chain:

```bash
check_dir=$(mktemp -d)
cp ~/.agents/skills/skill-conflict-checker/scripts/check_skill_against_agents.py "$check_dir/"
cp analytics/poetry_uv_migration_20261026/reviewed-skill-overlaps.json "$check_dir/reviewed-overlaps.json"
python "$check_dir/check_skill_against_agents.py" --cwd "$PWD" .claude/skills/make-release/SKILL.md
```

Any subsequent edit of the skill or matched rule invalidates the corresponding
hash-bound review, so new overlaps remain checked.
