# Poetry to uv Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task; use superpowers:subagent-driven-development only if delegation is selected and the subscription routing policy admits it. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate okama's development, CI, documentation, notebooks, packaging and worktree setup from Poetry to uv while preserving library compatibility.

**Architecture:** Use PEP 621 metadata, PEP 735 dependency groups and `uv_build` for the existing flat `okama/` package. Keep environment management isolated per worktree, local lockfiles ignored, and installer choice compatible with old branches until their migration.

**Tech Stack:** Python 3.11–3.14, uv, uv_build, pytest, Ruff, Jupyter/nbmake, Sphinx, GitHub Actions, Read the Docs, external project-setup.

**Spec:** [Migration design](../specs/2026-10-26-poetry-to-uv-design.md).

**Execution evidence:** [Local validation and remaining acceptance](../../../analytics/poetry_uv_migration_20261026/README.md). The release flow also gained `scripts/with_pypi_token.py`, with synthetic-token tests, after review identified shell-state coupling.

## Global constraints

- Preserve `requires-python = ">=3.11,<4.0.0"`; local `.python-version` stays `3.14`, notebook/Colab compatibility includes Python 3.12.
- Preserve package version `4.0.0` during migration preparation, public APIs, runtime dependency constraints and existing release gates.
- Runtime dependencies remain separate from `test`, `docs`, `jupyter` and `dev`; all four groups remain enabled by default.
- Keep both `uv.lock` and legacy `poetry.lock` ignored; no committed library lockfile or mandatory fresh-clone `--locked`/`--frozen`.
- All project documentation, comments and code introduced during implementation are in English.
- Follow RED → verified RED → GREEN for any new executable helper or runner logic. Configuration/content edits require parsing and operational checks, not redundant unit tests.
- Run both `uv run ruff check .` and `uv run ruff format --check .` for implementation changes. Run the library suite when executable library code changes; the migration additionally requires the compatibility matrix and packaging checks below.
- Plan preparation does not authorize implementation, commits, pushes, a release upload or modification of other live worktree environments.

## Review focus

1. A new clone without a lock must work; normal `uv run` must not unexpectedly remove dependencies needed by docs or notebooks. Covered by Tasks 1 and 6.
2. Python selection must not silently run every CI job on `.python-version`'s 3.14. Covered by Task 4's interpreter assertions and matrix.
3. Local notebook installation cells must not replace editable okama with PyPI okama; an uv environment may not contain pip. Covered by Task 3's fresh-kernel execution and provenance checks.
4. Editable installation is insufficient proof of distributable correctness: wheel and sdist need all modules, metadata and license, and imports outside the source checkout. Covered by Tasks 1 and 6.
5. A global setup update must not break older branches, reuse another worktree's environment or retarget user kernels. Covered by Tasks 5 and 6.

## Inventory from tracked files, 2026-10-26

Ten current files contain Poetry references:

| File | Required treatment |
|---|---|
| `pyproject.toml` | Replace metadata, groups and build backend |
| `.github/workflows/tests.yml` | Replace Poetry installation and commands; preserve Python matrix |
| `.travis.yml` | Migrate remaining test/coverage workflow |
| `README.md` | Replace development installation instructions |
| `CONTRIBUTING.md` | Replace lint, pre-commit and contributor commands |
| `docs/index.rst` | Replace duplicated development installation instructions |
| `AGENTS.md` | Replace environment rules, commands and metadata location |
| `.claude/skills/make-release/SKILL.md` | Migrate dependency refresh, tests, kernels, versioning, build, authentication and uploads |
| `tests/portfolio/dcf/test_mc_engine_benchmark.py` | Update the manual benchmark command in its docstring |
| `.gitignore` | Add `uv.lock`; retain the legacy lock ignore |

Nine older files in `docs/superpowers/plans/` and one `CHANGELOG.md` entry also mention Poetry. Assess whether any old plan remains executable work before labeling it historical; do not mechanically rewrite completed history. The new design and plan necessarily mention migration terminology.

Additional relevant files without Poetry references: `requirements.txt`, `docs/requirements.txt`, `.readthedocs.yml`, `docs/readthedocs.yaml`, `examples/README.md`, `.python-version`, `.pre-commit-config.yaml`, `.github/workflows/ruff.yml`, `.github/workflows/security-audit.yml`, `.github/workflows/codeql-analysis.yml`, `.agents/skills/create-readme/SKILL.md`, `CLAUDE.md` and the README-skill symlink.

All 11 tracked `examples/*.ipynb` notebooks contain a `%pip install --quiet -U okama` cell; none contains Poetry in cell source. Their metadata uses machine-specific or dated kernels, including `py13dev`, `py13` and `py313`.

External integration: `project_setup/profiles.json`, `project_setup/runner.py` and `project_setup/test_runner.py` in `~/projects/ops/infra`. The active installed setup is a versioned release, not the editable source of truth. Re-read its current launcher and source diff before changing it; another task can update it meanwhile.

## Task 1: Standardize package metadata and adopt uv

**Files:** `pyproject.toml`, `.gitignore`, `requirements.txt`, `docs/requirements.txt`; packaging evidence under `analytics/poetry_uv_migration_20261026/`.

**Produces:** one canonical metadata/dependency definition, working `uv sync`, wheel/sdist artifacts and their metadata/file inventories.

- [x] Capture a baseline build from the unchanged Poetry configuration into an isolated output directory. Record normalized `Name`, `Version`, `Requires-Python`, `Requires-Dist`, authors, URLs, classifiers, README content, license and module paths. Keep the benchmark scripts/results reproducible under `analytics/poetry_uv_migration_20261026/`; do not commit virtualenvs or bulky caches.
- [x] Move metadata into `[project]`, translate author records and URLs into their standard fields, and declare `license = "MIT"` plus `license-files = ["LICENSE.txt"]`. Preserve the original values and all runtime dependency constraints.
- [x] Move all four tool groups to `[dependency-groups]`; set `[tool.uv] default-groups = ["test", "docs", "jupyter", "dev"]` so plain `uv sync` preserves today's full development environment.
- [x] Replace poetry-core with `build-backend = "uv_build"`; use the uv_build 0.12 compatibility series (`>=0.12,<0.13`) with the currently measured uv 0.12.23. Set `[tool.uv.build-backend] module-name = "okama"` and `module-root = ""`. Validate this range resolves and works on Python 3.11 before accepting it. A tool-version change must be explicit and rechecked, not hidden in migration.
- [x] Keep direct runtime ranges mirrored in `requirements.txt`. Resolve existing drift explicitly: its `pandas` entry lacks the manifest's `>=2.0.0` lower bound. Do not replace this file with a frozen transitive export.
- [x] Reconcile the docs group with `docs/requirements.txt`, preserving `sphinx<9`, `docutils<0.22`, `nbsphinx-link>=1.3.0`, `pandoc>=1.1.0` and the docs IPython requirement. Keep these as documentation dependencies, not new runtime requirements. Record the existing drift and align both entry points.
- [x] Ignore `uv.lock` while retaining the `poetry.lock` ignore. Do not remove locks or `.venv` directories belonging to existing worktrees.
- [x] Run `uv sync --python 3.14` from an isolated copy without `.venv`, `uv.lock` or `poetry.lock`; verify editable `okama.__file__` points into that copy and `okama.__version__` matches `[project].version`. Run `uv pip check`.
- [x] Run `uv build --no-sources --out-dir <isolated-artifact-dir>`. Compare baseline/new metadata and package files, allowing documented packaging-format changes such as license metadata normalization. Verify no test, docs, temporary or credential files enter the wheel. Rebuild a wheel from the produced sdist and compare its package modules with the directly built wheel.

## Task 2: Update installation and contributor documentation

**Files:** `README.md`, `CONTRIBUTING.md`, `docs/index.rst`, `examples/README.md`, `tests/portfolio/dcf/test_mc_engine_benchmark.py` (docstring only), `CHANGELOG.md`.

**Consumes:** Task 1's default groups, metadata and supported Python range.

- [x] Document installing uv, cloning the existing development branch, `uv sync`, `uv run python`, `uv run pytest tests -q`, `uv run ruff check .`, `uv run ruff format --check .`, dependency additions with `uv add` and group additions with `uv add --group <name>`.
- [x] Preserve the public `pip install okama` option and existing uv consumer instructions; end users must not need uv to install the library. Distinguish the library's Python 3.11 minimum from the local contributor pin of 3.14.
- [x] Replace Poetry contributor commands. Since pre-commit is already declared, use `uv sync --group dev` followed by `uv run pre-commit install`; do not tell contributors to re-add the dependency as initial setup.
- [x] Add local notebook instructions to `examples/README.md`: `uv sync`, `uv run jupyter lab`, editor selection of the worktree's `.venv`, Colab behavior and explicit kernel selection for nbmake. Link this section from the main README and docs rather than copying a long procedure.
- [x] Replace the manual benchmark docstring command with `OKAMA_BENCH=1 uv run pytest tests/portfolio/dcf/test_mc_engine_benchmark.py -s`.
- [x] Add one concise Unreleased changelog entry about developer tooling. Preserve historical Poetry changelog entries and completed plan command blocks.
- [ ] Run each newly documented setup/verification command in a disposable clone. Check relative links and agreement between README, Sphinx installation text and contributor instructions.

## Task 3: Make notebooks work locally and in Colab

**Files:** all 11 tracked `examples/*.ipynb`; notebook execution instructions in `examples/README.md` and the release skill.

**Produces:** portable notebook metadata and a bootstrap that preserves the local editable package.

- [x] Inventory the exact changed source cells by parsing notebook JSON; do not search or replace serialized outputs or images.
- [x] Replace the unconditional PyPI upgrade cell with an explicit Colab-only installation path. Keep `%pip` targeting the active Colab kernel and preserve Python 3.12 compatibility. In a local project kernel, skip installation and import the worktree package. Do not require uv inside Colab and do not assume local uv environments seed pip.
- [x] If bootstrap logic is introduced, first demonstrate the two required behaviors with executable cell checks: Colab requests installation in its current kernel; a non-Colab editable environment never invokes an installer or changes installed okama. Verify failure before the minimal change and success afterwards.
- [x] Normalize shared notebook kernelspec metadata to portable `python3` / `Python 3`. Do not invent a new recorded language version without actually executing the notebook on it. Preserve unrelated cells, results and embedded figures.
- [x] For tests, create a disposable kernelspec pointing at the exact test environment's interpreter, with a temporary `JUPYTER_PATH`. Execute using `uv run --no-sync pytest --nbmake --nbmake-kernel=python3 examples -q`; verify the kernel interpreter and editable package path before running. Avoid persistent user-global kernel changes for test execution.
- [ ] Run all 11 notebooks in local Python 3.14 and Python 3.12 environments, against the worktree installation. These are integration tests and can use the existing data API; report real API failures separately from tooling failures. Verify the Colab branch in an actual fresh Colab Python 3.12 kernel; if that service cannot be accessed, retain this as an explicit unverified acceptance item rather than treating local 3.12 as full Colab proof.

## Task 4: Migrate CI and documentation builds

**Files:** `.github/workflows/tests.yml`, `.travis.yml`, `.readthedocs.yml`, `docs/readthedocs.yaml`, `docs/requirements.txt`; review `.github/workflows/ruff.yml`, `.pre-commit-config.yaml` and `docs/conf.py`.

**Produces:** version-correct test jobs and documentation installation without Poetry.

- [x] Use `astral-sh/setup-uv` at a reviewed pinned action revision and uv version. Preserve the 3.11, 3.12, 3.13 and 3.14 matrix and pass its selected interpreter explicitly to uv, overriding `.python-version`. Remove Poetry-only environment settings.
- [x] Install test jobs with `uv sync --python <matrix-version> --no-default-groups --group test`. Run `uv run --no-sync python -c ...` to assert the actual interpreter, print package version/provenance and check import; then `uv run --no-sync pytest tests -q`. `--no-sync` prevents uv run from restoring every default group during the job.
- [x] Cache uv's package cache using pyproject content and interpreter information, without requiring a tracked lockfile. Demonstrate a clean checkout with no lock successfully installs. Do not share `.venv` across matrix jobs.
- [x] Migrate Travis's installer, tests and coverage upload to uv too. Use `uv run --with pytest-cov --with coveralls ...` for CI-only coverage tools instead of introducing them as library runtime dependencies. Verify the selected Travis Python interpreter explicitly; do not assume Python 3.14 is offered by its existing Focal image. If Travis cannot run the declared matrix, document and settle that limitation rather than silently deleting the CI workflow.
- [x] For Read the Docs, retain its managed virtualenv and Sphinx build. Install the chosen uv version in a supported build job, then install the package and docs group into the explicit RTD interpreter with `uv pip install --python "$READTHEDOCS_VIRTUALENV_PATH/bin/python" . --group docs`. This avoids creating an unrelated `.venv` which RTD's Sphinx command would not use. Replace conflicting automatic installation settings in the same change.
- [x] Determine which RTD config path the actual project selects. The root config currently targets 3.14 and `docs/readthedocs.yaml` targets 3.12; align their installation procedure and remove contradictory guidance without silently changing the configured path or dropping coverage. Keep `docs/requirements.txt` usable as the compatibility installation route.
- [ ] Run `uv run sphinx-build -b html docs docs/_build`; compare warnings with the baseline and fix new migration-induced failures. Verify a hosted RTD preview/build after the authorized branch is published; record its URL and status.
- [x] Review Ruff/pre-commit, security-audit and CodeQL workflows for metadata/install assumptions. The latter workflows contain no direct Poetry commands today; do not modernize unrelated action versions or triggers as part of this migration.

## Task 5: Update release instructions and worktree setup

**Files:** `AGENTS.md`, `.claude/skills/make-release/SKILL.md`; external `infra/project_setup/{profiles.json,runner.py,test_runner.py}` in its own task worktree.

**Produces:** consistent agent/release commands and branch-compatible bootstrap.

- [x] Apply `instruction-files-maintenance` when updating `AGENTS.md`: uv environment/add/run/lint rules; Python minimum comes from `[project].requires-python`; retain TDD, clean-output requirements, bounded retry policy and runtime requirements mirroring. `CLAUDE.md` already imports `AGENTS.md`; verify the link rather than duplicating rules.
- [x] Apply skill maintenance and `skill-conflict-checker` for the release skill. Its actual file is in `.claude/skills/make-release/`; do not create a second competing copy or globally reinstall tools.
- [x] Replace dependency refresh with `uv lock --upgrade` plus `uv sync`; retain dependency-change review. Replace build with `uv build --no-sources`, and define explicit current-version wheel/sdist paths for `uv publish`. Do not publish during migration validation.
- [x] Replace Poetry environment switching with separate uv environments, including `UV_PROJECT_ENVIRONMENT=<isolated-minimum-python-env>` for the required Python 3.11 release gate. Assert its interpreter and package source; never swap the active worktree's 3.14 `.venv` in place.
- [x] Keep unit tests, notebook tests, minimum-Python tests, both Ruff checks, Sphinx and hosted documentation verification as release gates. After changing `[project].version`, sync/reinstall the editable package and verify `importlib.metadata.version("okama")` and `okama.__version__` match it before kernel registration or building artifacts.
- [x] Replace Poetry-keyed test kernels with disposable kernels as in Task 3. Preserve existing per-release `okama<X.Y.Z>` naming for user-requested persistent kernels, verify their executable paths, and never remove old kernels automatically.
- [x] Replace Poetry-specific keyring troubleshooting and credential preflight with uv's supported credential mechanism. Discover the existing authorized PyPI token without printing it; explicitly pass it through a supported environment/secret source such as `UV_PUBLISH_TOKEN`. Do not assume uv reads `~/.config/pypoetry/auth.toml`, rotate credentials or redesign publishing authentication. Retain exact-artifact hash checks and partial-upload recovery rules.
- [x] Reconcile the prior local Python-3.14 profile change with the current infra source and installed launcher before editing. Change bootstrap to use uv for target manifests with standard metadata/groups; keep the old Poetry path for Poetry-format branches. This compatibility guard is required because the global profile also serves older live worktrees.
- [x] If selecting commands requires new runner logic, add failing tests in `project_setup/test_runner.py` first: migrated okama selects uv and Python 3.14; legacy okama selects Poetry; invalid manifests fail before installation; other project profiles remain unchanged. Use a narrowly scoped selection mechanism, not arbitrary manifest command execution.
- [x] Update the setup fingerprint to include `.python-version` and optionally `uv.lock` for the uv path, retaining appropriate legacy inputs. Preserve the existing per-worktree environment, CodeGraph, path validation and provenance checks. Verify repeated unchanged setup uses its cache.
- [ ] Install the setup change with the existing versioned installer only after target manifest compatibility is demonstrated. Check the active launcher and `project-setup plan`, then run actual setup on one migrated disposable worktree and one legacy worktree. Preserve live trees and environments; complete source delivery through the infra project's normal authorized route.

## Task 6: Final migration acceptance and delivery evidence

**Depends on:** Tasks 1–5. Update the evidence under `analytics/poetry_uv_migration_20261026/` and link it from the final migration report.

- [x] Create an isolated checkout with no lockfiles and no `.venv`, and use a restricted PATH without the Poetry executable. Complete `uv sync`, import checks, test invocation and Sphinx setup. This proves the Poetry CLI is no longer an implicit prerequisite.
- [x] Confirm unit-test jobs pass on Python 3.11–3.14, notebook checks from Task 3 pass, and both `uv run ruff check .` and `uv run ruff format --check .` pass. Respect the project's maximum two fix/rerun cycles for test failures; preserve unresolved evidence.
- [x] Install the built wheel and the sdist into independent environments outside the repository using a standard pip frontend as well as uv. Assert version, `Requires-Python`, runtime dependency metadata and imports. Include a Python 3.11 install/build check and verify a wheel rebuilt from sdist contains the same library modules. Run `uvx --from twine twine check <exact-wheel> <exact-sdist>` without uploading.
- [x] Repeat the fresh-tree installation benchmark with the final backend: at least three sequential, uniquely named empty environments per tool where feasible, alternating order and keeping identical group selection. Report versions, package inventory parity, warm-cache medians/ranges and any tool-load interference. Do not claim controlled cold-cache results unless both tools actually use comparable isolated empty caches; never purge global cache roots.
- [x] Search tracked textual files case-insensitively for `poetry`, `poetry-core`, `POETRY_`, `pypoetry`, legacy kernel names and stale interpreter-selection instructions. Parse notebook cell source and kernelspec separately. Classify remaining hits: historical record, migration documentation, credential migration explanation or required legacy bootstrap. Current normal-use commands must all use uv.
- [x] Confirm `.agents/skills/create-readme/SKILL.md`, its `.claude` symlink, docs Makefiles and remaining workflows do not regenerate or invoke the old setup. If no dependency exists, record the review without unrelated edits.
- [ ] Record local checks separately from hosted CI, RTD and real Colab acceptance. Provide exact artifact paths and hashes, remaining checks, source branches and fresh-tree timing. Check Git locks, preserving foreign/live holds.
- [ ] After explicit delivery authorization, commit only the scoped library and infra changes, publish through their respective project routes, verify actual default branches and installed setup, and synchronize retained owned trees. A successful merge alone does not complete notebook/RTD/setup acceptance. Do not release okama to PyPI merely to deliver this tooling migration.

## Completion criteria

The migration is complete when the library, contributor instructions, all 11 notebooks, CI, docs builds, release procedure and setup agree on uv; package compatibility and artifact checks pass; old branches retain a working bootstrap; and remaining historical Poetry references are explicitly accounted for. Until hosted checks run, report local completion and the exact outstanding acceptance items separately.
