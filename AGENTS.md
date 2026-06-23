# okama — Development rules for AI agents

## environment
- project uses poetry for the environment & dependencies management
- new dependency must be added in pyproject.toml and additionally to requirements.txt
- Use interpreter in poetry env (poetry run python ...).
- Always use "poetry add" instead of "pip install"

## Test-Driven Development (TDD)
Any change to production code (new feature, bugfix, refactor, behavior change) must follow TDD: **write a failing test first, then the minimal code that makes it pass**. This overrides the default "write code, then tests" workflow.

The required workflow is the `superpowers:test-driven-development` skill. Cycle: **RED → verify RED → GREEN → verify GREEN → REFACTOR**.

Rules for this repo:
- Tests run via: `pytest -q` (or `poetry run pytest -q` if not inside the poetry shell).
- Before writing code, see the test fail for a real reason (`AssertionError` / missing function), not a typo/import error.
- For bugfix: first a test reproducing the bug, then the fix. Without a reproducing test the bug is not considered fixed.
- One test = one behavior. Test name describes the behavior meaningfully, no `test1` / `test_works`.
- Real code instead of mocks wherever possible.
- After GREEN, run the full test suite of the file/module to make sure nothing broke; output must be clean (no warnings/errors).
- Exceptions where TDD can be skipped: only by explicit user request (one-time data migration scripts, generated code, throwaway prototypes). Notebooks — partial exception: cover the library code with tests, not the notebook rendering itself.

## Post-change checklist
1) Determine whether *executable Python code* was changed, not just comments or docstrings.
2) If executable code was changed — always run tests: `pytest -q`.
3) If only comments, docstrings or Jupyter Notebooks files were changed — do not run tests.
4) If test execution reveals any failures or errors, attempt to fix them and re-run the tests.
   Do not repeat this cycle more than 2 times.
   If tests are still failing after that, stop and report the remaining issues instead of continuing.
5) Before finishing any code change (including notebooks), run `poetry run ruff check .`
   and fix every reported issue. If a warning is truly unavoidable, silence it with a
   targeted `# noqa: <CODE>` comment on the offending line and include a brief rationale.
   Never disable rules globally or use a bare `# noqa`.

## Project structure

- `okama/` — library source code
  - `api/` — data API client
  - `common/` — shared utilities and helpers
  - `frontier/` — efficient frontier optimization
  - `portfolios/` — portfolio classes
- `tests/` — pytest test suite (mirrors `okama/` layout: `asset_list/`, `helpers/`, `portfolio/`)
- `examples/` — Jupyter Notebook examples (also published to Google Colab)
- `docs/` — Sphinx documentation source

## Code search & navigation (codegraph MCP)

This repo can be indexed by the **codegraph** MCP server (a symbol graph: functions, classes, methods, call edges). It is **optional local tooling** — its config (`.mcp.json`, `.codegraph/`) is git-ignored locally and not shipped with the package. When available, prefer it for *symbol-relationship* questions over plain text search; the two are complementary. The guidance below was validated by A/B runs.

**Use codegraph when the question is about symbol relationships:**

- **who calls a symbol** — `codegraph_callers` (returns the enclosing function/method, without import/comment noise).
- **what breaks if a symbol changes** — `codegraph_impact` (transitive; depth-2 may include structural neighbours — sanity-check relevance).
- **where a symbol is defined / find a symbol** — `codegraph_search`.
- **relationship path between symbols** — `codegraph_trace`.
- **quick map of a topic** (which classes/modules are involved) — `codegraph_context`.

This pays off here because the library is class-heavy (`AssetList` / `Portfolio` / frontier hierarchies) and several modules are large (`asset_list.py`, `portfolios/core.py`, `frontier/*` run 1200–1800 lines) — jumping by symbol beats reading whole files.

**Stay on Grep/Glob or read the file when:**

- **what a function calls (`codegraph_callees`)** — the graph does **not** capture pandas/numpy method chains (`df.resample().mean()`, `.loc[...]`), so it under-reports dependencies in this data-heavy codebase. Read the body or grep instead.
- **arbitrary text** — strings, config, docstrings, ruff `# noqa` markers; and when you need *every* literal match (codegraph ranks and truncates).
- **external consumers** — the index covers this repo only; it won't show callers in okama-dash, notebooks, or downstream code.

**Scoping text searches:** when looking for *production code only*, scope the search to the library dir — `rg <pattern> okama/` (or Grep with `path: okama/`) — instead of searching the whole repo; use `rg -g '!tests/'` for one-off exclusions. Do **not** add a global ignore file (`.rgignore` / `.ignore`) that hides `tests/` or `examples/`: ripgrep-based search would skip them silently, and this repo's TDD workflow depends on finding existing tests and fixtures ("no matches" must mean "no tests cover this", not "tests were excluded"). Repo-wide search is already fast (~0.03 s); scoping is about result noise, not speed.

Name note: CLI subcommands are `query` / `callers` / `callees` / `impact` / `context`; MCP tools carry the `codegraph_` prefix (`codegraph_search` ≡ CLI `query`). Rebuild a stale index with `codegraph sync` (or `codegraph index`).

## Python style & modernization

- Always write all code comments, docstrings, and documentation in **English**, even if the task description or existing code is in another language (e.g. Russian).
- Use type hints for all function parameters and return types.
- Use f-string formatting for all logging and print messages.
- **Minimum supported Python version is taken from `pyproject.toml`** (the
  `python = "..."` constraint under `[tool.poetry.dependencies]`). All library
  code must run unchanged on that minimum version. For example, if
  `pyproject.toml` declares `python = ">=3.11,<4.0.0"`, then Python 3.11 must be
  fully supported and no 3.12+ only syntax or stdlib features may be used in
  library code without a `sys.version_info` gate and a fallback.
- **Notebook examples in `/examples` additionally target Google Colab**, which
  currently ships Python 3.12 by default. Notebooks must run on Colab's 3.12
  unchanged. Do not use 3.13-only syntax or stdlib features in notebooks
  (e.g. PEP 695 `type` statement improvements, `typing` additions introduced
  in 3.13). If a 3.13-only construct is genuinely needed, gate it behind a
  `sys.version_info` check and provide a 3.12 fallback. When editing notebook
  examples, ensure that the code is up-to-date with the current codebase in the Git branch.
- When the `pyproject.toml` minimum is bumped, update this section accordingly
  and re-evaluate which modern-syntax features can be used unconditionally.
- Write new code with modern syntax and avoid legacy forms:
  - Use built-in generics: `list[int]`, `dict[str, Any]`, `tuple[int, ...]`
    instead of `typing.List` / `Dict` / `Tuple`.
  - Use union syntax `X | Y` and `X | None` instead of `typing.Union` / `typing.Optional`.
  - Prefer literals over constructor calls: `{}`, `[]`, `set()` — avoid `dict()`, `list()`
    when a literal works. In particular, never write `dict()` for an empty dict,
    and never wrap `kwargs`-style pairs as `dict(a=1, b=2)` when a literal is clearer.
  - Use `dict(zip(a, b))` instead of `{k: v for k, v in zip(a, b)}` (ruff C416).
  - Use set literals `{"a", "b"}` instead of `set(["a", "b"])` (ruff C405).
  - Never use mutable default arguments (`def f(x=[])` / `x={}`). Use `None` and
    initialize inside the function body (ruff B006). For simple fixed pairs like
    `figsize=(12, 6)`, prefer a tuple.
  - Keep `matplotlib` `bbox=` / style kwargs as dict literals, not `dict(...)` calls.
- **Ruff configuration** is in `pyproject.toml` (`[tool.ruff.lint]`, selecting `C,E,F,W,B`).
  Treat it as the authoritative style guide — if ruff is silent, the style is acceptable.
- **Jupyter notebooks** follow the same rules. Keep all `import` statements at the top
  of each code cell (ruff E402); put configuration calls (`plt.rcParams[...]`,
  `pd.set_option(...)`, `warnings.filterwarnings(...)`) *after* the imports.
- When refactoring a function to reduce complexity (ruff C901) would be risky, it is
  acceptable to add `# noqa: C901` on the `def` line rather than restructure working code.

## Preparing release notes
- When you write a release description, always include the names of the specific classes and methods when describing new features and bug fixes.
- When referencing Jupyter Notebooks at /examples always add links.

## Maintaining this file
This file is a stable reference, not a worklog. It follows a global
instruction-file maintenance policy: route each kind of information to its
proper home and link here instead of copying. Project specifics:
- **Task tracker** — GitHub Issues at `mbk-dev/okama`. Issue status and numbers
  live there, not in this file. Keep here only a short list of active directions
  if needed.
- **History of changes** — `CHANGELOG.md` plus git history. Commit hashes, merge
  dates, and "what was done when" go there, never into this file.
- **Reference data** (ad-hoc findings, per-agent notes, access details) — the
  project's Claude Code memory, not this file. Stable conventions that belong
  here (Python style, project structure) are already inlined above.
- **Drafts and scratch artifacts** — `tmp/` (gitignored); clean these up when the
  task is done rather than letting them accumulate.
