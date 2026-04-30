# okama — Development rules for AI agents

## environment
- project uses poetry for the envirobment & dependecies management
- new dependecy must be added in pyproject.toml and additionally to requirements.txt
- Use interpreter in poetry env (poetry run python ...).
- Always use "poetry add" instead of "pip install"

## After any code changes:
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

## Python style & modernization

- Target **Python 3.13+**. Write new code with modern syntax and avoid legacy forms:
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
- When referencing Jupyter Notebooks at /examples always add links

## Additional rules:
- Always write all code comments, docstrings, and documentation in **English**, even if the task description or existing code is in another language (e.g. Russian).
- Use type hints for all function parameters and return types.
- Use f-string formatting for all logging and print messages.
- When editing Jupyter Notebook examples in the `/examples` directory, ensure that the code examples are up-to-date with the current codebase in the Git branch.


