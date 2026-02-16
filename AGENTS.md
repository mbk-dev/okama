# okama — Development rules for AI agents

After any code changes:
1) Determine whether *executable Python code* was changed, not just comments or docstrings.
2) If executable code was changed — always run tests: `pytest -q`.
3) If only comments, docstrings or Jupyter Notebooks files were changed — do not run tests.
4) If test execution reveals any failures or errors, attempt to fix them and re-run the tests.
   Do not repeat this cycle more than 2 times.
   If tests are still failing after that, stop and report the remaining issues instead of continuing.

Additional rules:
- Always write all code comments, docstrings, and documentation in **English**, even if the task description or existing code is in another language (e.g. Russian).
- Do not make any changes to `main_notebook.ipynb`.
- Use type hints for all function parameters and return types.
- Use f-string formatting for all logging and print messages.
- When editing Jupyter Notebook examples in the `/examples` directory, ensure that the code examples are up-to-date with the current codebase in the Git branch.
