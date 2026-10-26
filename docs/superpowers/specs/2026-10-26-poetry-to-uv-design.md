# Poetry to uv migration design

Status: approved for implementation. Local migration and validation are complete; repository delivery, hosted CI/Read the Docs and actual Colab acceptance remain separate.

## Intent and evidence

Use uv for okama environment management, dependency resolution, builds and release uploads. Update the contributor experience, documentation, notebook execution and worktree bootstrap together.

The local experiment used Python 3.14.4, Poetry 2.5.1 and uv 0.12.23. Two new-path, no-lock installations with warmed caches took 10.00–11.71 seconds with Poetry and 0.82–1.07 seconds with uv, a median ratio of 11.5. All 167 project distributions and their versions matched; Poetry additionally seeded pip. The experiment used poetry-core, so repeat it after changing the build backend. It did not measure controlled empty-cache downloads or total Orca startup. Logs, inventories and reproduction scripts currently reside in `tmp/poetry-uv-bench/`.

## Approaches and recommendation

1. Replace the installer only, retaining Poetry metadata and poetry-core. Smallest immediate change, but leaves two tool conventions and does not complete the requested migration.
2. Adopt standard project metadata and dependency groups, uv commands, and the `uv_build` backend. Recommended: okama is pure Python with a single `okama/` package at the repository root. Verify artifact contents and metadata before accepting the backend change.
3. Adopt uv plus Hatchling. Technically viable, but introduces another packaging tool without a requirement for its additional build features.

Approach 2 was selected when the user approved execution of the migration plan.

## Target behavior

- `[project]` is the source of package metadata and runtime dependencies; `[dependency-groups]` contains `test`, `docs`, `jupyter` and `dev`.
- Keep `requires-python = ">=3.11,<4.0.0"`, classifiers 3.11–3.14, Ruff target `py311` and local `.python-version` of `3.14`.
- Keep all four groups enabled by default, matching current `poetry install`. CI and documentation builders select their required groups explicitly.
- Use `uv_build` with `module-name = "okama"` and `module-root = ""`. Preserve the published package name, version, metadata, license, package modules and installation through pip.
- Keep `uv.lock` local and ignored, preserving the existing library lockfile policy. Do not introduce `--locked` or `--frozen` on a fresh clone that has no lock.
- Keep `requirements.txt` as a runtime compatibility list of direct dependencies with ranges, not a pinned export of the development environment.
- Keep `docs/requirements.txt` as a compatibility entry point; preserve its existing documentation constraints and align them with the docs dependency group.
- Replace current contributor and release commands with uv. Historical changelog entries and completed plans retain their original commands, identified as historical rather than treated as the current workflow.
- Colab installs the public package into its current Python 3.12 kernel. Local notebooks and nbmake use the editable worktree installation and do not upgrade it from PyPI.
- Test notebook execution explicitly selects the current worktree's Python kernel. User-global kernels from other worktrees are preserved.
- Switch the external `project-setup` profile only after the target branch has the new manifest; keep a compatibility path for older Poetry-format branches.

## Scope and limits

Do not change public financial APIs, numerical behavior, supported Python versions, dependency version policy, release branch routing, release gates or authentication providers. Migration validation builds artifacts but does not publish a release. Do not remove Poetry globally or replace existing environments used by other worktrees.

The PyPI upload workflow must load the existing authorized credential into uv's supported authentication mechanism without exposing it; uv must not be assumed to read Poetry's credential store. Scope uploads to the two verified current-version artifacts rather than `dist/*`.

## Acceptance

Fresh-clone development, the Python 3.11–3.14 test matrix, local and Colab notebook paths, Sphinx/Read the Docs installation and packaging all work without the Poetry CLI. Wheel and sdist installation preserve package metadata and imports. Current operational instructions contain no Poetry commands except an explicitly documented legacy setup fallback or credential migration note. Fresh-tree performance is measured again after migration.

## Primary references

- [Dependency metadata and groups](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [uv build backend and flat layouts](https://docs.astral.sh/uv/concepts/build-backend/)
- [Jupyter integration and environment targeting](https://docs.astral.sh/uv/guides/integration/jupyter/)
- [GitHub Actions and Python matrix selection](https://docs.astral.sh/uv/guides/integration/github/)
- [Building and publishing](https://docs.astral.sh/uv/guides/package/)
- [Read the Docs build customization](https://docs.readthedocs.com/platform/stable/build-customization.html)
