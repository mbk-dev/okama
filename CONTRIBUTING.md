# Contributing to Okama

Thank you for your interest in _okama_ and wanting to contribute. Your help is much appreciated.

Here are some guidelines for contributing.

Please refer to the [RoadMap](https://github.com/mbk-dev/okama/blob/master/README.md#roadmap) for a list of areas where _okama_ could benefit
from. In addition, the following is always welcome:

- Report bugs using [Issues](https://github.com/mbk-dev/okama/issues)
- Request new functionality and enhancement in [Discussions](https://github.com/mbk-dev/okama/discussions)
- Improve performance of existing code
- Help us to write tests. If you learn about quantitative finance and/or unit testing in python it's a good place to start. Test coverage of _okama_ could be verified at [Coveralls](https://coveralls.io/github/mbk-dev/okama?branch=dev).

## Guidelines

### Seek early feedback

Before you start coding your contribution, it may be wise to raise an issue on GitHub to discuss whether the contribution is appropriate for the project.

### Create a fork
First off, you should create a fork. Within your fork, create a new branch. Depending on what you want to do, choose one of the following prefixes for your branch:
- fix: <name of your fix> to be used for bug fixes
- feat: <name of new feature> to be used for adding a new feature

### Commit your changes
Make your changes to the code, and write sensible commit messages.

In general [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/) is recommended for the commits.

### Code style

To keep everything consistent, the project uses [Ruff](https://github.com/astral-sh/ruff)
for linting and formatting. The configuration lives in `pyproject.toml` under
`[tool.ruff]` — treat it as the authoritative style guide.

Before committing, run:

```bash
poetry run ruff check .
poetry run ruff format .
```

Fix every reported issue. If a warning is truly unavoidable, silence it with a
targeted `# noqa: <CODE>` comment on the offending line together with a short
rationale. Never disable rules globally or use a bare `# noqa`.

### Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to run Ruff automatically on every
`git commit`. The hook configuration is in [.pre-commit-config.yaml](.pre-commit-config.yaml)
and currently runs `ruff --fix` and `ruff-format` on the staged files.

**First-time setup** (once per clone):

```bash
poetry add --group dev pre-commit        # install into the project env
poetry run pre-commit install            # register the git hook
```

After this, `pre-commit` will run automatically on every `git commit` and
block the commit if any check fails. Ruff will auto-fix what it can — just
re-stage the modified files (`git add -u`) and commit again.

**Useful commands:**

```bash
# run all hooks against every file in the repo (not just staged ones)
poetry run pre-commit run --all-files

# run a specific hook
poetry run pre-commit run ruff --all-files

# update hook versions pinned in .pre-commit-config.yaml
poetry run pre-commit autoupdate

# bypass hooks for a single commit (discouraged — use only when justified)
git commit --no-verify
```

If a hook fails on CI but passes locally, make sure your local hooks are
installed and up to date (`pre-commit install && pre-commit autoupdate`).

### Testing

Any contributions **must** be accompanied by unit tests (written with `pytest`). These are very simple to write. Just find the relevant test file (or create a new one), and write some `assert` statements. If you need a data presets please use [tests/conftests.py](tests/conftests.py) fixtures. Tests should cover core functionality, warnings/errors (check that they are raised as expected), and limiting behaviour or edge cases.

### Documentation

We would appreciate if changes are accompanied by relevant documentation.

Inline comments (and docstrings!) are great when needed, but don't go overboard. A lot of the explanation can and should be offloaded to ReadTheDocs. Docstrings should follow [PEP257](https://stackoverflow.com/questions/2557110/what-to-put-in-a-python-module-docstring) semantically. Okama uses `sphinx` and "numpy" style generate Documentation automatically from docstrings.

### Create a Pull Request
Create a new [Pull Request](https://github.com/mbk-dev/okama/pulls). Describe what your changes are in the Pull Request. If your contribution fixes a bug, or adds a features listed under [issues](https://github.com/mbk-dev/okama/issues) as "#12", please add "fixes #12" or "closes #12". 

## Questions

If you have any questions related to the project, it is probably easiest to ask it in [Discussions](https://github.com/mbk-dev/okama/discussions).

## Bugs/issues

If you find any bugs, feel free to [raise an issue](https://github.com/mbk-dev/okama/issues) include as many details as possible and follow the following guidelines:

- Descriptive title so that other users can see the existing issues
- Operating system, python version, and python distribution (optional)
- _okama_ version you are using
- Minimal example for reproducing the issue
- What you expected to happen
- What actually happened
- error messages if applicable
