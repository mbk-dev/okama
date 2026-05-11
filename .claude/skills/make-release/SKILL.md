---
name: make-release
description: Cut a new release of the okama Python package — runs the full release workflow (poetry update, pytest, nbmake notebook tests, ruff, sphinx docs, version bump, CHANGELOG, git tag on master, GitHub release, PyPI publish, post-release dev sync). Use whenever the user asks to release okama, ship a new okama version, publish okama to PyPI, cut a tag, or says any of "make release", "release okama", "release the library", "сделай релиз", "выпусти версию", "релиз окамы", "опубликуй на pypi" — even if they don't spell out every step. The skill enforces confirm gates before any irreversible action (push, merge, tag push, GitHub release, PyPI publish).
---

# make-release — okama release workflow

This skill ships a new release of the `okama` Python package end-to-end. The workflow is intentionally rigid because release operations are irreversible (PyPI versions cannot be re-uploaded, pushed tags propagate to CI and Read the Docs). Run the phases **in order** and **stop on the first failure** unless the failure is auto-fixable by `ruff --fix` / `ruff format`.

The user has approved confirm gates at four points — these are non-negotiable, even if the user says "go ahead" earlier. Pause and ask before each:
1. before `git push` (and `git push --tags`)
2. before merging `dev` into `master`
3. before creating the GitHub Release
4. before `poetry publish` to PyPI

All other steps run without prompting unless they fail.

## Working directory

The package lives at `/home/chilango/pyprojects/okama_projects/okama`. Run all commands from that directory. Verify with `pwd` if unsure — running poetry from a wrong directory silently uses the wrong env.

## Phase 0 — Preflight

Run these checks first. If any fails, stop and report — do not try to fix automatically.

```bash
# 1) On dev branch
test "$(git rev-parse --abbrev-ref HEAD)" = "dev" || echo "FAIL: not on dev"

# 2) Working tree clean
test -z "$(git status --porcelain)" || echo "FAIL: dirty tree"

# 3) gh CLI authenticated
gh auth status

# 4) PyPI token configured for poetry
poetry config pypi-token.pypi 2>/dev/null | grep -q . || echo "FAIL: no pypi token"

# 5) Read the Docs API token available in .env
grep -q '^READTHEDOCS_TOKEN=' .env 2>/dev/null || echo "WARN: no RTD token — Phase 10 will be skipped"
```

`.env` is gitignored. The RTD token lets Phase 10 query and poll Read the Docs builds via the v3 API. If it is missing, the skill falls back to checking only the public docs URL.

## Phase 1 — poetry update

```bash
poetry update
```

If anything in `poetry.lock` changes meaningfully, mention it in the release notes draft later.

## Phase 2 — Tests

Both unit tests (`pytest`) **and** notebook tests (`pytest --nbmake examples`) are mandatory before any version bump. Notebook tests are the only thing that catches API drift in the user-facing examples — never skip them, even for a "doc-only" release.

### 2a. Install current dev source into the env

`pytest --nbmake` runs each notebook against a registered Jupyter kernel, and that kernel imports `okama` from the env. Without an explicit install the kernel may import a stale build of okama and miss the very changes you are about to release. Run `poetry install` first so the env reflects the current `dev` branch:

```bash
poetry install
```

Confirm the installed version matches `pyproject.toml` (`Installing the current project: okama (<NEW_VERSION>)`).

### 2b. Resolve / register the Jupyter kernel

Derive the kernel name from the env's Python version — do **not** ask the user, and do **not** hardcode a kernel name. The convention is `okama_poetry<MAJOR>.<MINOR>` (e.g. `okama_poetry3.14` for Python 3.14.x):

```bash
PYVER=$(poetry run python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
KERNEL_NAME="okama_poetry${PYVER}"
```

Check whether that kernel is already registered against the poetry env:

```bash
poetry run python -c "from jupyter_client.kernelspec import KernelSpecManager; \
  ks=KernelSpecManager().get_all_specs(); print('FOUND' if '${KERNEL_NAME}' in ks else 'MISSING')"
```

If `MISSING`, register it from the poetry env so the kernel's `python` points at the okama venv:

```bash
poetry run python -m ipykernel install --user --name="${KERNEL_NAME}" --display-name="${KERNEL_NAME}"
```

If a stale kernel with a previous Python version exists (e.g. `okama_poetry3.13` after a 3.14 upgrade), leave it alone unless the user asks — old kernels do not break anything, but uninstalling someone else's tooling without consent does.

### 2c. Run the tests

```bash
poetry run pytest -n=auto
poetry run pytest --nbmake --nbmake-kernel="${KERNEL_NAME}" -n=auto examples
```

If tests fail: stop. Report the failures to the user. Do not attempt fixes — the user will decide. (Per `okama/AGENTS.md`, retry at most twice if the fix is obvious — but during a release, prefer to stop and let the user decide whether to defer.)

## Phase 3 — Lint & format

`ruff` issues are auto-fixable; apply fixes silently, then verify.

```bash
poetry run ruff check --fix .
poetry run ruff format .
poetry run ruff check .          # must exit 0
poetry run ruff format --check . # must exit 0
poetry run pre-commit run --all-files
```

If `ruff check .` still fails after `--fix`, or `pre-commit` reports issues that aren't auto-fixed: stop. Show the user the remaining diagnostics.

## Phase 4 — Docs build

Sphinx docs live in `docs/`. Regenerate `.rst` from sources and build HTML.

```bash
cd docs
poetry run sphinx-apidoc -o source/ ../okama
poetry run python -m sphinx -b html . _build/html
cd ..
```

Stop on any sphinx error. Warnings about missing references in third-party packages are usually OK; new warnings introduced by this release are not.

This phase only validates that docs build **locally**. Read the Docs is verified separately in Phase 10, after the tag is pushed (RTD auto-builds on tag).

## Phase 5 — Version bump

Read the current version from `pyproject.toml` (the `version = "X.Y.Z"` line under `[tool.poetry]` or `[project]`). Show it to the user and ask which bump to apply:

- **patch** — `X.Y.Z → X.Y.(Z+1)` (bug fixes, doc-only, dependency bumps)
- **minor** — `X.Y.Z → X.(Y+1).0` (new features, no breaking changes)
- **major** — `X.Y.Z → (X+1).0.0` (breaking changes)

The user picks. Then update `pyproject.toml` (single line edit). Do not edit any other version field — okama uses `pyproject.toml` as the single source of truth. The new version is `<NEW_VERSION>` for the rest of the workflow.

## Phase 6 — Release notes & CHANGELOG

Generate a draft from git history since the last tag, then have the user edit it.

```bash
LAST_TAG=$(git describe --tags --abbrev=0)
git log --no-merges --pretty=format:"- %s (%h)" "$LAST_TAG"..HEAD
```

Group commits by Conventional Commit prefix into the four sections used in `CHANGELOG.md`:

- `feat:` / `feat(...)` → **Added**
- `fix:` → **Fixed**
- `docs:` / notebook updates → **Docs**
- everything else (`chore:`, `style:`, `refactor:`) → omit unless user-visible

Match the style of the existing `## [2.0.1] - 2026-04` section in `CHANGELOG.md`: brief lead paragraph describing the theme of the release, then the section headers. Reference classes and methods by name (e.g. ``` `PortfolioDCF.wealth_index()` ```) — this is required by `okama/AGENTS.md`. Link notebooks under `/examples` with markdown links.

Show the draft to the user and let them edit before continuing. Insert the final block at the top of `CHANGELOG.md` under the title.

Then build the GitHub Release body — usually identical to the new CHANGELOG section, but follow the format of <https://github.com/mbk-dev/okama/releases/tag/v2.0.0> if the user wants something richer. Save it to a temp file (e.g., `/tmp/release_notes_v<NEW_VERSION>.md`) for `gh release create --notes-file` later.

## Phase 7 — Sync requirements.txt

If `poetry update` (Phase 1) changed any dependency versions, sync `requirements.txt`. Per `okama/AGENTS.md`, every dependency change must be reflected in both `pyproject.toml` and `requirements.txt`.

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

If the export plugin is not installed, fall back to manually updating the changed lines.

## Phase 8 — Commit on dev, push, merge into master, tag

This is the git-flow phase. The user approved this exact order: commit on dev → push dev → merge into master → tag on master → push tags.

```bash
# Stage & commit on dev
git add pyproject.toml CHANGELOG.md requirements.txt docs/
git commit -m "chore(release): prepare v<NEW_VERSION>"
```

**CONFIRM GATE 1** — before `git push`. Show `git log --oneline @{u}..` so the user can see what they are about to push. Only push after explicit approval.

```bash
git push
```

Wait for CI to go green (Travis CI + flake8/ruff). If the user says "skip waiting", note it in the report but proceed.

**CONFIRM GATE 2** — before merging into master. Show the diff range:

```bash
git log --oneline master..dev
```

After approval:

```bash
git switch master
git pull
git merge --no-ff dev -m "Merge dev for v<NEW_VERSION>"
git tag "v<NEW_VERSION>"
git describe --tags  # sanity check: should print v<NEW_VERSION>
```

**CONFIRM GATE 1 (again)** — before pushing master and the tag. Pushed tags are very hard to remove cleanly:

```bash
git push
git push --tags
```

## Phase 9 — GitHub Release

**CONFIRM GATE 3** — before creating the GitHub Release. Show the user the `--notes-file` content one more time.

```bash
gh release create "v<NEW_VERSION>" \
  --title "v<NEW_VERSION>" \
  --notes-file /tmp/release_notes_v<NEW_VERSION>.md \
  --target master
```

If the user wants to mark it as a pre-release, add `--prerelease`.

## Phase 10 — Read the Docs verification

The tag push in Phase 8 triggers a Read the Docs build for both the `latest` version (rebuilt on every master push) and the new version slug (e.g. `v2.0.2`). Verify the build succeeded **before** publishing to PyPI — a broken docs build usually means a real bug (import error, broken example) that PyPI users will hit too.

Load the token, then poll the API:

```bash
set -a; source .env; set +a   # exports READTHEDOCS_TOKEN

API="https://readthedocs.org/api/v3/projects/okama"
AUTH="Authorization: Token $READTHEDOCS_TOKEN"
SHA=$(git rev-parse HEAD)     # the merge commit on master

# Find the build for our commit; RTD usually creates it within 30s of the push
BUILD_ID=$(curl -s -H "$AUTH" "$API/builds/?limit=20" \
  | python3 -c "import json,sys,os; sha=os.environ['SHA']; \
       print(next((b['id'] for b in json.load(sys.stdin)['results'] if b['commit']==sha), ''))")
```

If `BUILD_ID` is empty, wait 30s and retry — RTD is sometimes slow to register the webhook. Give up after ~5 minutes and tell the user to check the dashboard manually.

Once the ID is found, poll until state is `finished`:

```bash
while :; do
  STATE=$(curl -s -H "$AUTH" "$API/builds/$BUILD_ID/" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state']['code'], d['success'])")
  echo "$STATE"
  case "$STATE" in
    "finished True")  echo "RTD build OK"; break ;;
    "finished False") echo "RTD build FAILED"; exit 1 ;;
  esac
  sleep 30
done
```

If the build fails: stop. Show the user `error` and the build URL (`https://app.readthedocs.org/projects/okama/builds/$BUILD_ID/`). Do not proceed to PyPI — fix the docs first, push a fix to master, wait for the new build to pass, then resume from Phase 11.

If the build passes, also do a quick public-docs sanity check:

```bash
curl -s -o /dev/null -w "%{http_code}\n" "https://okama.readthedocs.io/en/latest/"
```

Should be `200`. Anything else (typically `404` or `5xx`) is a CDN issue worth noting but not blocking.

For the new tagged version (`v<NEW_VERSION>`), RTD may keep it inactive by default. If `https://okama.readthedocs.io/en/v<NEW_VERSION>/` returns 404 even after the build, mention it in the final report — the user activates the version in the RTD dashboard.

## Phase 11 — PyPI publish

**CONFIRM GATE 4** — before `poetry publish`. PyPI does not allow re-uploading the same version, even if you delete it. Once this command succeeds, `<NEW_VERSION>` is permanent.

```bash
poetry publish --build
```

Verify by visiting <https://pypi.org/project/okama/> and confirming the new version is listed. The skill cannot do this automatically, but `pip index versions okama` (if available) can be a quick check.

## Phase 12 — Post-release

Sync `dev` with `master` via fast-forward (the user explicitly chose this over delete-and-recreate).

```bash
git switch dev
git merge --ff-only master
git push
```

If `--ff-only` fails, dev has diverged from master post-merge — stop and ask the user how to handle it. Do not force-push, do not rebase without permission.

The Jupyter kernel was already registered in Phase 2b; no further kernel work is needed here.

## Reporting

At the end, give the user a one-paragraph summary:

- new version
- tag name
- GitHub Release URL (from `gh release view --json url -q .url`)
- PyPI URL
- whether `dev` is fast-forwarded
- anything that was skipped or that needs manual follow-up (Read the Docs build, etc.)

Do not list every command that ran — the user does not need a transcript.

## What NOT to do

- Do not run `git push --force` or `git push --force-with-lease` at any point in this workflow. If a push is rejected, stop and report.
- Do not amend a commit that has already been pushed. Make a new commit instead.
- Do not delete the `dev` branch (the user chose fast-forward over recreation).
- Do not skip pre-commit hooks with `--no-verify` (forbidden by the user's global rules).
- Do not bump dependencies "while we're at it" — the only dependency change in this workflow comes from `poetry update` in Phase 1.
- Do not modify `main_notebook.ipynb` (forbidden by `okama/AGENTS.md`).
- Do not write release notes or CHANGELOG entries in any language other than English (forbidden by `okama/AGENTS.md`).

## Recovery

If something fails between phases 8–12, the state is partially-released. Tell the user exactly what completed and what didn't, then ask before any cleanup. Common partial states:

- **Tagged locally, not pushed**: `git tag -d v<NEW_VERSION>` is safe.
- **Tag pushed, no GitHub Release**: `gh release create` can still run, or delete the remote tag with `git push --delete origin v<NEW_VERSION>` (ask first).
- **GitHub Release created, RTD build failed**: do not publish to PyPI. Fix the docs on master, RTD will rebuild automatically on the next push; once green, resume from Phase 11.
- **GitHub Release created, PyPI publish failed**: do not delete the GitHub Release — fix the PyPI issue (auth, network) and rerun `poetry publish --build`. PyPI will accept the same version only if the previous attempt did not actually upload.
- **PyPI publish succeeded, post-release sync failed**: the release is done; just finish Phase 12 manually.
