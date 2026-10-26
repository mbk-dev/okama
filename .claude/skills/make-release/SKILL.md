---
name: make-release
description: Cut a new release of the okama Python package — runs the full release workflow (uv lock --upgrade, pytest, nbmake notebook tests, ruff, sphinx docs, version bump, CHANGELOG, git tag on master, GitHub release, PyPI publish, post-release dev sync). Use whenever the user asks to release okama, ship a new okama version, publish okama to PyPI, cut a tag, or says any of "make release", "release okama", "release the library", "сделай релиз", "выпусти версию", "релиз окамы", "опубликуй на pypi" — even if they don't spell out every step. Publication follows the user’s authorization and requires verified release artifacts, CI and hosted documentation.
---

# make-release — okama release workflow

This skill ships a new release of the `okama` Python package end-to-end. The workflow is intentionally rigid because release operations are irreversible (PyPI versions cannot be re-uploaded, pushed tags propagate to CI and Read the Docs). Run the phases in order. Required failures block publication; diagnose and fix them within the authorized scope and the project’s retry limits.

Use the authorization already given in the session. A request to commit everything
and make a new release covers the release commits, normal pushes, dev/master
integration, release tag, GitHub Release and PyPI publication. Do not repeat
permission questions for these authorized steps. When publication has not been
authorized, finish the checks, notes and inspectable artifacts before requesting
approval for the specific external action. Destructive recovery needs its own scope.

## Working directory

The main checkout is `/home/chilango/projects/rs/okama-projects/okama`.
Use the task’s isolated worktree for preparation, tests and uv commands;
verify its branch, the worktree `.venv` path, interpreter and imported `okama.__file__`.
Do not switch to the shared checkout just to run the release workflow. Preserve
unrelated files there. A release worktree may publish its checked HEAD to `dev`,
then integrate the fresh `origin/dev` into `master`. If `dev` is checked out
elsewhere, fast-forward it in its owning checkout after inspecting its status.
Read an existing main-checkout `.env` privately by explicit path when needed;
keep the credentials out of the worktree and command output.

## Environment and authentication

Run `uv sync` in the release worktree; verify `uv run --no-sync python -V` and
`okama.__file__` before proceeding. Do not replace another worktree's environment.
uv does not require Poetry's keyring settings. Use `scripts/with_pypi_token.py` to pass the existing authorized PyPI credential
to one command, without printing its value or relying on shell state between
agent tool calls. It prefers `UV_PUBLISH_TOKEN` and reads the existing legacy store
when needed. Disable shell tracing around authentication. uv itself does not read
the legacy Poetry credential store.

## Phase 0 — Preflight

Run these checks first. Resolve missing prerequisites within the authorized scope; report a remaining blocker without publishing.

```bash
# 1) Verify the task branch and baseline before selecting a publication target
git branch --show-current
git status -sb
git log --oneline origin/dev..HEAD

# 2) Working tree clean
test -z "$(git status --porcelain)" || echo "FAIL: dirty tree"

# 3) gh CLI authenticated
gh auth status

# 4) The existing PyPI credential has been loaded privately for uv
# This checks presence without displaying the credential.
uv run --no-sync python scripts/with_pypi_token.py --check

# 5) Read the Docs API token available in .env
grep -q '^READTHEDOCS_TOKEN=' .env 2>/dev/null || echo "WARN: RTD API authorization unavailable here — verify the existing main .env or public version page"
```

`.env` is gitignored. The RTD token lets Phase 10 query and poll Read the Docs builds via the v3 API. If it is missing, the skill falls back to checking only the public docs URL.

### Existing or backup PyPI authorization

If the token exists only in the legacy `~/.config/pypoetry/auth.toml`, parse
`pypi-token.pypi` locally into `UV_PUBLISH_TOKEN` without displaying the value,
logging it, copying it into the repository or changing the issuer credential.
Verify it belongs to the intended registry/account before upload. If the user
requests recovery from a backup, inspect only the selected backup and manifest,
verify its supplied checksum, and preserve unrelated files and permissions.
A nonempty secret is presence evidence, not proof of registry acceptance.

## Phase 1 — uv lock --upgrade

```bash
uv lock --upgrade
uv sync
```

If anything in `uv.lock` changes meaningfully, mention it in the release notes draft later.

## Phase 2 — Tests

Three runs are mandatory before any version bump, and none of them may be skipped — not even for a "doc-only" release:

1. unit tests in the main uv environment (2c);
2. notebook tests `pytest --nbmake examples` (2c) — the only thing that catches API drift in the user-facing examples;
3. unit tests on the **minimum supported Python** (2d) — the only thing that catches breakage on the oldest version the package claims to support.

Plus a check that the CI matrix is green (2e).

### 2a. Install current dev source into the env

`pytest --nbmake` runs each notebook against a registered Jupyter kernel, and that kernel imports `okama` from the env. Without an explicit install the kernel may import a stale build of okama and miss the very changes you are about to release. Run `uv sync` first so the env reflects the current `dev` branch:

```bash
uv sync
```

Confirm the installed version matches `pyproject.toml` (`Installing the current project: okama (<NEW_VERSION>)`).

### 2b. Kernel isolation

The notebook command in Phase 2c creates a disposable kernel pointing at the
release worktree interpreter. Keep kernel creation and notebook execution in the
same shell invocation; variables exported by an earlier agent tool call are not
available in a later one. User-global kernels are not used or replaced.

### 2c. Run the tests

Run this complete block in one shell call. The trap removes only its own temporary
kernel prefix after pytest exits, including a failed notebook run:

```bash
uv run --no-sync pytest tests -q
kernel_root=$(mktemp -d)
trap 'rm -rf -- "$kernel_root"' EXIT
uv run --no-sync python -m ipykernel install --prefix "$kernel_root" --name python3 --display-name "Python 3"
JUPYTER_PATH="$kernel_root/share/jupyter" uv run --no-sync pytest --nbmake --nbmake-kernel=python3 -n=1 examples -q
```

Use the worker limit in `tests/pytest.ini`; run notebooks serially unless a measured
resource budget supports more. Check the actual execution host before heavy runs.
If the main environment already runs the minimum supported Python and has just
been installed, its full unit run also covers Phase 2d; record the interpreter.
For failures, follow AGENTS.md’s bounded fix-and-rerun policy and TDD for logic
fixes. Do not publish while a required gate fails.

### 2d. Run the unit tests on the **minimum supported Python** — mandatory

The main uv environment tracks the newest Python, and a green run there does **not** prove the release works on the declared minimum. Starting with Python 3.14, PEP 649 defers annotation evaluation, so a type hint that is invalid at runtime (e.g. `list | np.array`, where `np.array` is a function rather than a type) passes silently on 3.14 while breaking `import okama` outright on 3.11–3.13. That is exactly how v2.2.3, v2.2.4 and v2.3.0 shipped unimportable on three of the four supported versions (GH #95).

Read the minimum from `pyproject.toml` — the `requires-python` constraint under `[project]` — do not hardcode it here.

**Prepare a separate minimum-version environment.** Keep the main `.venv` intact.
Derive `MIN` from `[project].requires-python` (currently the `>=X.Y,<4.0.0` form),
and install fresh dependencies for that interpreter every release:

```bash
MIN=$(uv run --no-sync python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["requires-python"].split(">=")[1].split(",")[0])')
MIN_ENV="$PWD/tmp/release-minimum"
UV_PROJECT_ENVIRONMENT="$MIN_ENV" uv sync --python "$MIN" --no-default-groups --group test
"$MIN_ENV/bin/python" -c "import sys, okama; print(sys.version); print(okama.__version__, okama.__file__)"
"$MIN_ENV/bin/python" -m pytest tests -q
uv run --no-sync python -V
```

Verify the minimum interpreter matches `MIN`, its import points into the release
worktree, and the main interpreter remains the expected newest version. Never
use the user-global environment or switch `.venv` between Python versions.

Pass `tests` explicitly so pytest picks up `tests/pytest.ini`. The suite is offline (`tests/conftest.py` blocks `socket.socket`), so this run needs no network.

Notebook tests are **not** repeated here: `/examples` targets Google Colab, which currently ships a newer Python than the project minimum. Unit tests are the gate.

If this run fails while the 2c run passed, the failure is a real minimum-version incompatibility — stop and report; do not release.

### 2e. Confirm the CI matrix is green

`.github/workflows/tests.yml` runs the unit suite on every supported Python version on push and pull request. The check that counts is the run for the **release commit itself**, not merely the most recent green run on `dev` — there is no committed `uv.lock`, so every CI run resolves dependencies fresh from PyPI and yesterday's green says nothing about today's resolution. Do this check after the release commit has been pushed in Phase 8, keyed on its SHA:

```bash
gh run list --workflow=tests.yml --commit "$(git rev-parse HEAD)" --limit 5
```

If the run has not appeared yet, wait for it; do not proceed to the merge on the strength of an earlier run.

CI is the authoritative gate; 2d is the fast local check that keeps a broken minimum version from ever reaching a push. If the workflow's matrix no longer matches the `requires-python` constraint in `pyproject.toml` (for example after bumping the minimum), fix the workflow before releasing.

## Phase 3 — Lint & format

`ruff` issues are auto-fixable; apply fixes silently, then verify.

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .          # must exit 0
uv run ruff format --check . # must exit 0
uv run pre-commit run --all-files
```

If `ruff check .` still fails after `--fix`, or `pre-commit` reports issues that aren't auto-fixed: stop. Show the user the remaining diagnostics.

## Phase 4 — Docs build

Sphinx docs live in `docs/`; `.readthedocs.yml` is the hosted build configuration.
Check `command -v pandoc` before building: notebook conversion requires the system
Pandoc executable, which Python package dependencies alone do not provide. If missing,
resolve that prerequisite before retrying.

```bash
uv run python -m sphinx -E -b html docs docs/_build/html
```

The existing autosummary configuration generates the API pages. Do not add a
`sphinx-apidoc` source tree to this build: it duplicates the descriptions and is
absent from the hosted build. If an earlier local attempt generated `docs/source`,
inspect its contents and move only those generated files to ignored `tmp/` before
rebuilding with `-E`. Plot examples must construct every object they use; verify
that the build has no traceback even if Sphinx exits successfully.

Stop on any sphinx error. Warnings about missing references in third-party packages are usually OK; new warnings introduced by this release are not.

This phase only validates that docs build **locally**. Read the Docs is verified separately in Phase 10, after the tag is pushed (RTD auto-builds on tag).

## Phase 5 — Version bump

Read the current version from `pyproject.toml` (the `version = "X.Y.Z"` line under `[project]`). Use a version already specified by the user. Otherwise select the bump from the reviewed public behavior and explain the choice:

- **patch** — `X.Y.Z → X.Y.(Z+1)` (bug fixes, doc-only, dependency bumps)
- **minor** — `X.Y.Z → X.(Y+1).0` (new features, no breaking changes)
- **major** — `X.Y.Z → (X+1).0.0` (breaking changes)

If competing choices depend on missing requirements, clarify them; routine SemVer selection does not require a permission gate. Then update `pyproject.toml` (single line edit). Do not edit any other version field — okama uses `pyproject.toml` as the single source of truth. The new version is `<NEW_VERSION>` for the rest of the workflow.

Immediately after the bump, re-install the project so the env's package metadata matches:

```bash
uv sync
```

`okama.__version__` is read from the installed distribution metadata (`importlib.metadata.version("okama")` in `okama/__init__.py`), which is written only at install time. The editable install keeps the *code* current, but bumping `pyproject.toml` alone leaves the env — and every Jupyter kernel pointing at it — reporting the previous version (observed after v2.2.4: the env still said 2.2.3). Confirm the output ends with `Installing the current project: okama (<NEW_VERSION>)`.

## Phase 6 — Release notes & CHANGELOG

Draft the notes from git history since the last tag and the reviewed behavior.

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

Present the release theme and migration decisions in the progress update. Insert
the final block at the top of `CHANGELOG.md` under the title; publication already
authorized in the session does not require another approval gate for the notes.

Then build the GitHub Release body — usually identical to the new CHANGELOG section, but follow the format of <https://github.com/mbk-dev/okama/releases/tag/v2.0.0> if the user wants something richer. Save it to a temp file (e.g., `/tmp/release_notes_v<NEW_VERSION>.md`) for `gh release create --notes-file` later.

## Phase 7 — Sync requirements.txt

Keep `requirements.txt` aligned with the direct runtime constraints in
`[project].dependencies`; development groups do not belong in that file. Preserve
ranges rather than exporting the transitive versions in `uv.lock`. The lockfile
is local and ignored for this library. Keep the docs group and
`docs/requirements.txt` compatibility constraints aligned too.

## Phase 8 — Commit on dev, push, merge into master, tag

This is the git-flow phase. The user approved this exact order: commit on dev → push dev → merge into master → tag on master → push tags.

```bash
# Stage & commit on dev
git add pyproject.toml CHANGELOG.md
# Add other reviewed changed paths explicitly, then inspect the index and audit secrets.
git diff --cached --name-status
git commit -m "chore(release): prepare v<NEW_VERSION>"
```

Inspect the complete outgoing range against fresh `origin/dev`; publish only the
reviewed release work. Apply the authorization rule at the start of this skill.

```bash
git push origin HEAD:dev
```

Wait for GitHub Actions Tests and Ruff on that exact commit to pass. Inspect any
other release checks that run, including CodeQL; do not substitute an older run.

Before integrating into master, inspect the diff range:

```bash
git log --oneline origin/master..origin/dev
```

After the required checks pass and integration is authorized:

```bash
git fetch origin
git switch master
git merge --ff-only origin/master
git merge --no-ff origin/dev -m "Merge dev for v<NEW_VERSION>"
git tag "v<NEW_VERSION>"
git describe --tags  # sanity check: should print v<NEW_VERSION>
```

Publish master and the specific new tag; avoid pushing unrelated local tags.

```bash
git push origin master "refs/tags/v<NEW_VERSION>"
```

Verify the pushed merge commit’s required CI checks as well.

## Phase 9 — GitHub Release

Use the reviewed notes file and existing release authorization. Before retrying a
failed create call, check whether the release already exists.

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
set -a; source /home/chilango/projects/rs/okama-projects/okama/.env; set +a   # exports READTHEDOCS_TOKEN

API="https://readthedocs.org/api/v3/projects/okama"
AUTH="Authorization: Token $READTHEDOCS_TOKEN"
export VER="v<NEW_VERSION>"

# Find the build for the NEW VERSION specifically — NOT just any build matching the merge
# commit. RTD builds latest/stable/master AND the vX.Y.Z tag from the same commit and
# CANCELS the vX.Y.Z build as a duplicate of latest/master (observed for v2.2.4, 2026-07),
# so keying on the commit picks the successful latest build and reports a false green while
# en/vX.Y.Z/ 404s. Key on the version instead:
BUILD_ID=$(curl -s -H "$AUTH" "$API/builds/?limit=30" \
  | python3 -c "import json,sys,os; ver=os.environ['VER']; \
       print(next((b['id'] for b in json.load(sys.stdin).get('results',[]) if (b.get('version') or '')==ver), ''))")
```

If `BUILD_ID` is empty, wait 30s and retry. Once a build is registered, continue
watching its terminal result; a queued or running build is not a failed build.
Keep waits short enough to communicate progress. For a long unchanged state,
compare API timestamps with the build page’s detail endpoint
`https://app.readthedocs.org/api/v2/build/<BUILD_ID>/` to inspect command results;
do not infer a failure or cancellation from elapsed time alone.

Once the ID is found, poll until the build reaches a terminal state. Because the
`$VER` build is often **cancelled** as a duplicate of latest/master, re-trigger it when
that happens — a cancelled (or failed) version build leaves `en/$VER/` 404ing:

```bash
set -a; source /home/chilango/projects/rs/okama-projects/okama/.env; set +a   # exports READTHEDOCS_TOKEN

API="https://readthedocs.org/api/v3/projects/okama"
AUTH="Authorization: Token $READTHEDOCS_TOKEN"
export VER="v<NEW_VERSION>"

BUILD_ID=$(curl -s -H "$AUTH" "$API/builds/?limit=30" \
  | python3 -c "import json,sys,os; ver=os.environ['VER']; \
       print(next((b['id'] for b in json.load(sys.stdin).get('results',[]) if (b.get('version') or '')==ver), ''))")
while :; do
  STATE=$(curl -s -H "$AUTH" "$API/builds/$BUILD_ID/" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state']['code'], d['success'])")
  echo "$STATE"
  case "$STATE" in
    "finished True")  echo "RTD $VER OK"; break ;;
    "finished False") echo "RTD $VER build FAILED"; exit 1 ;;
    cancelled*)       echo "version build cancelled — re-triggering"; \
      BUILD_ID=$(curl -s -X POST -H "$AUTH" "$API/versions/$VER/builds/" \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('build',{}).get('id',''))") ;;
  esac
  sleep 30
done
```

If a real build (not a cancellation) fails: stop. Show the user the build URL (`https://app.readthedocs.org/projects/okama/builds/$BUILD_ID/`). Do not proceed to PyPI — fix the docs first, push a fix to master, wait for the new build to pass, then resume from Phase 11.

The success criterion for the release is that the **version's own** docs page returns `200`, not just `latest`. Confirm both:

```bash
VER="v<NEW_VERSION>"
curl -s -o /dev/null -w "latest:%{http_code}\n" "https://okama.readthedocs.io/en/latest/"
curl -s -o /dev/null -w "$VER:%{http_code}\n"    "https://okama.readthedocs.io/en/$VER/"
```

`latest` is rebuilt on every master push and should be `200`. If `en/$VER/` is still `404` after the version build finished, the version may be inactive: check `GET $API/versions/$VER/` for `active`/`built`, activate it (PATCH `{"active": true}` or the RTD dashboard), then re-trigger its build. Do not treat a green `latest` as proof the tagged version is served.

## Phase 11 — PyPI publish

Publish only after the version-specific documentation gate passes and publication
is authorized. PyPI does not allow re-uploading the same version. Before retrying
an upload error, inspect the version’s registry files for a partial upload.

Build and publish as **two separate steps**. Use an output directory dedicated
to this release; do not upload a broad `dist/*` that may include older artifacts.

```bash
ARTIFACT_DIR="$PWD/tmp/release-<NEW_VERSION>"
uv build --no-sources --out-dir "$ARTIFACT_DIR"
WHEEL="$ARTIFACT_DIR/okama-<NEW_VERSION>-py3-none-any.whl"
SDIST="$ARTIFACT_DIR/okama-<NEW_VERSION>.tar.gz"
uvx --from twine twine check "$WHEEL" "$SDIST"
sha256sum "$WHEEL" "$SDIST"
# Run only after artifact inspection, docs verification and publication authorization.
uv run --no-sync python scripts/with_pypi_token.py -- uv publish "$WHEEL" "$SDIST"
```

The wrapper loads the existing token in this invocation and passes it only to
the upload process as `UV_PUBLISH_TOKEN`. Preserve
both verified artifacts and their hashes for readback and partial-upload recovery.
Investigate stalled commands using their logs; stop only a process owned by this
release attempt and do not replay an upload without checking registry state.

Inspect wheel metadata (`Version`, `Requires-Python`) and the source archive before
upload. Afterward query the exact version and compare each published file’s
`digests.sha256` with its local artifact. Verify both wheel and source archive;
a version label alone does not prove the checked files were uploaded.

```bash
curl -fsS "https://pypi.org/pypi/okama/<NEW_VERSION>/json"
```

## Phase 12 — Post-release

Sync `dev` with `master` via fast-forward (the user explicitly chose this over delete-and-recreate).

```bash
git switch dev
git merge --ff-only master
git push
```

If `--ff-only` fails, dev has diverged from master post-merge — stop and ask the user how to handle it. Do not force-push, do not rebase without permission.

### Per-release Jupyter kernel

Register the requested per-release kernel separately from Phase 2b's disposable
test kernel. First synchronize after the version bump and verify distribution
metadata matches the intended release:

```bash
uv sync
uv run --no-sync python -c "import okama; v=okama.__version__; assert v=='<NEW_VERSION>', f'env metadata says {v}'; print('okama', v)"
uv run --no-sync python -m ipykernel install --user --name="okama<NEW_VERSION>" --display-name="okama<NEW_VERSION>"
```

Verify its executable points at the release worktree environment. Preserve the
`okama<X.Y.Z>` naming convention and all kernels from earlier releases; do not
remove user tooling as part of a release.

## Reporting

At the end, give the user a one-paragraph summary:

- new version
- tag name
- GitHub Release URL (from `gh release view --json url -q .url`)
- PyPI URL
- whether `dev` is fast-forwarded
- the per-release Jupyter kernel name that was registered (e.g. `okama2.2.4`)
- anything that was skipped or that needs manual follow-up (Read the Docs build, etc.)

Do not list every command that ran — the user does not need a transcript.

## What NOT to do

- Do not run `git push --force` or `git push --force-with-lease` at any point in this workflow. If a push is rejected, stop and report.
- Do not amend a commit that has already been pushed. Make a new commit instead.
- Do not delete the `dev` branch (the user chose fast-forward over recreation).
- Do not skip pre-commit hooks with `--no-verify` (forbidden by the user's global rules).
- Do not bump dependencies "while we're at it" — the only dependency change in this workflow comes from `uv lock --upgrade` in Phase 1.
- Do not write release notes or CHANGELOG entries in any language other than English (forbidden by `okama/AGENTS.md`).

## Recovery

If something fails between phases 8–12, the state is partially-released. Tell the user exactly what completed and what didn't, then ask before any cleanup. Common partial states:

- **Tagged locally, not pushed**: `git tag -d v<NEW_VERSION>` is safe.
- **Tag pushed, no GitHub Release**: `gh release create` can still run, or delete the remote tag with `git push --delete origin v<NEW_VERSION>` (ask first).
- **GitHub Release created, RTD build failed**: do not publish to PyPI. Fix the docs on master, RTD will rebuild automatically on the next push; once green, resume from Phase 11.
- **GitHub Release created, PyPI publish failed**: do not delete the GitHub Release — fix the PyPI issue (auth, network) and rerun `uv build` + `uv publish`. PyPI will accept the same version only if the previous attempt did not actually upload.
- **PyPI publish succeeded, post-release sync failed**: the release is done; just finish Phase 12 manually.
