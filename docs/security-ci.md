# Recurring source security audit

Native task: https://github.com/mbk-dev/okama/issues/128. Coordination: infra#269.
Project: okama. The caller contains configuration and documentation; scanner
implementation is the reviewed public composite action pinned below. Application
code, dependencies and business processes are unchanged by this integration.

Action: `mbk-dev/cbrapi/.github/actions/security-audit@08be62bd91bcc85c04605889701a4bcd34e25a07`.

PR/relevant main or master pushes, weekly and manual runs use GitHub-hosted Ubuntu,
read-only contents and checkout without persisted credentials. The action runs its
contract tests in its trusted source tree with isolated Python imports, then
scans this repository. It never installs or runs the caller application. Bootstrap
and scanner child processes run in private tool state with an environment
allowlist. Inherited Python environment/user-site settings cannot load caller code.

Trivy includes development dependencies and refreshes its vulnerability database;
Gitleaks scans the PR commit range or full history; Semgrep uses immutable upstream
rules. All tool archives/wheel/rule digests are verified before execution. Pins
are in the action's reviewed source and action updates require a full-SHA change.

Findings are advisory. Tool/bootstrap/JSON failures, new parser errors and timeouts
fail visibly. Semgrep top-level errors and time.fixpoint_timeouts are aggregated,
with a separate timeout count. A timeout can never be accepted by parser policy.
Known manually reviewed parser gaps remain errors/partial_known, bound to exact
source bytes, structured spans where available, occurrence count and tool/rules.
PRs read the base commit's policy; candidate changes cannot approve themselves.
Findings are never suppressed. Missing dependency inputs mean coverage_gap and
null findings, not zero vulnerabilities. Unsupported/unlocked/vendor inputs and
vulnerability reachability need separate analysis; no library lockfile is added.

Only counts/status/pins appear in logs and Step Summary. Raw source, paths,
messages and secrets stay private on the ephemeral runner; no artifacts are
uploaded. This source audit does not check deployed images, host OS, live
configuration, issuer revocation or client databases. It has separate concurrency
and does not dispatch application deployment. Relevant native push/deploy gates
are preserved; the weekly audit does not invoke unrelated deploy workflows.

For private local reproduction, use the canonical source at the exact action SHA:
run its contract with isolated Python module search, then run its run.py from the
target checkout using Python -E -s. Do not install application dependencies just
to scan them. Review any parser acceptance against a real private scanner report.
A successful source job is not a claim of complete coverage or application health.
