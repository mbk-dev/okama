# Dependency alert triage — 2026-10-26

Issue: [#129, Triage 78 manifest Dependabot alerts and library dependency ranges](https://github.com/mbk-dev/okama/issues/129).

GitHub returned 78 open records: 6 critical, 42 high, 25 medium, and 5 low.
The [per-alert ledger](dependabot-2026-10-26.csv) accounts for every record,
including the primary advisory, affected range, first patched version, observed
dependency snapshot, project scope, relevant code path, and disposition. These
are alert records, not 78 distinct exploitable defects in okama.

## Findings and changes

The alert payloads identify exact historical dependency versions under
`pyproject.toml`. The manifest itself permits wider ranges. A fresh local uv
resolution already selected patched versions, which does not protect a consumer
resolving an older allowed version and does not close a GitHub alert.

Runtime security floors are therefore explicit package metadata and are mirrored
in `requirements.txt`: `pillow>=13.0.0`, `certifi>=2026.10.19`, and
`urllib3>=2.8.0`. They constrain dependencies already introduced by matplotlib
and requests; no application-style lockfile is shipped.

Development-only floors stay in the groups that use them:

| Group | Minimum versions |
|---|---|
| test | black[jupyter] 26.10.1 |
| docs | nbconvert 7.17.2, tornado 6.5.9 |
| jupyter | notebook 7.6.3, jupyterlab 4.6.4, nbconvert 7.17.2, tornado 6.5.9 |
| dev | virtualenv 21.14.9, filelock 4.2.1 |

The runtime library performs verified HTTPS requests through requests, so the
certifi trust-store advisory is relevant. urllib3 response handling is also in
that path; exploitability of each reported parser/streaming condition has not
been established for the configured API. Pillow is used transitively by
matplotlib; there is no direct PIL/Image.open call in `okama/`. This does not
establish safety for downstream applications that decode untrusted images.

Notebook conversion is used by the docs and Jupyter environments. Notebook trust,
web-server input and local virtualenv/filelock advisories require their respective
interactive/server/local-user conditions. The checked-in workflows do not use
the affected `psf/black` Action with `use_pyproject`; its package is still updated.
The ledger retains these distinctions instead of silently dismissing alerts.

## Reproduce the inventory and environment evidence

Run from the repository using an account allowed to read security alerts:

```bash
gh api --paginate repos/mbk-dev/okama/dependabot/alerts --method GET \
  -f state=open -f per_page=100 > alerts.json
uv sync
uv tree --invert --depth 3 --package pillow --package urllib3 --package certifi \
  --package virtualenv --package filelock --package black --package nbconvert \
  --package tornado --package notebook --package jupyterlab
uv pip list --format json > installed.json
```

The following stdlib-only extraction reconstructs the advisory/range columns and
counts directly from the API snapshot; the reachability/disposition columns are
the source review recorded above and in the ledger:

```python
import collections
import json
from pathlib import Path

alerts = json.loads(Path("alerts.json").read_text())
print(f"Alert count: {len(alerts)}")
print(collections.Counter(a["security_advisory"]["severity"] for a in alerts))
for alert in sorted(alerts, key=lambda row: row["number"]):
    advisory = alert["security_advisory"]
    vulnerability = alert["security_vulnerability"]
    print(
        alert["number"], alert["dependency"]["package"]["name"],
        advisory["ghsa_id"], vulnerability["vulnerable_version_range"],
        vulnerability["first_patched_version"], advisory["summary"],
    )
```

Initial local resolution: Pillow 13.0.0, certifi 2026.10.23, urllib3 2.8.0,
black 26.10.1, virtualenv 21.14.13, filelock 4.2.4, nbconvert 7.17.2,
notebook 7.6.3, jupyterlab 4.6.4, tornado 6.6, pandas 3.1.0.

## Delivery and remaining limits

Native supported-version tests and package metadata checks are recorded in the
backlog acceptance report. Publication and a fresh GitHub dependency-graph/alert
read are separate acceptance steps. Until that read is available, the observed
remote count remains 78 open; the local range changes are not a claim of zero
remote alerts or of safety against advisories outside this snapshot. No alerts
have been dismissed and no security workflows have been disabled.
