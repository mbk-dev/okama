"""Compare complete fresh environments using the current and legacy manifests.

Uses existing host caches without purging them. Each trial has a unique project
path, no lock and no environment. Installer time excludes inventory/import probes.
Run from the repository root: python analytics/poetry_uv_migration_20261026/benchmark.py.
"""

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "tmp/uv-migration/benchmark"


def trial(tool: str, number: int) -> dict[str, object]:
    environment = {**os.environ, "POETRY_VIRTUALENVS_IN_PROJECT": "true", "POETRY_KEYRING_ENABLED": "false"}
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH"):
        environment.pop(key, None)
    with tempfile.TemporaryDirectory(prefix=f"{tool}-{number}-", dir=OUTPUT) as temporary:
        directory = Path(temporary)
        for name in ("README.md", "LICENSE.txt"):
            shutil.copyfile(ROOT / name, directory / name)
        shutil.copytree(ROOT / "okama", directory / "okama", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        manifest = ROOT / "pyproject.toml" if tool == "uv" else EVIDENCE / "legacy-pyproject.toml"
        shutil.copyfile(manifest, directory / "pyproject.toml")
        commands = (
            [["uv", "sync", "--python", "/usr/bin/python3.14"]]
            if tool == "uv"
            else [["poetry", "env", "use", "/usr/bin/python3.14"], ["poetry", "install", "--no-interaction"]]
        )
        start = time.perf_counter()
        with (OUTPUT / f"{tool}-{number}.log").open("w") as log:
            for command in commands:
                subprocess.run(
                    command,
                    cwd=directory,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=True,
                )
        elapsed = time.perf_counter() - start
        probe = 'import json,importlib.metadata as m; print(json.dumps({d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions() if d.metadata["Name"]!="pip"},sort_keys=True))'
        packages = json.loads(subprocess.check_output([str(directory / ".venv/bin/python"), "-c", probe], text=True))
        subprocess.run(
            [str(directory / ".venv/bin/python"), "-c", "import okama; assert okama.__version__=='4.0.0'"],
            cwd=directory,
            env=environment,
            check=True,
        )
        return {"tool": tool, "trial": number, "seconds": elapsed, "packages": packages}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for number in range(1, 4):
        for tool in ("poetry", "uv") if number % 2 else ("uv", "poetry"):
            row = trial(tool, number)
            rows.append(row)
            (OUTPUT / "results.json").write_text(json.dumps(rows, indent=2))
            print(f"{tool} trial {number}: {row['seconds']:.3f} seconds", flush=True)
    reference = rows[0]["packages"]
    assert all(row["packages"] == reference for row in rows), "Resolved package versions differ; inspect inventories"
    medians = {
        tool: statistics.median(row["seconds"] for row in rows if row["tool"] == tool) for tool in ("poetry", "uv")
    }
    (OUTPUT / "summary.json").write_text(json.dumps(medians, indent=2))
    print(f"Medians: {medians}; speedup: {medians['poetry'] / medians['uv']:.1f}x")


if __name__ == "__main__":
    main()
