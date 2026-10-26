"""Verify the installation cell without downloading or replacing any package.

Run with the full development environment: uv run pytest <this file> -q.
"""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
@pytest.mark.parametrize("colab", [False, True], ids=["local", "colab"])
def test_bootstrap_installs_only_in_colab(notebook: Path, colab: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(cell["source"]) for cell in cells if "%pip install" in "".join(cell.get("source", [])))
    if colab:
        monkeypatch.setitem(sys.modules, "google.colab", ModuleType("google.colab"))
    else:
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    calls = []
    transformer = InteractiveShell.instance()

    class InstallerRecorder:
        def run_line_magic(self, name: str, arguments: str) -> None:
            calls.append((name, arguments))

        def run_cell_magic(self, name: str, arguments: str, body: str) -> None:
            assert name == "capture"
            exec(transformer.transform_cell(body), namespace)

    namespace = {"get_ipython": lambda: InstallerRecorder()}
    exec(transformer.transform_cell(source), namespace)
    assert calls == ([("pip", "install --quiet -U okama")] if colab else [])
