"""Credential handoff for release tooling, using synthetic tokens only."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/with_pypi_token.py"


@pytest.fixture
def credential_wrapper() -> ModuleType:
    assert SCRIPT.is_file(), "The release credential wrapper has not been implemented"
    spec = importlib.util.spec_from_file_location("with_pypi_token", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_token_takes_precedence(credential_wrapper: ModuleType, tmp_path: Path) -> None:
    assert credential_wrapper.load_token({"UV_PUBLISH_TOKEN": "synthetic-env-token"}, tmp_path) == "synthetic-env-token"


def test_legacy_token_is_loaded_without_changing_its_file(credential_wrapper: ModuleType, tmp_path: Path) -> None:
    auth = tmp_path / ".config/pypoetry/auth.toml"
    auth.parent.mkdir(parents=True)
    content = '[pypi-token]\npypi = "synthetic-legacy-token"\n'
    auth.write_text(content)
    assert credential_wrapper.load_token({}, tmp_path) == "synthetic-legacy-token"
    assert auth.read_text() == content


def test_legacy_config_directory_is_respected(credential_wrapper: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "auth.toml").write_text('[pypi-token]\npypi = "synthetic-custom-token"\n')
    assert credential_wrapper.load_token({"POETRY_CONFIG_DIR": str(tmp_path)}, tmp_path) == "synthetic-custom-token"


@pytest.mark.parametrize("content", [None, "invalid=[", '[pypi-token]\npypi = ""', "[pypi-token]\npypi = 123"])
def test_missing_or_invalid_token_fails_safely(
    credential_wrapper: ModuleType, tmp_path: Path, content: str | None
) -> None:
    if content is not None:
        auth = tmp_path / ".config/pypoetry/auth.toml"
        auth.parent.mkdir(parents=True)
        auth.write_text(content)
    with pytest.raises(ValueError, match="PyPI credential"):
        credential_wrapper.load_token({}, tmp_path)


def test_wrapper_passes_token_to_child_without_printing_it(credential_wrapper: ModuleType, tmp_path: Path) -> None:
    env = {**os.environ, "HOME": str(tmp_path), "UV_PUBLISH_TOKEN": "synthetic-child-token"}
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--",
            sys.executable,
            "-c",
            'import os; assert os.environ["UV_PUBLISH_TOKEN"] == "synthetic-child-token"',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


def test_presence_check_never_prints_the_token(credential_wrapper: ModuleType, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        env={**os.environ, "HOME": str(tmp_path), "UV_PUBLISH_TOKEN": "synthetic-check-token"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "synthetic-check-token" not in result.stdout + result.stderr
