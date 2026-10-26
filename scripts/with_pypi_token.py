"""Pass an existing PyPI credential to one command without exposing its value.

Usage: uv run --no-sync python scripts/with_pypi_token.py --check
       uv run --no-sync python scripts/with_pypi_token.py -- uv publish <wheel> <sdist>
"""

from collections.abc import Mapping
import os
from pathlib import Path
import sys
import tomllib


def load_token(environment: Mapping[str, str], home: Path) -> str:
    """Prefer an explicit uv token, falling back to the existing legacy store."""
    explicit = environment.get("UV_PUBLISH_TOKEN", "").strip()
    if explicit:
        return explicit
    config_home = Path(environment.get("XDG_CONFIG_HOME", str(home / ".config")))
    legacy = Path(environment.get("POETRY_CONFIG_DIR", str(config_home / "pypoetry"))) / "auth.toml"
    try:
        data = tomllib.loads(legacy.read_text())
        section = data.get("pypi-token", {})
        token = section.get("pypi") if isinstance(section, dict) else None
    except (OSError, ValueError):
        raise ValueError("PyPI credential unavailable; load the existing token securely") from None
    if not isinstance(token, str) or not token.strip():
        raise ValueError("PyPI credential is missing or invalid")
    return token.strip()


def main() -> int:
    arguments = sys.argv[1:]
    if arguments != ["--check"] and (len(arguments) < 2 or arguments[0] != "--"):
        print("Usage: with_pypi_token.py --check | -- <command> [arguments]", file=sys.stderr)
        return 2
    try:
        token = load_token(os.environ, Path.home())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if arguments == ["--check"]:
        print("PyPI credential available (registry acceptance not checked)")
        return 0
    environment = {**os.environ, "UV_PUBLISH_TOKEN": token}
    try:
        os.execvpe(arguments[1], arguments[1:], environment)
    except OSError:
        print("Unable to start the requested release command", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
