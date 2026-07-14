"""Tests for EfficientFrontier parallel fan-out guards (issue #94).

These verify that ``settings.resolve_n_jobs`` never lets process-level
parallelism multiply: it collapses to a single job whenever the caller is
already inside a parallel context (a pytest-xdist worker or an active joblib
pool), and otherwise honours the ``OKAMA_N_JOBS`` environment override.
"""

from joblib import Parallel, delayed

from okama import settings


def test_resolve_n_jobs_defaults_to_all_cores(monkeypatch):
    """At top level with no configuration it uses all cores (-1)."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("OKAMA_N_JOBS", raising=False)
    assert settings.resolve_n_jobs() == -1


def test_resolve_n_jobs_honours_env_override(monkeypatch):
    """OKAMA_N_JOBS bounds the degree of parallelism at top level."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("OKAMA_N_JOBS", "2")
    assert settings.resolve_n_jobs() == 2


def test_resolve_n_jobs_collapses_under_xdist_worker(monkeypatch):
    """Inside a pytest-xdist worker it collapses to 1 regardless of the env cap."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    monkeypatch.setenv("OKAMA_N_JOBS", "8")
    assert settings.resolve_n_jobs() == 1


def test_resolve_n_jobs_collapses_inside_joblib_pool(monkeypatch):
    """Called from within an active joblib pool it collapses to 1, so an
    EfficientFrontier run nested in another Parallel cannot fan out N x N."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("OKAMA_N_JOBS", "-1")
    results = Parallel(n_jobs=2, backend="threading")(delayed(settings.resolve_n_jobs)() for _ in range(2))
    assert results == [1, 1]
