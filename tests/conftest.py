"""Pytest fixtures and hypothesis profiles for the keeloq test suite."""

from __future__ import annotations

import os
import shutil

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci", max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "dev", max_examples=10_000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


def _python2_available() -> bool:
    return shutil.which("python2") is not None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _binary_available(name: str) -> bool:
    return shutil.which(name) is not None


@pytest.fixture(scope="session")
def python2_available() -> bool:
    return _python2_available()


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return _cuda_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="CUDA unavailable")
    skip_legacy = pytest.mark.skip(reason="python2 not on PATH")
    skip_kissat = pytest.mark.skip(reason="kissat binary not on PATH")
    skip_minisat = pytest.mark.skip(reason="minisat binary not on PATH")

    cuda = _cuda_available()
    py2 = _python2_available()
    kissat = _binary_available("kissat")
    minisat = _binary_available("minisat")

    for item in items:
        if "gpu" in item.keywords and not cuda:
            item.add_marker(skip_gpu)
        if "legacy" in item.keywords and not py2:
            item.add_marker(skip_legacy)
        if "solver_kissat" in item.keywords and not kissat:
            item.add_marker(skip_kissat)
        if "solver_minisat" in item.keywords and not minisat:
            item.add_marker(skip_minisat)
