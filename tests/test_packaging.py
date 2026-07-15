"""Packaging invariants: one version, everywhere."""

import pytest


def test_version_sources_agree():
    from importlib.metadata import PackageNotFoundError, version

    import smallbatch

    try:
        installed = version("smallbatch")
    except PackageNotFoundError:
        pytest.skip("smallbatch not installed as a distribution")
    assert installed == smallbatch.__version__
