"""Empirical reliability is an explicit acceptance gate, separate from software tests."""
import pytest


def pytest_addoption(parser):
    parser.addoption('--run-empirical', action='store_true', default=False,
                     help='run the real reference-author reliability gate; requires downloaded benchmark sources')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-empirical'):
        return
    skip = pytest.mark.skip(reason='Empirical acceptance is opt-in: download benchmark data, then use --run-empirical')
    for item in items:
        if 'empirical' in item.keywords:
            item.add_marker(skip)
