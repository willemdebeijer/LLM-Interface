import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-api",
        action="store_true",
        default=False,
        help="run API tests that cost money",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "api: mark test as requiring API access")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-api"):
        skip_api = pytest.mark.skip(reason="need --run-api option to run")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)
