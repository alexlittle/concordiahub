"""
Fixtures that will get loaded automatically every time the tests are run
"""

pytest_plugins = [
    "tests.fixtures.models",
    "tests.fixtures.dataset",
    "tests.fixtures.model_explorer",
    "tests.fixtures.context",
]
