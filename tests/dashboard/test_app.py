"""
Tests for app.py

covers:
- launches without error
"""

from streamlit.testing.v1 import AppTest
from pathlib import Path

MODULE_UNDER_TEST = "xailens.dashboard.app"


APP_PATH = Path(__file__).parent.parent.parent / "xailens" / "dashboard" / "app.py"

def test_app_main():
    """Test main function."""
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception