"""
Tests for Streamlit home page

Covers:
- loading page
"""


from pathlib import Path
from streamlit.testing.v1 import AppTest


PAGE = Path(__file__).parents[3] / "concordiahub" / "dashboard" / "pages" / "home.py"

def test_homepage_loads(monkeypatch, mock_explorer):
    """Test loads page with overview tab"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.run()

    assert not at.exception
    assert not at.error