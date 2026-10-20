"""
Tests for cli.py to launch dashboard app

Tests:
- xailen-dash launches the streamlit app
- runs without launching browser
- runs with launching browser
"""


from unittest.mock import patch
from concordiahub.cli import main

MODULE_UNDER_TEST = "concordiahub.cli"

def test_main_runs_streamlit(tmp_path):
    """Test runs streamlit."""
    with patch("subprocess.run") as mock_run, patch("sys.argv", ["concordiahub-dash"]):
        main()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "streamlit"
        assert args[1] == "run"
        assert args[2].endswith("app.py")


def test_main_no_browser(tmp_path):
    """Test runs without launching browser."""
    with patch("subprocess.run") as mock_run, patch("sys.argv", ["concordiahub-dash", "--no-browser"]):
        main()
        args = mock_run.call_args[0][0]
        assert "--server.headless" in args
        assert "true" in args

def test_main_browser_by_default(tmp_path):
    """Test runs with browser."""
    with patch("subprocess.run") as mock_run, patch("sys.argv", ["concordiahub-dash"]):
        main()
        args = mock_run.call_args[0][0]
        assert "--server.headless" not in args

def test_main_data_dir_passed_via_env(tmp_path):
    """Test --data-dir is passed to the app via environment and created."""
    target = tmp_path / "my_data"
    with patch("subprocess.run") as mock_run, patch("sys.argv", ["concordiahub-dash", "--data-dir", str(target)]):
        main()
        env = mock_run.call_args.kwargs["env"]
        assert env["CONCORDIAHUB_DATA_DIR"] == str(target)
        assert target.is_dir()


def test_main_default_data_dir_created_in_cwd(tmp_path, monkeypatch):
    """Test default data dir is ./data_store in the current directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONCORDIAHUB_DATA_DIR", raising=False)
    with patch("subprocess.run"), patch("sys.argv", ["concordiahub-dash"]):
        main()
    assert (tmp_path / "data_store").is_dir()
