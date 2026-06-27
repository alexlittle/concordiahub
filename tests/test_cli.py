from unittest.mock import patch
from xailens.cli import main

MODULE_UNDER_TEST = "xailens.cli"

def test_main_runs_streamlit(tmp_path):
    with patch("subprocess.run") as mock_run, patch("sys.argv", ["xailens-dash"]):
        main()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "streamlit"
        assert args[1] == "run"
        assert args[2].endswith("app.py")


def test_main_no_browser(tmp_path):
    with patch("subprocess.run") as mock_run, patch("sys.argv", ["xailens-dash", "--no-browser"]):
        main()
        args = mock_run.call_args[0][0]
        assert "--server.headless" in args
        assert "true" in args

def test_main_browser_by_default(tmp_path):
    with patch("subprocess.run") as mock_run, patch("sys.argv", ["xailens-dash"]):
        main()
        args = mock_run.call_args[0][0]
        assert "--server.headless" not in args