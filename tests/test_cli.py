from unittest.mock import patch
from xailens.cli import main

MODULE_UNDER_TEST = "xailens.cli"

def test_main_runs_streamlit(tmp_path):
    with patch("subprocess.run") as mock_run:
        main()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "streamlit"
        assert args[1] == "run"
        assert args[2].endswith("app.py")