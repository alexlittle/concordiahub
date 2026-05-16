from unittest.mock import patch

from xai_compare.run_explorer import extract_dataset

MODULE_UNDER_TEST = "xai_compare.run_explorer"


def test_extract_dataset():
    import io
    fake_file = io.StringIO('{"dataset_id": "dataset_a"}')
    with patch("builtins.open", return_value=fake_file):
        assert extract_dataset("any_path") == "dataset_a"
