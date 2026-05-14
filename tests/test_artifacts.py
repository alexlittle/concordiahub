from unittest.mock import patch

import pytest

from xai_compare.artifacts import extract_dataset

MODULE_UNDER_TEST = "xai_compare.artifacts"



def test_extract_dataset():
    import io
    fake_file = io.StringIO('{"dataset_id": "dataset_a"}')
    with patch("builtins.open", return_value=fake_file):
        assert extract_dataset("any_path") == "dataset_a"
