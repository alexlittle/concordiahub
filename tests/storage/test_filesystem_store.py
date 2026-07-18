"""
Tests for FilesystemStore Class

Filesystem i/o is mocked as don't want any test data to actually be written to
disk.

Covers:
- class instantiation
- save_json
- save_dataframe
- save_series
- save_explanation (dataframe and JSON formats)
- save_model
- load_dataframe
- load_json
- load_series


"""
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, mock_open, patch

from xailens.storage.filesystem_store import (
    FilesystemStore,
    _cached_load_csv,
    _cached_load_json,
)

MODULE = "xailens.storage.filesystem_store"


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    """
    st.cache_resource caches across the test session, so need to clear before
    and after every test.
    """
    _cached_load_csv.clear()
    _cached_load_json.clear()
    yield
    _cached_load_csv.clear()
    _cached_load_json.clear()


@pytest.fixture
def store():
    """FilesystemStore with os.makedirs mocked so __init__ never touches disk."""
    with patch(f"{MODULE}.os.makedirs") as mock_makedirs:
        instance = FilesystemStore("fake_dir")
    instance._mock_makedirs = mock_makedirs
    return instance


# Test instantiation
def test_creates_dirname_with_exist_ok():
    with patch(f"{MODULE}.os.makedirs") as mock_makedirs:
        FilesystemStore("some_dir")
    mock_makedirs.assert_called_once_with("some_dir", exist_ok=True)

# test save_json
def test_writes_expected_path_and_content(store):
    m = mock_open()
    with patch(f"{MODULE}.open", m), patch(f"{MODULE}.json.dump") as mock_dump:
        store.save_json("myfile", {"a": 1})

    m.assert_called_once_with(os.path.join("fake_dir", "myfile.json"), "w")
    mock_dump.assert_called_once()
    args, kwargs = mock_dump.call_args
    assert args[0] == {"a": 1}
    assert kwargs.get("indent") == 4


# test save_dataframe
def test_calls_to_csv_with_expected_path(store):
    df = MagicMock(spec=pd.DataFrame)
    store.save_dataframe("mydf", df)
    df.to_csv.assert_called_once_with(
        os.path.join("fake_dir", "mydf.csv"), index=False)

# test save_series
def test_calls_to_csv_with_expected_path(store):
    series = MagicMock(spec=pd.Series)
    store.save_series("myseries", series)
    series.to_csv.assert_called_once_with(
        os.path.join("fake_dir", "myseries.csv"))


# test save_explanation - dataframe
def test_df_format_stringifies_columns_and_saves_csv(store):
    df = MagicMock(spec=pd.DataFrame)
    df.columns = pd.Index([1, 2, 3])
    store.save_explanation("myexp", df, data_format="df")

    # columns should have been reassigned to string dtype
    assert list(df.columns.astype(str)) == ["1", "2", "3"]
    df.to_csv.assert_called_once_with(
        os.path.join("fake_dir", "myexp.csv"), index=False)

# test save_explanation - json
def test_non_df_format_writes_json(store):
    m = mock_open()
    with patch(f"{MODULE}.open", m), patch(f"{MODULE}.json.dump") as mock_dump:
        store.save_explanation("myexp", {"k": "v"}, data_format="json")

    m.assert_called_once_with(os.path.join("fake_dir", "myexp.json"), "w")
    args, kwargs = mock_dump.call_args
    assert args[0] == {"k": "v"}
    assert kwargs.get("indent") == 4


# test save_model
def test_calls_joblib_dump_with_expected_path(store):
    model = MagicMock()
    with patch(f"{MODULE}.joblib.dump") as mock_dump:
        store.save_model("mymodel", model)
    mock_dump.assert_called_once_with(
        model, os.path.join("fake_dir", "mymodel.joblib"))


# test load_dataframe
def test_calls_read_csv_with_expected_path(store):
    expected_df = pd.DataFrame({"a": [1, 2]})
    with patch(f"{MODULE}.pd.read_csv", return_value=expected_df) as mock_read:
        result = store.load_dataframe("mydf")

    mock_read.assert_called_once_with(os.path.join("fake_dir", "mydf.csv"))
    pd.testing.assert_frame_equal(result, expected_df)

# test load_json
def test_calls_open_and_json_load_with_expected_path(store):
    m = mock_open(read_data='{"a": 1}')
    with patch(f"{MODULE}.open", m), patch(
        f"{MODULE}.json.load", return_value={"a": 1}
    ) as mock_load:
        result = store.load_json("myfile")

    m.assert_called_once_with(os.path.join("fake_dir", "myfile.json"), "r")
    mock_load.assert_called_once()
    assert result == {"a": 1}


# test load_series
def test_calls_read_csv_with_index_col_and_squeezes(store):
    # single-col frame that squeeze() will turn into a Series
    single_col_df = pd.DataFrame({"value": [1, 2, 3]})
    with patch(f"{MODULE}.pd.read_csv", return_value=single_col_df) as mock_read:
        result = store.load_series("myseries")

    mock_read.assert_called_once_with(
        os.path.join("fake_dir", "myseries.csv"), index_col=0)
    assert isinstance(result, pd.Series)