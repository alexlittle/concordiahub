"""
Tests for concordiahub.storage.store_base.DataStore abstract class

Covers:
- can't be instantiated
- partial concrete implementation can't be instantiated
- correct abstract methods
"""

import pytest

from concordiahub.storage.store_base import DataStore
from concordiahub.storage.filesystem_store import FilesystemStore


def test_datastore_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DataStore()


def test_incomplete_subclass_cannot_be_instantiated():
    class PartialStore(DataStore):
        def save_json(self, filename, data): pass

    with pytest.raises(TypeError):
        PartialStore()


def test_datastore_declares_expected_abstract_methods():
    assert DataStore.__abstractmethods__ == {
        "save_json", "save_dataframe", "save_series", "save_model",
        "save_explanation", "load_dataframe", "load_json", "load_series",
    }

def test_filesystem_store_implements_datastore():
    assert issubclass(FilesystemStore, DataStore)