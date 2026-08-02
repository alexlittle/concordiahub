"""
Tests model explorer

Covers:
- get_datasets
- get_models_for_dataset
- get_selected_models_for_dataset
- is_valid_dataset
- get_model_by_uuid
- get_models_by_uuids

"""
import json
import pytest

from xailens import config
from xailens.entities.model_explorer import ModelExplorer

@pytest.fixture
def explorer_factory(tmp_path, monkeypatch):
    """Builds a real ModelExplorer over a temporary metadata tree."""
    def _factory(metas=None):
        if metas is None:
            metas = [
                {"model_uuid": "lr-uuid", "dataset_id": "cleveland",
                 "model_display_name": "Logistic Regression", "datetime": "2026-01-02"},
                {"model_uuid": "xgb-uuid", "dataset_id": "cleveland",
                 "model_display_name": "XGBoost", "datetime": "2026-01-03"},
                {"model_uuid": "llm-uuid", "dataset_id": "medsynth",
                 "model_display_name": "LLM (CoT)", "datetime": "2026-01-01"},
            ]
        for meta in metas:
            model_dir = tmp_path / meta["model_uuid"]
            model_dir.mkdir()
            (model_dir / f"{config.META_INFO_FILENAME}.json").write_text(json.dumps(meta))

        monkeypatch.setattr(config, "MODEL_DATA_DIR", str(tmp_path))
        return ModelExplorer()

    return _factory


@pytest.fixture
def explorer(explorer_factory):
    return explorer_factory()


def test_load_all_metadata_sorts_by_datetime_descending(explorer):
    """Tests loading metadata from disk, newest first."""
    uuids = [m["model_uuid"] for m in explorer._metadata_cache]
    assert uuids == ["xgb-uuid", "lr-uuid", "llm-uuid"]


def test_load_all_metadata_records_path(explorer):
    """Tests each model's directory so Models can be built from it"""
    meta = explorer._uuid_index["lr-uuid"]
    assert meta["_path"].endswith("lr-uuid")


def test_get_datasets_returns_sorted_distinct(explorer):
    """Tests returns each dataset once, in alpha order"""
    assert explorer.get_datasets() == ["cleveland", "medsynth"]


def test_get_models_for_dataset(explorer):
    """Test returns models for the named dataset only."""
    models = explorer.get_models_for_dataset("cleveland")
    assert [m.get("model_display_name") for m in models] == ["XGBoost", "Logistic Regression"]


def test_get_selected_models_sorted_by_display_name(explorer):
    """Tests filtering by name and alpha sort"""
    models = explorer.get_selected_models_for_dataset(
        "cleveland", ["XGBoost", "Logistic Regression"])
    assert [m.get("model_display_name") for m in models] == ["Logistic Regression", "XGBoost"]


def test_get_selected_models_ignores_unknown_names(explorer):
    """Test skips names that don't match any model."""
    models = explorer.get_selected_models_for_dataset("cleveland", ["Nonexistent"])
    assert models == []


@pytest.mark.parametrize("dataset,expected", [("cleveland", True), ("nope", False)])
def test_is_valid_dataset(explorer, dataset, expected):
    """Tests dataset exists and has models"""
    assert explorer.is_valid_dataset(dataset) is expected


def test_get_model_by_uuid(explorer):
    """Tests returning a model built from the cached metadata"""
    model = explorer.get_model_by_uuid("lr-uuid")
    assert model.get("model_display_name") == "Logistic Regression"


def test_get_model_by_uuid_unknown_returns_none(explorer):
    """Test returns None for an unknown uuid."""
    assert explorer.get_model_by_uuid("nope") is None


def test_get_models_by_uuids_skips_unknown(explorer):
    """Test returning models for known uuids and skips the rest."""
    models = explorer.get_models_by_uuids(["lr-uuid", "nope", "llm-uuid"])
    assert [m.get("model_uuid") for m in models] == ["lr-uuid", "llm-uuid"]


def test_empty_metadata_directory(explorer_factory):
    """Tests handling data store with no models."""
    explorer = explorer_factory(metas=[])
    assert explorer.get_datasets() == []
    assert explorer.is_valid_dataset("cleveland") is False