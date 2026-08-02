"""
Tests for the Dataset class (xailens.entities.dataset)

Uses the tests.fixtures.dataset and tests.fixtures.model fixtures

Covers:
- get_metrics
- get_recommended_models
- get_instance_data
- get_global_explanations
- get_local_explanations
- get_instance_consensus_ids
- get_instance_consensus
- get_all_instances
- _get_common_predictions (private helper)
- _get_explanations (private helper)
- _agreement (private helper)



"""


import pandas as pd

MODULE_UNDER_TEST = "xailens.entities.dataset"

# ------------------------------------
# get_metrics / get_recommended_models
# ------------------------------------

def test_get_metrics_indexed_by_model_display_name(heart_disease_dataset):
    df = heart_disease_dataset.get_metrics()

    assert set(df.index) == {"Logistic Regression", "XGBoost", "LLM (CoT)"}
    assert df.index.name == "model"
    assert df.columns.name is None


def test_get_metrics_values_match_underlying_models(heart_disease_dataset):
    df = heart_disease_dataset.get_metrics()

    assert df.loc["Logistic Regression", "f1_weighted"] == 0.81
    assert df.loc["XGBoost", "f1_weighted"] == 0.88
    assert df.loc["LLM (CoT)", "f1_weighted"] == 0.77


def test_get_recommended_models_returns_single_best(heart_disease_dataset):
    # default fixtures: XGBoost has the highest f1_weighted (0.88)
    assert heart_disease_dataset.get_recommended_models() == ["XGBoost"]


def test_get_recommended_models_handles_tie(heart_disease_dataset_factory):
    tied_metrics = pd.Series({"f1_weighted": 0.9, "accuracy": 0.9})
    lower_metrics = pd.Series({"f1_weighted": 0.5, "accuracy": 0.5})

    dataset = heart_disease_dataset_factory(
        lr_kwargs={"metrics": tied_metrics},
        xgboost_kwargs={"metrics": tied_metrics},
        llm_kwargs={"metrics": lower_metrics},
    )

    recommended = dataset.get_recommended_models()

    assert set(recommended) == {"Logistic Regression", "XGBoost"}


# -------------------
# get_instance_data
# -------------------

def test_get_instance_data_delegates_to_first_model(
    heart_disease_dataset_factory, lr_model_factory):
    lr_model = lr_model_factory()
    dataset = heart_disease_dataset_factory(models=[lr_model])

    expected = lr_model.get_instance_data("inst_0")
    result = dataset.get_instance_data("inst_0")

    pd.testing.assert_series_equal(result, expected)


# ------------------------
# get_instance_consensus_ids
# -------------------------

def test_get_instance_consensus_ids_full_overlap(heart_disease_dataset):
    # default fixtures all use the same 5 instance ids
    common_ids = heart_disease_dataset.get_instance_consensus_ids()

    assert common_ids == {f"inst_{i}" for i in range(5)}


def test_get_instance_consensus_ids_partial_overlap(
    heart_disease_dataset_factory
):
    id_col, target_col, pred_col = "instance_id", "target", "prediction"

    # XGBoost only has predictions for a subset of the LR/LLM instance ids
    xgboost_y_test = pd.DataFrame({
        id_col: ["inst_0", "inst_1"],
        target_col: [0, 1],
    })
    xgboost_y_pred = pd.DataFrame({
        id_col: ["inst_0", "inst_1"],
        pred_col: [0, 1],
    })

    dataset = heart_disease_dataset_factory(
        xgboost_kwargs={"y_test": xgboost_y_test, "y_pred": xgboost_y_pred},
    )

    common_ids = dataset.get_instance_consensus_ids()

    assert common_ids == {"inst_0", "inst_1"}


# ---------------------
# get_instance_consensus
# ---------------------

def test_get_instance_consensus_default_data(heart_disease_dataset):
    # With default fixture data, LR is wrong on every instance while
    # XGBoost/LLM are correct on every instance -> every instance is a
    # "disagree" case (never unanimous either way).
    result = heart_disease_dataset.get_instance_consensus()

    assert result == {
        "total": 5,
        "all_correct": 0,
        "all_incorrect": 0,
        "models_disagree": 5,
    }


def test_get_instance_consensus_mixed_scenario(heart_disease_dataset_factory):
    """
    3-instance scenario:
      - "a": all three models correct
      - "b": all three models incorrect
      - "c": models split (LR/LLM correct, XGBoost wrong)
    """
    id_col, target_col, pred_col = "instance_id", "target", "prediction"
    ids = ["a", "b", "c"]

    def _y_test():
        return pd.DataFrame({id_col: ids, target_col: [1, 1, 1]})

    lr_y_pred = pd.DataFrame({id_col: ids, pred_col: [1, 0, 1]})
    xgboost_y_pred = pd.DataFrame({id_col: ids, pred_col: [1, 0, 0]})
    llm_y_pred = pd.DataFrame({id_col: ids, pred_col: [1, 0, 1]})

    dataset = heart_disease_dataset_factory(
        lr_kwargs={"y_test": _y_test(), "y_pred": lr_y_pred},
        xgboost_kwargs={"y_test": _y_test(), "y_pred": xgboost_y_pred},
        llm_kwargs={"y_test": _y_test(), "y_pred": llm_y_pred},
    )

    result = dataset.get_instance_consensus()

    assert result == {
        "total": 3,
        "all_correct": 1,
        "all_incorrect": 1,
        "models_disagree": 1,
    }


# ----------------
# get_all_instances
# ----------------

def test_get_all_instances_columns_and_shape(heart_disease_dataset):
    result = heart_disease_dataset.get_all_instances()

    assert list(result.columns) == [
        "instance_uuid", "true_label",
        "Logistic Regression", "XGBoost", "LLM (CoT)",
        "agreement",
    ]
    assert len(result) == 5
    assert set(result["instance_uuid"]) == {f"inst_{i}" for i in range(5)}


def test_get_all_instances_sorted_by_true_label(heart_disease_dataset):
    result = heart_disease_dataset.get_all_instances()

    assert result["true_label"].is_monotonic_increasing


def test_get_all_instances_agreement_matches_default_data(heart_disease_dataset):
    # default fixtures: LR always wrong, XGBoost/LLM always right -> every
    # instance is a "disagree" case
    result = heart_disease_dataset.get_all_instances()

    assert result["agreement"].eq("disagree").all()


def test_get_all_instances_agreement_mixed_scenario(heart_disease_dataset_factory):
    id_col, target_col, pred_col = "instance_id", "target", "prediction"
    ids = ["a", "b", "c"]

    def _y_test():
        return pd.DataFrame({id_col: ids, target_col: [1, 1, 1]})

    lr_y_pred = pd.DataFrame({id_col: ids, pred_col: [1, 0, 1]})
    xgboost_y_pred = pd.DataFrame({id_col: ids, pred_col: [1, 0, 0]})
    llm_y_pred = pd.DataFrame({id_col: ids, pred_col: [1, 0, 1]})

    dataset = heart_disease_dataset_factory(
        lr_kwargs={"y_test": _y_test(), "y_pred": lr_y_pred},
        xgboost_kwargs={"y_test": _y_test(), "y_pred": xgboost_y_pred},
        llm_kwargs={"y_test": _y_test(), "y_pred": llm_y_pred},
    )

    result = dataset.get_all_instances()
    agreement_by_id = result.set_index("instance_uuid")["agreement"]

    assert agreement_by_id["a"] == "allcorrect"
    assert agreement_by_id["b"] == "allincorrect"
    assert agreement_by_id["c"] == "disagree"


