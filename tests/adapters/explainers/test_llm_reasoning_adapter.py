import pytest
import json
import pandas as pd

from xailens.adapters.explainers.llm_reasoning_adapter import LLMReasoningAdapter, ResponseSchema, LLMResponse, ContribScore
from xailens.exceptions import MissingDataError

@pytest.fixture
def adapter():
    return LLMReasoningAdapter()

def make_y_exp(*raw_dicts, id_column="instance_id"):
    return pd.DataFrame({
        id_column: [f"inst_{i}" for i in range(len(raw_dicts))],
        "explanation": [json.dumps(r) for r in raw_dicts],
    })

# --- list-of-dicts format (contribution_key set) ---

LIST_RAW = {
    "instance_id": "inst_0",
    "reasoning": "high cholesterol and age are key risk factors",
    "prediction": 1,
    "confidence": 0.85,
    "contributions": [
        {"feature": "age", "score": 2, "reasoning": "older age increases risk"},
        {"feature": "cholesterol", "score": 1, "reasoning": "elevated levels noted"},
    ]
}

LIST_SCHEMA = ResponseSchema(contributions_field="contributions", contribution_key="feature")


def test_reformat_list_format_returns_llm_response(adapter):
    result = adapter._reformat(LIST_RAW, LIST_SCHEMA, 1)
    assert isinstance(result, LLMResponse)


def test_reformat_list_format_top_level_fields(adapter):
    result = adapter._reformat(LIST_RAW, LIST_SCHEMA, 1)
    assert result.reasoning == LIST_RAW["reasoning"]
    assert result.prediction == LIST_RAW["prediction"]
    assert result.confidence == LIST_RAW["confidence"]


def test_reformat_list_format_contributions_keyed_by_feature(adapter):
    result = adapter._reformat(LIST_RAW, LIST_SCHEMA, 1)
    assert set(result.contributions.keys()) == {"age", "cholesterol"}


def test_reformat_list_format_contrib_score_values(adapter):
    result = adapter._reformat(LIST_RAW, LIST_SCHEMA, 1)
    assert result.contributions["age"] == ContribScore(score=2, reasoning="older age increases risk")
    assert result.contributions["cholesterol"] == ContribScore(score=1, reasoning="elevated levels noted")


def test_reformat_list_format_raw_preserved(adapter):
    result = adapter._reformat(LIST_RAW, LIST_SCHEMA, 1)
    assert result.raw is LIST_RAW


# --- dict-of-dicts format (contribution_key is None) ---

DICT_RAW = {
    "instance_id": "uuid_0",
    "reasoning": "exercise and chest pain are dominant",
    "prediction": 0,
    "confidence": 0.72,
    "contributions": {
        "exercise_angina": {"score": -1, "reasoning": "absence is protective"},
        "chest_pain": {"score": 2, "reasoning": "atypical angina present"},
    }
}

DICT_SCHEMA = ResponseSchema(contributions_field="contributions", contribution_key=None)


def test_reformat_dict_format_contributions_keyed_by_feature(adapter):
    result = adapter._reformat(DICT_RAW, DICT_SCHEMA, 0)
    assert set(result.contributions.keys()) == {"exercise_angina", "chest_pain"}


def test_reformat_dict_format_contrib_score_values(adapter):
    result = adapter._reformat(DICT_RAW, DICT_SCHEMA,0)
    assert result.contributions["chest_pain"] == ContribScore(score=2, reasoning="atypical angina present")


# --- edge cases ---

def test_reformat_empty_contributions_list(adapter):
    raw = {**LIST_RAW, "contributions": []}
    result = adapter._reformat(raw, LIST_SCHEMA,1)
    assert result.contributions == {}


def test_reformat_empty_contributions_dict(adapter):
    raw = {**DICT_RAW, "contributions": {}}
    result = adapter._reformat(raw, DICT_SCHEMA, 0)
    assert result.contributions == {}


def test_reformat_custom_contributions_field(adapter):
    raw = {**LIST_RAW, "feature_scores": LIST_RAW["contributions"]}
    schema = ResponseSchema(contributions_field="feature_scores", contribution_key="feature")
    result = adapter._reformat(raw, schema, 1)
    assert set(result.contributions.keys()) == {"age", "cholesterol"}

# --------------------
# test explain_global_json function
# ----------------------

def test_explain_global_json_importance_and_occurrence(adapter, model_context_factory, model_data_factory):
    data = model_data_factory(y_exp=make_y_exp(LIST_RAW, LIST_RAW))
    ctx = model_context_factory(data=data, response_schema=LIST_SCHEMA)

    result = adapter.explain_global_json(ctx)

    age_row = result[result["feature"] == "age"].iloc[0]
    assert age_row["importance"] == 2.0
    assert age_row["occurrence_count"] == 2
    assert age_row["occurrence_rate"] == 1.0
    assert age_row["n_instances"] == 2

def test_explain_global_json_partial_feature_occurrence(adapter, model_context_factory, model_data_factory):
    # cholesterol only appears in one of two rows
    raw_no_cholesterol = {**LIST_RAW, "contributions": [LIST_RAW["contributions"][0]]}
    data = model_data_factory(y_exp=make_y_exp(LIST_RAW, raw_no_cholesterol))
    ctx = model_context_factory(data=data, response_schema=LIST_SCHEMA)

    result = adapter.explain_global_json(ctx)

    cholesterol_row = result[result["feature"] == "cholesterol"].iloc[0]
    assert cholesterol_row["occurrence_count"] == 1
    assert cholesterol_row["occurrence_rate"] == 0.5
    assert cholesterol_row["n_instances"] == 2

def test_explain_global_json_dict_format(adapter, model_context_factory, model_data_factory):
    data = model_data_factory(y_exp=make_y_exp(DICT_RAW))
    ctx = model_context_factory(data=data, response_schema=DICT_SCHEMA)

    result = adapter.explain_global_json(ctx)

    assert set(result["feature"]) == {"exercise_angina", "chest_pain"}

def test_explain_global_json_raises_without_y_exp(adapter, model_context_factory, model_data_factory):
    data = model_data_factory(y_exp=None)
    ctx = model_context_factory(data=data, response_schema=LIST_SCHEMA)

    with pytest.raises(MissingDataError):
        adapter.explain_global_json(ctx)

def test_explain_global_json_raises_without_response_schema(adapter, model_context_factory, model_data_factory):
    data = model_data_factory(y_exp=make_y_exp(LIST_RAW))
    ctx = model_context_factory(data=data, response_schema=None)

    with pytest.raises(MissingDataError):
        adapter.explain_global_json(ctx)


# --------------------
# test explain_local_json function
# ----------------------


def test_explain_local_json_importance_and_occurrence(adapter, model_context_factory, model_data_factory):
    data = model_data_factory(y_exp=make_y_exp(LIST_RAW, LIST_RAW))
    ctx = model_context_factory(data=data, response_schema=LIST_SCHEMA)

    result = adapter.explain_local_json(ctx)

    assert len(result) == 2
    first = result[0]
    assert first["instance_id"] == "inst_0"
    assert set(first["contributions"]) == {"age", "cholesterol"}
    assert first["contributions"]["age"]["score"] == 2

# --------------------
# test load_local_json function
# ----------------------
def test_load_local_explanation_correct(llm_model_factory):
    model = llm_model_factory()

    result = LLMReasoningAdapter.load_local_explanation(model, "inst_0")

    assert isinstance(result, pd.DataFrame)
    scores = result["score"].tolist()
    abs_max = max(abs(s) for s in scores)
    for _, row in result.iterrows():
        assert row["score_normalised_max"] == pytest.approx(row["score"] / abs_max)
    assert (result["overall_reasoning"] == "Fake reasoning for inst_0").all()
    assert (result["overall_confidence"] == 0.75).all()

def test_load_local_explanation_instance_not_found(llm_model_factory):
    model = llm_model_factory()

    with pytest.raises(MissingDataError, match="inst_999"):
        LLMReasoningAdapter.load_local_explanation(model, "inst_999")

def test_load_local_explanation_zero_scores_fallback(llm_model_factory):
    local_json = [{
        "instance_id": "inst_0",
        "reasoning": "no signal",
        "confidence": 0.5,
        "contributions": {
            "age": {"score": 0.0, "reasoning": "none"},
            "sex": {"score": 0.0, "reasoning": "none"},
        },
    }]
    model = llm_model_factory(local_explanations_json=local_json)

    result = LLMReasoningAdapter.load_local_explanation(model, "inst_0")

    assert (result["score_normalised_max"] == result["score"]).all()

def test_load_local_explanation_single_contribution(llm_model_factory):
    local_json = [{
        "instance_id": "inst_0",
        "reasoning": "only one feature",
        "confidence": 0.9,
        "contributions": {
            "age": {"score": 3.0, "reasoning": "big effect"},
        },
    }]
    model = llm_model_factory(local_explanations_json=local_json)

    result = LLMReasoningAdapter.load_local_explanation(model, "inst_0")

    assert len(result) == 1
    assert result.iloc[0]["score_normalised_max"] == 1.0

def test_load_local_explanation_string_scores_coerced_to_float(llm_model_factory):
    local_json = [{
        "instance_id": "inst_0",
        "reasoning": "mixed types",
        "confidence": 0.6,
        "contributions": {
            "age": {"score": "2", "reasoning": "string score"},
            "sex": {"score": -1.5, "reasoning": "float score"},
        },
    }]
    model = llm_model_factory(local_explanations_json=local_json)

    result = LLMReasoningAdapter.load_local_explanation(model, "inst_0")

    assert result["score"].tolist() == [2.0, -1.5]