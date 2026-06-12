import pytest
from xailens.adapters.explainers.cor_adapter import CoRAdapter, ResponseSchema, LLMResponse, ContribScore


@pytest.fixture
def adapter():
    return CoRAdapter()


# --- list-of-dicts format (contribution_key set) ---

LIST_RAW = {
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