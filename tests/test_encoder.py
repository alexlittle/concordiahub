import pytest
import numpy as np
import pandas as pd

from unittest.mock import MagicMock, patch

from xailens.encoder import XAITextEncoder

MODULE_UNDER_TEST = "xailens.encoder"

@pytest.fixture
def sample_df():
    return pd.DataFrame(columns=["age", "cholesterol", "blood_pressure"])

@pytest.fixture
def mock_sentence_transformer():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(3, 384)
    return mock_model

def test_encode_feature_names_shape(sample_df, mock_sentence_transformer):
    with patch(f"{MODULE_UNDER_TEST}.SentenceTransformer", return_value=mock_sentence_transformer):
        encoder = XAITextEncoder()
        result = encoder.encode_feature_names(sample_df)
        assert result.shape == (3, 384)

def test_encode_feature_names_index(sample_df, mock_sentence_transformer):
    with patch(f"{MODULE_UNDER_TEST}.SentenceTransformer", return_value=mock_sentence_transformer):
        encoder = XAITextEncoder()
        result = encoder.encode_feature_names(sample_df)
        assert result.index.tolist() == ["age", "cholesterol", "blood_pressure"]

def test_encode_feature_names_returns_dataframe(sample_df, mock_sentence_transformer):
    with patch(f"{MODULE_UNDER_TEST}.SentenceTransformer", return_value=mock_sentence_transformer):
        encoder = XAITextEncoder()
        result = encoder.encode_feature_names(sample_df)
        assert isinstance(result, pd.DataFrame)