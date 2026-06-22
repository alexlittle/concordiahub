import pandas as pd

from xailens.dashboard.utils.comparison import generate_comparison_table_html
from xailens import constants

class MockModelEval:
    def __init__(self, model_dir_name, model_display_name):
        self.model_dir_name = model_dir_name
        self.model_display_name = model_display_name

    def get(self, key):
        if key == "dataset_id":
            return "dataset_1"
        if key == "datetime":
            return "2024-01-01"
        if key == "model_type":
            return "XGBoost"
        if key == "model_display_name":
            return self.model_display_name
        if key == constants.DATA_SCHEMA_ID_COLUMN:
            return "sample_uuid"
        return None

    def classification_report(self):
        return {"accuracy": 0.9}


def test_generate_comparison_table_html():
    df = pd.DataFrame([
        {
            "sample_uuid": "123",
            "Comparison": "Match",
            "True Value": 1,
            "model 1 prediction": 1,
            "model 2 prediction": 0,
        }
    ])

    modeleval_a = MockModelEval("Model A", "model 1")
    modeleval_b = MockModelEval("Model B", "model 2")

    html = generate_comparison_table_html(df, modeleval_a, modeleval_b)

    assert "<table" in html
    assert "123" in html
    assert "Match" in html
