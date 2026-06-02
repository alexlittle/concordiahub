import pandas as pd

from xai_compare.dashboard.utils.comparison import generate_comparison_table_html
from xai_compare import constants

class MockRun:
    def __init__(self, model_name):
        self.model_name = model_name

    def get(self, key):
        if key == "dataset_id":
            return "dataset_1"
        if key == "datetime":
            return "2024-01-01"
        if key == "model_type":
            return self.model_name
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
            "Model A prediction": 1,
            "Model B prediction": 0,
        }
    ])

    run_a = MockRun("Model A")
    run_b = MockRun("Model B")

    html = generate_comparison_table_html(df, run_a, run_b)

    assert "<table" in html
    assert "123" in html
    assert "Match" in html
