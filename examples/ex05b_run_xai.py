import pandas as pd
import xai_compare as xai

from examples import config
from examples.helpers import safe_dirname

dir_name = f"llm_{safe_dirname(config.MEDSYNTH_OLLAMA_MODEL)}"
X = pd.read_parquet(f"./cache/medsynth/{dir_name}/X.parquet")
y = pd.read_parquet(f"./cache/medsynth/{dir_name}/y.parquet")
y_preds = pd.read_parquet(f"./cache/medsynth/{dir_name}/y_preds.parquet")
y_exp = pd.read_parquet(f"./cache/medsynth/{dir_name}/y_exp.parquet")

schema = xai.ResponseSchema(
    contributions_field="feature_contributions",
    contribution_key=None,
)

data_schema = xai.DataSchema(
    sample_id_column=config.SAMPLE_UUID_COL,
    target_column="target",
    prediction_column="prediction")

run_data = xai.RunData(
    data_schema=data_schema,
    x=X,
    y=y,
    y_pred=y_preds,
    y_exp=y_exp)

ctx = xai.RunContext(
    "llm",
    ["cor"],
    "medsynth",
    run_data,
    model_subtype=config.MEDSYNTH_OLLAMA_MODEL,
    run_name="llm_medsynth",
    model_id="LLM v1",
    response_schema=schema)

xai.run(ctx)