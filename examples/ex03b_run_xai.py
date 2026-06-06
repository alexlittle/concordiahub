
import pandas as pd
import xai_compare as xai

from examples import config
from examples.helpers import safe_dirname

current_model = "gemma3:27b-it-qat"
dir_name = f"llm_{safe_dirname(current_model)}"
X = pd.read_parquet(f"./cache/heart/{dir_name}/X.parquet")
y = pd.read_parquet(f"./cache/heart/{dir_name}/y.parquet")
y_preds = pd.read_parquet(f"./cache/heart/{dir_name}/y_preds.parquet")
y_exp = pd.read_parquet(f"./cache/heart/{dir_name}/y_exp.parquet")


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
    "cleveland-heart-data",
    run_data,
    model_subtype=current_model,
    run_name=f"llm_heart_{safe_dirname(current_model)}",
    model_id="LLM")

xai.run(ctx)