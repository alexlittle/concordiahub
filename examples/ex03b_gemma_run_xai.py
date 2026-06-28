
import pandas as pd
import xailens as xai

from examples import config
from examples.helpers import safe_dirname

current_model = "gemma3:27b-it-qat"
dir_name = f"llm_{safe_dirname(current_model)}"

raw = pd.read_parquet(f"./cache/heart/{dir_name}/raw_instances.parquet")
X = pd.read_parquet(f"./cache/heart/{dir_name}/X.parquet")
y = pd.read_parquet(f"./cache/heart/{dir_name}/y.parquet")
y_preds = pd.read_parquet(f"./cache/heart/{dir_name}/y_preds.parquet")
y_exp = pd.read_parquet(f"./cache/heart/{dir_name}/y_exp.parquet")

schema = xai.ResponseSchema(
    contributions_field="feature_contributions",
    contribution_key=None,
)

data_schema = xai.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="target",
    prediction_column="prediction")

model_data = xai.ModelData(
    data_schema=data_schema,
    raw_instances=raw,
    x_test=X,
    y_test=y,
    y_pred=y_preds,
    y_exp=y_exp)

ctx = xai.ModelContext(
    "llm",
    ["llm_reasoning"],
    "cleveland-heart-data",
    model_data,
    model_subtype=current_model,
    model_dir_name=f"heart_llm_{safe_dirname(current_model)}",
    model_display_name="Gemma 3",
    response_schema=schema)

xai.run(ctx)