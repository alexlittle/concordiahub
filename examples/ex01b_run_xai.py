
import joblib
import pandas as pd

import xailens as xai
from examples import config

X = pd.read_parquet('./cache/heart/baseline/X.parquet')
y = pd.read_parquet('./cache/heart/baseline/y.parquet')
y_pred = pd.read_parquet('./cache/heart/baseline/y_pred.parquet')
model = joblib.load('./cache/heart/baseline/model.joblib')

data_schema = xai.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="target",
    prediction_column="y_pred")

model_data = xai.ModelData(
    data_schema=data_schema,
    x=X,
    y=y,
    y_pred=y_pred)

ctx = xai.ModelContext(
    "logistic_regression",
    ["coefficients"],
    "cleveland-heart-data",
    model_data,
    model=model,
    model_dir_name="heart_lr",
    model_display_name="Logistic Regression",)

xai.run(ctx)