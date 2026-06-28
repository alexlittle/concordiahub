import joblib
import pandas as pd

import xailens as xai
from examples import config
from examples.helpers import HEART_COLUMN_MAPPINGS


raw = pd.read_parquet('./cache/heart/xgboost/raw_instances.parquet')
X_test = pd.read_parquet('./cache/heart/xgboost/X_test.parquet')
y_test = pd.read_parquet('./cache/heart/xgboost/y_test.parquet')
y_pred = pd.read_parquet('./cache/heart/xgboost/y_pred.parquet')
model = joblib.load('./cache/heart/xgboost/model.joblib')

for col, mapping in HEART_COLUMN_MAPPINGS.items():
    raw[col] = raw[col].map(mapping)


data_schema = xai.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="target",
    prediction_column="y_pred")

model_data = xai.ModelData(
    data_schema=data_schema,
    raw_instances=raw,
    x_test=X_test,
    y_test=y_test,
    y_pred=y_pred)

ctx = xai.ModelContext(
    "xgboost",
    ["tree_shap"],
    "cleveland-heart-data",
    model_data,
    model=model,
    model_dir_name="heart_xgboost",
    model_display_name="XGBoost")

xai.run(ctx)