import joblib
import pandas as pd

import xai_compare as xai


X_train = pd.read_parquet('./cache/medsynth/xgboost/X_train.parquet')
X_test = pd.read_parquet('./cache/medsynth/xgboost/X_test.parquet')
y_train = pd.read_parquet('./cache/medsynth/xgboost/y_train.parquet')
y_test = pd.read_parquet('./cache/medsynth/xgboost/y_test.parquet')
y_pred = pd.read_parquet('./cache/medsynth/xgboost/y_pred.parquet')
model = joblib.load('./cache/medsynth/xgboost/model.joblib')


ctx = xai.RunContext("xgboost",
                ["shap_tree"],
                "medsynth",
                xai.RunData(
                   x_train=X_train,
                   x_test=X_test,
                   y_train=y_train,
                   y_test=y_test,
                   y_pred=y_pred),
                model=model,
                run_name="xgboost_medsynth",)
xai.run(ctx)