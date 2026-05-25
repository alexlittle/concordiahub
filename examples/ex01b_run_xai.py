
import joblib
import pandas as pd

import xai_compare as xai


X = pd.read_parquet('./cache/heart/baseline/X.parquet')
y = pd.read_parquet('./cache/heart/baseline/y.parquet')
y_pred = pd.read_parquet('./cache/heart/baseline/y_pred.parquet')
model = joblib.load('./cache/heart/baseline/model.joblib')

run_data = xai.RunData(
    x=X,
    y=y,
    y_pred=y_pred,
)
ctx = xai.RunContext("logistic_regression",
               ["coefficients"],
               "cleveland-heart-data",
               run_data,
               model=model)
xai.run(ctx)