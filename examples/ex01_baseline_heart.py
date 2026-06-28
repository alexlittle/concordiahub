# UCI Cleveland Heart Disease dataset
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from examples import config

df = pd.read_csv(config.HEART_INPUT_FILE)

X = df.drop(columns=["target"])
y = df[["target", config.INSTANCE_UUID_COL]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y["target"]
)

model = LogisticRegression(solver="lbfgs", max_iter=5000)
model.fit(X_train.drop(columns=[config.INSTANCE_UUID_COL]), y_train["target"])

y_pred_array = model.predict(X_test.drop(columns=[config.INSTANCE_UUID_COL]))
y_pred = pd.DataFrame({
    config.INSTANCE_UUID_COL: X_test[config.INSTANCE_UUID_COL].values,
    'y_pred': y_pred_array,
})

print(accuracy_score(y_test["target"], y_pred['y_pred']))

raw_instances = df.set_index(config.INSTANCE_UUID_COL).drop(columns=['target']).loc[
    X_test[config.INSTANCE_UUID_COL].values
]
raw_instances.reset_index().to_parquet('./cache/heart/baseline/raw_instances.parquet')
X_train.to_parquet('./cache/heart/baseline/X_train.parquet', index=False)
X_test.to_parquet('./cache/heart/baseline/X_test.parquet', index=False)
y_train.to_parquet('./cache/heart/baseline/y_train.parquet', index=False)
y_test.to_parquet('./cache/heart/baseline/y_test.parquet', index=False)
y_pred.to_parquet('./cache/heart/baseline/y_pred.parquet', index=False)

joblib.dump(model, './cache/heart/baseline/model.joblib')





