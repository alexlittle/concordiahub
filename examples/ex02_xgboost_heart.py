import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from examples import config

from helpers import check_gpu

device = check_gpu()
print(f"System check: {device} detected.")

df = pd.read_csv(config.HEART_INPUT_FILE)
X = df.drop(columns=[config.INSTANCE_UUID_COL, "target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Add instance_uuid to all splits (using iloc position to slice from df)
uuids = df[config.INSTANCE_UUID_COL]
X_train.insert(0, config.INSTANCE_UUID_COL, uuids.iloc[X_train.index].values)
X_test.insert(0, config.INSTANCE_UUID_COL, uuids.iloc[X_test.index].values)

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42,
    tree_method='hist',
    device=device
)

feature_cols = [c for c in X_train.columns if c != config.INSTANCE_UUID_COL]
model.fit(X_train[feature_cols], y_train)

y_pred_vals = model.predict(X_test[feature_cols])
y_pred = pd.DataFrame({
    config.INSTANCE_UUID_COL: X_test[config.INSTANCE_UUID_COL].values,
    'y_pred': y_pred_vals
})

print("Accuracy:", accuracy_score(y_test, y_pred_vals))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_vals))

y_train_df = pd.DataFrame({config.INSTANCE_UUID_COL: X_train[config.INSTANCE_UUID_COL].values, 'target': y_train.values})
y_test_df = pd.DataFrame({config.INSTANCE_UUID_COL: X_test[config.INSTANCE_UUID_COL].values, 'target': y_test.values})

X_train.to_parquet('./cache/heart/xgboost/X_train.parquet', index=False)
X_test.to_parquet('./cache/heart/xgboost/X_test.parquet', index=False)
y_train_df.to_parquet('./cache/heart/xgboost/y_train.parquet', index=False)
y_test_df.to_parquet('./cache/heart/xgboost/y_test.parquet', index=False)
y_pred.to_parquet('./cache/heart/xgboost/y_pred.parquet', index=False)
joblib.dump(model, './cache/heart/xgboost/model.joblib')
