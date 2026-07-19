"""
XGBoost model for the UCI Cleveland Heart Disease dataset
"""

import pandas as pd
import xailens

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from examples import config
from examples.helpers import (HEART_COLUMN_MAPPINGS,
                              create_confusion_matrix,
                              check_gpu,
                              save_class_report)

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
report = classification_report(y_test, y_pred_vals)

print(report)

save_class_report(report, "heart", "xgboost")
create_confusion_matrix(y_test,
                        y_pred_vals,
                        "heart",
                        "xgboost")

y_train_df = pd.DataFrame({config.INSTANCE_UUID_COL: X_train[config.INSTANCE_UUID_COL].values, 'target': y_train.values})
y_test_df = pd.DataFrame({config.INSTANCE_UUID_COL: X_test[config.INSTANCE_UUID_COL].values, 'target': y_test.values})

raw_instances = df.set_index(config.INSTANCE_UUID_COL).drop(columns=['target']).loc[
    X_test[config.INSTANCE_UUID_COL].values
]

for col, mapping in HEART_COLUMN_MAPPINGS.items():
    raw_instances[col] = raw_instances[col].map(mapping)

data_schema = xailens.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="target",
    prediction_column="y_pred")

model_data = xailens.ModelData(
    data_schema=data_schema,
    raw_instances=raw_instances,
    x_test=X_test,
    y_test=y_test_df,
    y_pred=y_pred)

ctx = xailens.ModelContext(
    "xgboost",
    ["tree_shap"],
    "cleveland-heart-data",
    model_data,
    model=model,
    model_dir_name="heart_xgboost",
    model_display_name="XGBoost")

xailens.run(ctx)
