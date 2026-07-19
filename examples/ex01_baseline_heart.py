"""
Logistic Regression model for the UCI Cleveland Heart Disease dataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import xailens
from examples.helpers import (HEART_COLUMN_MAPPINGS,
                              create_confusion_matrix,
                              save_class_report)
from examples import config


df = pd.read_csv(config.HEART_INPUT_FILE)

X = df.drop(columns=["target"])
y = df[["target", config.INSTANCE_UUID_COL]]

# create test/train split
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

report = classification_report(y_test["target"], y_pred['y_pred'])
print(report)

save_class_report(report, "heart", "lr")
create_confusion_matrix(y_test["target"],
                        y_pred['y_pred'],
                        "heart",
                        "lr")


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
    y_pred=y_pred,
    y_test=y_test)

ctx = xailens.ModelContext(
    "logistic_regression",
    ["coefficients"],
    "cleveland-heart-data",
    model_data,
    model=model,
    model_dir_name="heart_lr",
    model_display_name="Logistic Regression",)

xailens.run(ctx)