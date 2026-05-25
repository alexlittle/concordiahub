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

X = df.drop(columns=[config.SAMPLE_UUID_COL, "target"])
X.index = df[config.SAMPLE_UUID_COL]

y = df["target"]
y.index = df[config.SAMPLE_UUID_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42,
    tree_method='hist',  # needed for GPU usage
    device=device # put on GPU if available
)

model.fit(X_train, y_train)

y_pred = pd.DataFrame({'y_pred': model.predict(X_test)}, index=X_test.index)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# save to files for loading to XAI later
X_train.to_parquet('./cache/heart/xgboost/X_train.parquet')
X_test.to_parquet('./cache/heart/xgboost/X_test.parquet')
y_train.to_frame().to_parquet('./cache/heart/xgboost/y_train.parquet')
y_test.to_frame().to_parquet('./cache/heart/xgboost/y_test.parquet')
y_pred.to_parquet('./cache/heart/xgboost/y_pred.parquet')
joblib.dump(model, './cache/heart/xgboost/model.joblib')
