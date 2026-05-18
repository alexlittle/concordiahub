import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import xai_compare as xai


from helpers import check_gpu

device = check_gpu()
print(f"System check: {device} detected.")

df = pd.read_csv("./data/cleveland-heart-data.csv")

X = df.drop(columns=["sample_uuid", "target"])
X.index = df['sample_uuid']

y = df["target"]
y.index = df['sample_uuid']

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

y_pred = model.predict(X_test)
y_pred = pd.DataFrame({'y_pred': model.predict(X_test)}, index=X_test.index)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

ctx = xai.RunContext("xgboost",
               ["shap_tree"],
               "cleveland-heart-data",
               X_train,
               X_test,
               y_train,
               y_test,
               y_pred,
               model=model)
xai.run(ctx)
