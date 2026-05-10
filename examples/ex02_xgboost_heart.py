import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import xai_compare as xai


from helpers import check_gpu

device = check_gpu()
print(f"System check: {device} detected.")

df = pd.read_csv("./data/cleveland-heart-data.csv")

# Simple imputation using median
df.fillna(df.median(), inplace=True)

X = df.drop(columns=["num"])
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42,
    tree_method='hist',  # needed for GPU usage
    device=device # put on GPU if available
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

'''
plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()
'''

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
