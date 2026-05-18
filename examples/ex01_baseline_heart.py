# UCI Cleveland Heart Disease dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import xai_compare as xai

df = pd.read_csv("./data/cleveland-heart-data.csv")

X = df.drop(columns=["sample_uuid", "target"])
X.index = df['sample_uuid']

y = df["target"]
y.index = df['sample_uuid']


imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)


model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
)

model.fit(X_scaled, y)

y_pred = pd.DataFrame({'y_pred': model.predict(X_scaled)}, index=X_scaled.index)

print(accuracy_score(y, y_pred))


coef_df = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=[f"class_{c}" for c in model.classes_[1:]]
)
print(coef_df)

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





