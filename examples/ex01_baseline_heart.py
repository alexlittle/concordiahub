# UCI Cleveland Heart Disease dataset
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from examples import config

df = pd.read_csv(config.HEART_INPUT_FILE)

X = df.drop(columns=[config.INSTANCE_UUID_COL, "target"])
X.index = df[config.INSTANCE_UUID_COL]

y = df["target"]
y.index = df[config.INSTANCE_UUID_COL]


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

for frame in [X_scaled, y.to_frame(), y_pred]:
    frame.insert(0, config.INSTANCE_UUID_COL, df[config.INSTANCE_UUID_COL].values)

print(accuracy_score(y, y_pred['y_pred']))

coef_df = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=[f"class_{c}" for c in model.classes_[1:]]
)
print(coef_df)



# save to files for loading to XAI later
X_scaled.to_parquet('./cache/heart/baseline/X.parquet', index=False)
y_df = y.to_frame()
y_df.insert(0, config.INSTANCE_UUID_COL, df[config.INSTANCE_UUID_COL].values)
y_df.to_parquet('./cache/heart/baseline/y.parquet', index=False)
y_pred.to_parquet('./cache/heart/baseline/y_pred.parquet', index=False)
joblib.dump(model, './cache/heart/baseline/model.joblib')





