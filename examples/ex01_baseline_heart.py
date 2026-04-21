# UCI Cleveland Heart Disease dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("./data/cleveland-heart-data.csv")

print(df.head())

X = df.drop(columns="num")
y = df["num"]


imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
)

model.fit(X_scaled, y)


y_pred = model.predict(X_scaled)
print(accuracy_score(y, y_pred))


coef_df = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=[f"class_{c}" for c in model.classes_]
)
print(coef_df)






