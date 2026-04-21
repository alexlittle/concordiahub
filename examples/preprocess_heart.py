
# Processes the  Cleveland heart data to add the column names
import pandas as pd
'''
1. age        – age in years
2. sex        – sex (1 = male; 0 = female)
3. cp         – chest pain type
4. trestbps   – resting blood pressure (mm Hg)
5. chol       – serum cholesterol (mg/dl)
6. fbs        – fasting blood sugar > 120 mg/dl
7. restecg    – resting ECG results
8. thalach    – maximum heart rate achieved
9. exang      – exercise induced angina
10. oldpeak   – ST depression induced by exercise
11. slope     – slope of the peak exercise ST segment
12. ca        – number of major vessels (0–3)
13. thal      – thalassemia
14. num       – diagnosis of heart disease (target)
'''

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

df = pd.read_csv("./data/processed.cleveland.data.csv", header=None, names=columns, na_values="?")

df.to_csv("./data/cleveland-heart-data.csv", index=False)