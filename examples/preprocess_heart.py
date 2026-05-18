
# Processes the  Cleveland heart data to add the column names & uuid
import pandas as pd
import uuid
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
    "age", "sex", "chest pain type", "resting blood pressure", "serum cholesterol", "fasting blood sugar", "resting ECG",
    "maximum heart rate", "exercise induced angina", "ST depression", "slope of the peak exercise",
    "number of major vessels", "thalassemia", "target"
]

# Read in file and match to the columns
df = pd.read_csv("./data/cleveland-data-downloaded.csv", header=None, names=columns, na_values="?")

# Assign a UUID
df.insert(0, 'sample_uuid', [str(uuid.uuid4()) for _ in range(len(df))])

# Change to a binary target
df['target'] = (df['target'] > 0).astype(int)

# Simple imputation using median
df.fillna(df.median(numeric_only=True), inplace=True)

# Save processed file
df.to_csv("./data/cleveland-heart-data.csv", index=False)