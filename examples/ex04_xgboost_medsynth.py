import pandas as pd
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.sparse import hstack


from helpers import check_gpu

device = check_gpu()
print(f"System check: {device} detected.")



# data to extract from the input file
df = pd.read_csv("./data/medsynth_utterances.csv")

convs = df.groupby(['conversation_id', 'icd_chapter', 'speaker'])['text'].apply(' '.join).unstack(fill_value='')
convs = convs.reset_index()

# encode labels
le = LabelEncoder()
y = le.fit_transform(convs['icd_chapter'])

# vectorize texts
pat_tfidf = TfidfVectorizer(
    max_features=3000,
    sublinear_tf=True,
    max_df=0.8,
    stop_words='english'
)

doc_tfidf = TfidfVectorizer(
    max_features=2000,
    sublinear_tf=True,
    max_df=0.8,
    stop_words='english'
)

# fit the text
X_pat = pat_tfidf.fit_transform(convs['patient'])
X_doc = doc_tfidf.fit_transform(convs['doctor'])

# create the combined array for input to model
X_combined = hstack([X_pat, X_doc])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# train
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42,
    tree_method='hist',  # needed for GPU usage
    device=device # put on GPU if available
)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# show stats for the model
report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

print("Classification Report:\n", report)





'''
with open("./models/medsynth_xgboost/classification_report.txt", "w") as f:
    f.write("XGBoost ICD-10 Chapter Classification Report\n")
    f.write("=" * 50 + "\n")
    f.write(report)


cm = confusion_matrix(y_test, y_pred)

# show percentages instead of raw counts
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

# wrap labels
wrapped_labels = [textwrap.fill(label, 20) for label in le.classes_]

plt.figure(figsize=(16, 12))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=wrapped_labels,
    yticklabels=wrapped_labels,
    cbar_kws={'label': 'Proportion of Correct Predictions'}
)

plt.title('Normalized Confusion Matrix: XGBoost ICD-10 Classification', fontsize=16)
plt.ylabel('Actual Category', fontsize=12)
plt.xlabel('Predicted Category', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./models/medsynth_xgboost/confusion_matrix.svg')
plt.show()


# save model
model.save_model("./models/medsynth_xgboost/icd10_xgboost_model.ubj")

# save vectorizers
joblib.dump(pat_tfidf, "./models/medsynth_xgboost/patient_vectorizer.joblib")
joblib.dump(doc_tfidf, "./models/medsynth_xgboost/doctor_vectorizer.joblib")

# save label encoder
joblib.dump(le, "./models/medsynth_xgboost/label_encoder.joblib")

# save the specific x/y-train/test datasets
joblib.dump(X_train, "./models/medsynth_xgboost/X_train.joblib")
joblib.dump(y_train, "./models/medsynth_xgboost/y_train.joblib")
joblib.dump(X_test, "./models/medsynth_xgboost/X_test.joblib")
joblib.dump(y_test, "./models/medsynth_xgboost/y_test.joblib")
joblib.dump(y_pred, "./models/medsynth_xgboost/y_pred.joblib")
'''