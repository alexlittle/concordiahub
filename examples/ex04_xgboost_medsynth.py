import pandas as pd
import xai_compare as xai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.sparse import hstack


from helpers import check_gpu
from examples import config
device = check_gpu()
print(f"System check: {device} detected.")

# data to extract from the input file
df = pd.read_csv(config.MEDSYNTH_UTTERANCES_FILE)

# temp selection of first 10 conversations, only for testng
df = df[df[config.SAMPLE_UUID_COL].isin(df[config.SAMPLE_UUID_COL].unique()[:10])]

convs = df.groupby([config.SAMPLE_UUID_COL, 'icd_chapter', 'speaker'])['text'].apply(' '.join).unstack(fill_value='')
convs = convs.reset_index()

le = LabelEncoder()
y = pd.Series(
    le.fit_transform(convs['icd_chapter']),
    index=convs[config.SAMPLE_UUID_COL]
)

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

X_pat = pat_tfidf.fit_transform(convs['patient'])
X_doc = doc_tfidf.fit_transform(convs['doctor'])
X_combined = hstack([X_pat, X_doc])

feature_names = [f"pat_{f}" for f in pat_tfidf.get_feature_names_out()] + \
                [f"doc_{f}" for f in doc_tfidf.get_feature_names_out()]

X_df = pd.DataFrame(
    X_combined.toarray(),
    columns=feature_names,
    index=convs[config.SAMPLE_UUID_COL]
)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42, #stratify=y
)

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42,
    tree_method='hist',
    device=device
)
model.fit(X_train, y_train)

# keep raw array for sklearn metrics, wrap as DataFrame for framework
y_pred_array = model.predict(X_test)
y_pred = pd.DataFrame({'y_pred': y_pred_array}, index=X_test.index).replace(dict(enumerate(le.classes_)))
y_train = y_train.to_frame(name='y_train').replace(dict(enumerate(le.classes_)))
y_test = y_test.to_frame(name='y_test').replace(dict(enumerate(le.classes_)))

ctx = xai.RunContext("xgboost",
               ["shap_tree"],
               "medsynth",
               xai.RunData(
                   x_train=X_train,
                   x_test=X_test,
                   y_train=y_train,
                   y_test=y_test,
                   y_pred=y_pred,
                   class_names=le.classes_.tolist()),
               model=model,
                     run_id="xgboost_medsynth",)
xai.run(ctx)
