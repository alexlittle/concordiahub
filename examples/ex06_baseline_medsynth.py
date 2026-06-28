import pandas as pd
import joblib
import spacy

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from helpers import check_gpu, clean_text, extract_spans_from_doc
from examples import config


device = check_gpu()
print(f"System check: {device} detected.")

# load the spaCy model
# uses: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("spaCy model not found. Run: python -m spacy download en_core_web_sm")

df = pd.read_csv(config.MEDSYNTH_FULL_FILE)
y = df.set_index(config.INSTANCE_UUID_COL)['target'].astype('category').cat.codes
docs = list(tqdm(
    nlp.pipe(df['scenario'].map(clean_text), batch_size=64),
    total=len(df),
    desc="Extracting spans"
))
df['scenario_spans'] = [extract_spans_from_doc(doc) for doc in docs]

scenario_tfidf = TfidfVectorizer(
    token_pattern=r"[^|]+",
    max_features=3000,
    sublinear_tf=True,
    max_df=0.8,
)
X = scenario_tfidf.fit_transform(df['scenario_spans'])
feature_names = scenario_tfidf.get_feature_names_out()
X_df = pd.DataFrame(
    X.toarray(),
    columns=feature_names,
    index=df[config.INSTANCE_UUID_COL]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)
model.fit(X_train, y_train)


y_pred_array = model.predict(X_test)
category_map = dict(enumerate(df['target'].astype('category').cat.categories))
y_pred = pd.DataFrame({'y_pred': y_pred_array}, index=X_test.index).replace(category_map)
y_train = y_train.to_frame(name='y_train').replace(category_map)
y_test = y_test.to_frame(name='y_test').replace(category_map)


# save to files for loading to XAI later
raw_instances = df.set_index(config.INSTANCE_UUID_COL)[['scenario']].loc[X_test.index]
raw_instances.reset_index().to_parquet('./cache/medsynth/baseline/raw_instances.parquet')
X_train.reset_index().to_parquet('./cache/medsynth/baseline/X_train.parquet')
X_test.reset_index().to_parquet('./cache/medsynth/baseline/X_test.parquet')
y_train.reset_index().to_parquet('./cache/medsynth/baseline/y_train.parquet')
y_test.reset_index().to_parquet('./cache/medsynth/baseline/y_test.parquet')
y_pred.reset_index().to_parquet('./cache/medsynth/baseline/y_pred.parquet')
joblib.dump(model, './cache/medsynth/baseline/model.joblib')

