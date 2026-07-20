"""
XGBoost model for the MedSynth dataset
"""

import pandas as pd
import joblib
import spacy

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from helpers import clean_text, extract_spans_from_doc
from examples import config


# load the spaCy model
# uses: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("spaCy model not found. Run: python -m spacy download en_core_web_sm")



# data to extract from the input file
df = pd.read_csv(config.MEDSYNTH_FULL_FILE)

y = df.set_index(config.INSTANCE_UUID_COL)['target'].astype('category').cat.codes

# run spaCy over the scenario texts to extract the noun chunks/entities and
# create parsed docs
docs = list(tqdm(
    nlp.pipe(df['scenario'].map(clean_text), batch_size=64),
    total=len(df),
    desc="Extracting spans"
))

# turn each parsed doc into a string of phrases, joined by "|"
df['scenario_spans'] = [extract_spans_from_doc(doc) for doc in docs]


# vectorise the text using  "|" as the separator - prevents the phrases being
# broken up into individual wordss
scenario_tfidf = TfidfVectorizer(
    token_pattern=r"[^|]+",
    max_features=3000,
    sublinear_tf=True,
    max_df=0.8,
)

# build a dataframe with the phrases as column names, values are the TF-IDF
# weights
X = scenario_tfidf.fit_transform(df['scenario_spans'])

feature_names = scenario_tfidf.get_feature_names_out()

X_df = pd.DataFrame(
    X.toarray(),
    columns=feature_names,
    index=df[config.INSTANCE_UUID_COL]
)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42,
    tree_method='hist',
    device="cpu"
)

model.fit(X_train, y_train, verbose=10)

# keep raw array for sklearn metrics, wrap as DataFrame for framework
y_pred_array = model.predict(X_test)

category_map = dict(enumerate(df['target'].astype('category').cat.categories))
y_pred = pd.DataFrame({'y_pred': y_pred_array}, index=X_test.index).replace(category_map)
y_test = y_test.to_frame(name='y_test').replace(category_map)

# save to files for loading to XAI later
raw_instances = df.set_index(config.INSTANCE_UUID_COL)[['scenario']].loc[X_test.index]
raw_instances.reset_index().to_parquet('./cache/medsynth/xgboost/raw_instances.parquet')
X_test.reset_index().to_parquet('./cache/medsynth/xgboost/X_test.parquet')
y_test.reset_index().to_parquet('./cache/medsynth/xgboost/y_test.parquet')
y_pred.reset_index().to_parquet('./cache/medsynth/xgboost/y_pred.parquet')
joblib.dump(model, './cache/medsynth/xgboost/model.joblib')

