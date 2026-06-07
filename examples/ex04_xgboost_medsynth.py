import pandas as pd
import joblib
import spacy
import re
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from helpers import check_gpu
from examples import config


device = check_gpu()
print(f"System check: {device} detected.")

# load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_spans_from_doc(doc) -> str:
    """Extracts noun chunks from a sentence."""
    spans = (
            [chunk.text.lower() for chunk in doc.noun_chunks] +
            [ent.text.lower() for ent in doc.ents]
    )
    seen = set()
    unique_spans = []
    for s in spans:
        s = re.sub(r'[\[\]<>]', '', s).strip()
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            unique_spans.append(s)
    return " | ".join(unique_spans)


# data to extract from the input file
df = pd.read_csv(config.MEDSYNTH_FULL_FILE)

y = df.set_index(config.SAMPLE_UUID_COL)['target'].astype('category').cat.codes

docs = list(tqdm(
    nlp.pipe(df['scenario'], batch_size=64),
    total=len(df),
    desc="Extracting spans"
))
df['scenario_spans'] = [extract_spans_from_doc(doc) for doc in docs]


# vectorize texts based on the "noun chunks"
# don't drop stop words as could break the phrase chunking and spaCy already
# handles stop words so they won't be individual tokens
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
    index=df[config.SAMPLE_UUID_COL]
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
    device=device
)
model.fit(X_train, y_train, verbose=10)

# keep raw array for sklearn metrics, wrap as DataFrame for framework
y_pred_array = model.predict(X_test)

category_map = dict(enumerate(df['target'].astype('category').cat.categories))
y_pred = pd.DataFrame({'y_pred': y_pred_array}, index=X_test.index).replace(category_map)
y_train = y_train.to_frame(name='y_train').replace(category_map)
y_test = y_test.to_frame(name='y_test').replace(category_map)

# save to files for loading to XAI later
X_train.reset_index().to_parquet('./cache/medsynth/xgboost/X_train.parquet')
X_test.reset_index().to_parquet('./cache/medsynth/xgboost/X_test.parquet')
y_train.reset_index().to_parquet('./cache/medsynth/xgboost/y_train.parquet')
y_test.reset_index().to_parquet('./cache/medsynth/xgboost/y_test.parquet')
y_pred.reset_index().to_parquet('./cache/medsynth/xgboost/y_pred.parquet')
joblib.dump(model, './cache/medsynth/xgboost/model.joblib')

