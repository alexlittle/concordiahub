import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


from helpers import check_gpu
from examples import config


device = check_gpu()
print(f"System check: {device} detected.")

# data to extract from the input file
df = pd.read_csv(config.MEDSYNTH_FULL_FILE)

y = df.set_index(config.SAMPLE_UUID_COL)['target'].astype('category').cat.codes

# vectorize texts
scenario_tfidf = TfidfVectorizer(
    max_features=3000,
    sublinear_tf=True,
    max_df=0.8,
    stop_words='english'
)

X = scenario_tfidf.fit_transform(df['scenario'])

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
model.fit(X_train, y_train)

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

