import pandas as pd
from sentence_transformers import SentenceTransformer
from xai_compare.config import EMBEDDING_ENCODER

class XAITextEncoder():

    def __init__(self):
        pass

    def encode(self, x):
        pass

    def encode_feature_names(self, df) -> pd.DataFrame:
        model = SentenceTransformer(EMBEDDING_ENCODER)
        feature_names = df.columns.tolist()
        embeddings = model.encode(feature_names)
        return pd.DataFrame(embeddings, index=feature_names)