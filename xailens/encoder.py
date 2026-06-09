import pandas as pd
from sentence_transformers import SentenceTransformer
from xailens.config import EMBEDDING_ENCODER


class XAITextEncoder:

    @staticmethod
    def encode_feature_names(df) -> pd.DataFrame:
        model = SentenceTransformer(EMBEDDING_ENCODER)
        feature_names = df.columns.tolist()
        embeddings = model.encode(feature_names)
        return pd.DataFrame(embeddings, index=feature_names)

