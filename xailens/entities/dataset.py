import pandas as pd

from xailens.entities.modeleval import ModelEval

class Dataset:

    def __init__(self, dataset_id: str,  modelevals: [ModelEval]):
        self.dataset_id = dataset_id
        self.modelevals = modelevals

    def get_metrics(self) -> pd.DataFrame:
        rows = {me.get("model_display_name"): me.get_metrics() for me in self.modelevals}
        df = pd.DataFrame(rows).T
        df.index.name = "model"
        df.columns.name = None
        return df
