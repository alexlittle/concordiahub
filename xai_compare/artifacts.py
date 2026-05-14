# ./artifacts.py
import os
import joblib
import json
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

from xai_compare import RunContext
from xai_compare import config

# Helper functions
def extract_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("dataset_id")

class ArtifactStore:
    """
    ArtifactStore - handles all the data storage, file set up and retrieval
    """
    def __init__(self, ctx: RunContext):

        self.ctx = ctx

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        if ctx.run_id:
            self.dirname = ctx.run_id
        else:
            self.dirname = os.path.join(BASE_DIR, config.RUNS_DIR_NAME, datetime.now().strftime("%Y%m%d-%H%M%S"))

        os.makedirs(self.dirname, exist_ok=True)

    def write_metadata(self):
        obj = {
            "model_type": self.ctx.model_type,
            "xai_methods": self.ctx.xai_methods,
            "datetime": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "dataset_id": self.ctx.dataset_id,
            "run_id": self.ctx.run_id,
            "model_id": self.ctx.model_id
        }
        with open(os.path.join(self.dirname, config.META_INFO_FILENAME), "w") as f:
            json.dump(obj, f, indent=4)

    def write_model(self):
        if self.ctx.model:
            joblib.dump(self.ctx.model, os.path.join(self.dirname, config.MODEL_FILENAME))

    def write_data(self, name, data):
        if not hasattr(self.ctx, name):
            return
        out = os.path.join(self.dirname, name)
        if isinstance(data, pd.DataFrame):
            data.to_parquet(out + ".parquet", index=False)
        elif isinstance(data, pd.Series):
            data.to_frame().to_parquet(out + ".parquet")
        elif isinstance(data, np.ndarray):
            np.save(out + ".npy", data)



    def get_datasets(self):
        '''
        Returns a list of all datasets stored in the runs directory
        '''
        datasets = set()
        root = Path(self.dirname)
        for meta_file in root.glob(f"*/{config.META_INFO_FILENAME}"):
            datasets.add(extract_dataset(meta_file))
        return datasets

