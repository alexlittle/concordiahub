# xaicompare/artifacts.py
import os
import json
from datetime import datetime

from xai_compare import RunContext


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
            self.dirname = os.path.join(BASE_DIR, 'runs', datetime.now().strftime("%Y%m%d-%H%M%S"))

        os.makedirs(self.dirname, exist_ok=True)

    def write_metadata(self):
        obj = {
            "model_type": self.ctx.model_type,
            "xai_method": self.ctx.xai_method,
            "datetime": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "dataset_id": self.ctx.dataset_id,
            "run_id": self.ctx.run_id,
            "model_id": self.ctx.model_id
        }
        with open(os.path.join(self.dirname, "metadata.json"), "w") as f:
            json.dump(obj, f, indent=4)

    def write_parquet(self, name, df):
        out = os.path.join(self.dirname, name)
        df.to_parquet(out, index=False)