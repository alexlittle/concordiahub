import os
import pandas as pd
import numpy as np

from datetime import datetime

from xai_compare import RunContext
from xai_compare import config
from xai_compare.storage.store_base import DataStore
from xai_compare.storage.filesystem_store import FilesystemStore


class ArtifactStore:
    """ArtifactStore - handles the data storage

     Args:
        ctx (RunContext): The current run configuration.
        store (DataStore): Storage backend. Defaults to FilesystemStore.
    """

    def __init__(self, ctx: RunContext, store: DataStore = None):
        self.ctx = ctx
        dirname = self._resolve_dirname(ctx)
        self.store = store or FilesystemStore(dirname)

    def _resolve_dirname(self, ctx):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        if ctx.run_id:
            return ctx.run_id
        return os.path.join(BASE_DIR, config.RUNS_DIR_NAME,
                            datetime.now().strftime("%Y%m%d-%H%M%S"))


    def write_metadata(self):
        obj = {
            "model_type": self.ctx.model_type,
            "xai_methods": self.ctx.xai_methods,
            "datetime": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "dataset_id": self.ctx.dataset_id,
            "run_id": self.ctx.run_id,
            "model_id": self.ctx.model_id
        }
        self.store.save_json(config.META_INFO_FILENAME, obj)

    def write_model(self):
        if self.ctx.model:
            self.store.save_model(config.MODEL_FILENAME ,self.ctx.model)

    def write_data(self, name, data):
        if data is None:
            return
        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        self.store.save_dataframe(f"{name}.parquet", data)

