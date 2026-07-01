import os
import json

from collections import defaultdict
from pathlib import Path
from itertools import combinations

from xailens import config, strings
from xailens.entities.model import Model


class ModelExplorer:

    def __init__(self):
        """Initializes ModelExplorer"""
        base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = os.path.join(base_dir, config.MODEL_DATA_DIR_NAME)
        self._metadata_cache = self._load_all_metadata()
        self._uuid_index = {
            model.get("model_uuid"): model for model in self._metadata_cache
        }

    def get_datasets(self):
        """Gets names of all datasets stored in the models directory

        Returns:
            set: set of distinct dataset names
        """
        datasets = set()
        for model in self._metadata_cache:
            datasets.add(model["dataset_id"])
        return sorted(datasets)

    def get_models_for_dataset(self, dataset: str) -> dict:
        """ Gets all the models for the given dataset name
        Args:
            dataset str: dataset name
        Returns:
            dict of model names and model_data"""
    
        models = {}
        for model in self._metadata_cache:
            if model["dataset_id"] == dataset:
                model_obj = self.get_model_by_uuid(model["model_uuid"])
                if model_obj:
                    models[model_obj.get("model_display_name")] = model["model_uuid"]
        return dict(sorted(models.items()))

    def get_selected_models_for_dataset(self, dataset: str, model_names: list[str]) -> [Model]:
        """ Gets all the models for the given dataset name
        Args:
            dataset str: dataset name
        Returns:
            list of Model (sorted by model_display_name)"""
        models = []
        for model in self._metadata_cache:
            if model["dataset_id"] == dataset and model["model_display_name"] in model_names:
                models.append(self.get_model_by_uuid(model["model_uuid"]))
        return sorted(models, key=lambda m: m.get("model_display_name"))

    def get_models_for_dataset(self, dataset: str) -> [Model]:
        """ Gets all the models for the given dataset name
        Args:
            dataset str: dataset name
        Returns:
            list of Model"""
        models = []
        for model in self._metadata_cache:
            if model["dataset_id"] == dataset:
                models.append(self.get_model_by_uuid(model["model_uuid"]))
        return models

    def is_valid_dataset(self, dataset) -> bool:
        """Checks if dataset is valid
        Args:
            dataset str: dataset name
        Returns:
        """
        for model in self._metadata_cache:
            if model["dataset_id"] == dataset:
                return True
        return False


    def get_model_by_uuid(self, uuid: str) -> Model | None:
        """Gets a Run by its UUID
        Args:
            uuid: Run's UUID

        Returns:
            Run or None"""
        meta = self._uuid_index.get(uuid)
        return Model(path=meta["_path"], metadata=meta) if meta else None

    def get_models_by_uuids(self, uuids: list[str]) -> list[Model]:
        """Gets a list of Runs by their UUIDs
        Args:
            List[uuid]: Run UUIDs

        Returns:
            List[Run] or None"""

        models = []
        for u in uuids:
            meta = self._uuid_index.get(u)
            if not meta:
                continue
            models.append(Model(path=meta["_path"], metadata=meta))
        return models

    #---------------------------
    # Private helper functions
    #-----------------------------
    def _load_all_metadata(self) -> list[dict]:
        """Cache all metadata from disk"""
        if hasattr(self, '_metadata_cache'):
            return self._metadata_cache

        root = Path(self.models_dir)
        models = []
        for meta_file in root.glob(f"*/{config.META_INFO_FILENAME}.json"):
            with open(meta_file, "r") as f:
                meta = json.load(f)
                meta['_path'] = str(meta_file.parent)
                models.append(meta)

        self._metadata_cache = sorted(models, key=lambda x: x.get("datetime", ""), reverse=True)
        return self._metadata_cache