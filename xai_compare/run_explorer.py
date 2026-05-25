import os
import json
from collections import defaultdict
from pathlib import Path

from xai_compare import config
from xai_compare.run import Run

#-----------------------
# Helper functions
#------------------------
def extract_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("dataset_id")


class RunExplorer():

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.runs_dir = os.path.join(BASE_DIR, config.RUNS_DIR_NAME)
        self._metadata_cache = self._load_all_metadata()
        self._uuid_index = {
            run.get("uuid"): run for run in self._metadata_cache
        }

    def get_datasets(self):
        """Gets names of all datasets stored in the runs directory

        Returns:
            set: set of distinct dataset names
        """
        datasets = set()
        root = Path(self.runs_dir)
        for meta_file in root.glob(f"*/{config.META_INFO_FILENAME}"):
            datasets.add(extract_dataset(meta_file))
        return sorted(datasets)

    def get_comparable_runs(self, n: int = 3) -> list[list[dict]]:
        """Returns groups of runs that share a dataset but have different model types
        Args:
            n: number of comparison groups to return
        Returns:
            list of groups, each group is a list of metadata dicts with same dataset, different models
        """
        datasets = defaultdict(dict)
        for run in self._load_all_metadata():
            dataset = run["dataset_id"]
            model = run["model_type"]
            if model not in datasets[dataset]:
                datasets[dataset][model] = run

        # only return datasets that have more than one model type
        comparable = [
            list(models.values())
            for models in datasets.values()
            if len(models) > 1
        ]

        return sorted(comparable, key=lambda g: max(r["datetime"] for r in g), reverse=True)[:n]

    def get_run_by_uuid(self, uuid: str) -> Run | None:
        meta = self._uuid_index.get(uuid)
        return Run(path=meta["_path"]) if meta else None

    def get_runs_by_uuids(self, uuids: list[str]) -> list[Run]:
        runs = []
        for u in uuids:
            meta = self._uuid_index.get(u)
            if meta:
                runs.append(Run(path=meta["_path"]))
        return runs

    #---------------------------
    # Private helper functions
    #-----------------------------
    def _load_all_metadata(self) -> list[dict]:
        """Cache all metadata from disk"""
        if hasattr(self, '_metadata_cache'):
            return self._metadata_cache

        root = Path(self.runs_dir)
        runs = []
        for meta_file in root.glob(f"*/{config.META_INFO_FILENAME}.json"):
            with open(meta_file, "r") as f:
                meta = json.load(f)
                meta['_path'] = str(meta_file.parent)
                runs.append(meta)

        self._metadata_cache = sorted(runs, key=lambda x: x.get("datetime", ""), reverse=True)
        return self._metadata_cache