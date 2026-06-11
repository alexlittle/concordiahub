import os
import json

from collections import defaultdict
from pathlib import Path
from itertools import combinations

from xailens import config, strings
from xailens.run import Run
from xailens.dashboard.utils.formatting import format_model_name


class RunExplorer():

    def __init__(self):
        """Initializes RunExplorer"""
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.runs_dir = os.path.join(BASE_DIR, config.RUNS_DIR_NAME)
        self._metadata_cache = self._load_all_metadata()
        self._uuid_index = {
            run.get("run_uuid"): run for run in self._metadata_cache
        }

    def get_datasets(self):
        """Gets names of all datasets stored in the runs directory

        Returns:
            set: set of distinct dataset names
        """
        datasets = set()
        for run in self._metadata_cache:
            datasets.add(run["dataset_id"])
        return sorted(datasets)

    def get_models_for_dataset(self, dataset: str) -> dict:
        """ Gets all the models for the given dataset name
        Args:
            dataset str: dataset name
        Returns:
            dict of model names and runs"""
    
        models = {}
        for run in self._metadata_cache:
            if run["dataset_id"] == dataset:
                run_obj = self.get_run_by_uuid(run["run_uuid"])
                if run_obj:
                    models[format_model_name(run_obj)] = run["run_uuid"]
        return dict(sorted(models.items()))

    def get_comparable_runs(self, n: int = 10) -> list[list[dict]]:
        """Returns pairs of runs that share a dataset but have different model types
        Args:
            n: number of comparison groups to return
        Returns:
            list of groups, each group is a pair of metadata dicts with same dataset, different models
        """
        datasets = defaultdict(dict)
        for run_md in self._load_all_metadata():
            dataset = run_md["dataset_id"]
            model = run_md["model_type"]
            if run_md["model_subtype"]:
                model = f"{model}_{run_md['model_subtype']}"
            run = self.get_run_by_uuid(run_md["run_uuid"])
            if model not in datasets[dataset]:
                datasets[dataset][model] = run

        pairs = []
        for models in datasets.values():
            if len(models) > 1:
                for pair in combinations(models.values(), 2):
                    pairs.append(list(pair))

        return sorted(pairs, key=lambda g: max(r.datetime for r in g), reverse=True)

    def get_dataset_filter_options(self) -> dict[str, str]:
        """Returns dataset filter options based on available datasets in metadata."""
        datasets = sorted({
            run_md["dataset_id"]
            for run_md in self._load_all_metadata()
        })

        return {config.ALL_DATASET_OPTIONS[0]: config.ALL_DATASET_OPTIONS[1]} | {d: d for d in datasets}

    def get_model_filter_options(self) -> dict[str, str]:
        datasets = defaultdict(list)
        for run_md in self._load_all_metadata():
            datasets[run_md["dataset_id"]].append(run_md["model_type"])

        valid_pairs = set()
        for model_types in datasets.values():
            types = sorted(model_types)
            for a, b in combinations(types, 2):
                valid_pairs.add((a, b))
            for t in set(types):
                if types.count(t) >= 2:
                    valid_pairs.add((t, t))

        options = {config.ALL_COMP_OPTIONS[0]: config.ALL_COMP_OPTIONS[1]}
        for a, b in sorted(valid_pairs):
            options[f"{a}_vs_{b}"] = f"{strings.MODEL_DISPLAY_NAME[a]} vs {strings.MODEL_DISPLAY_NAME[b]}"

        return options

    def get_run_by_uuid(self, uuid: str) -> Run | None:
        """Gets a Run by its UUID
        Args:
            uuid: Run's UUID

        Returns:
            Run or None"""
        meta = self._uuid_index.get(uuid)
        return Run(path=meta["_path"], metadata=meta) if meta else None

    def get_runs_by_uuids(self, uuids: list[str]) -> list[Run]:
        """Gets a list of Runs by their UUIDs
        Args:
            List[uuid]: Run UUIDs

        Returns:
            List[Run] or None"""

        runs = []
        for u in uuids:
            meta = self._uuid_index.get(u)
            if not meta:
                continue
            runs.append(Run(path=meta["_path"], metadata=meta))
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