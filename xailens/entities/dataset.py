import pandas as pd

from xailens.entities.modeleval import ModelEval
from xailens import constants


class Dataset:

    def __init__(self, dataset_id: str,  modelevals: list[ModelEval]):
        self.dataset_id = dataset_id
        self.modelevals = modelevals

    def get_metrics(self) -> pd.DataFrame:
        rows = {me.get("model_display_name"): me.get_metrics() for me in self.modelevals}
        df = pd.DataFrame(rows).T
        df.index.name = "model"
        df.columns.name = None
        return df

    def get_instance_consensus_ids(self) -> pd.Index:
        """Returns instance IDs common to all models."""
        common_ids = None
        for me in self.modelevals:
            y_true, _ = me.load_targets()
            ids = set(y_true[me.get(constants.DATA_SCHEMA_ID_COLUMN)].tolist())
            if common_ids is None:
                common_ids = ids
            else:
                common_ids = common_ids.intersection(ids)
        return common_ids

    def get_instance_consensus(self) -> dict:
        common_ids = self.get_instance_consensus_ids()

        # For each model, get predictions for the common instance IDs
        model_preds = {}
        y_true_common = None
        for me in self.modelevals:
            y_true, y_pred = me.load_targets()
            id_col = me.get(constants.DATA_SCHEMA_ID_COLUMN)
            pred_col = me.get(constants.DATA_SCHEMA_PREDICTION_COLUMN)
            target_col = me.get(constants.DATA_SCHEMA_TARGET_COLUMN)

            y_pred_filtered = y_pred[y_pred[id_col].isin(common_ids)]
            y_true_filtered = y_true[y_true[id_col].isin(common_ids)]

            model_preds[me.get("model_display_name")] = {
                "preds": y_pred_filtered.set_index(id_col),
                "pred_col": pred_col,
                "target_col": target_col,
            }
            if y_true_common is None:
                y_true_common = y_true_filtered.set_index(id_col)
                target_col_common = target_col

        correct_df = pd.DataFrame({
            model_id: (data["preds"][data["pred_col"]].values == y_true_common[target_col_common].values)
            for model_id, data in model_preds.items()
        }, index=y_true_common.index)

        total = len(common_ids)
        all_correct = int(correct_df.all(axis=1).sum())
        all_incorrect = int((~correct_df).all(axis=1).sum())
        models_disagree = total - all_correct - all_incorrect

        return {
            "total": total,
            "all_correct": all_correct,
            "all_incorrect": all_incorrect,
            "models_disagree": models_disagree,
        }

