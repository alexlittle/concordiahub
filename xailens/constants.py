

DATA_SCHEMA_ID_COLUMN = 'data_schema.instance_id_column'
DATA_SCHEMA_TARGET_COLUMN = 'data_schema.target_column'
DATA_SCHEMA_PREDICTION_COLUMN = 'data_schema.prediction_column'


# to remove?
PREDICTION_COLUMN_LABEL = "prediction"
TRUE_VALUE_COLUMN_LABEL = "True Value"


INSTANCE_GROUPS = {
    "all": "All Instances",
    "allcorrect": "All Correct Instances",
    "allincorrect": "All Incorrect Instances",
    "disagree": "Models Disagree"
}

# for instances df display
COL_INSTANCE_UUID = "instance_uuid"
COL_TRUE_LABEL = "true_label"

DISPLAY_NAMES = {
    COL_INSTANCE_UUID: "Instance ID",
    COL_TRUE_LABEL: "Ground Truth",
}

# to remove?
COMPARISON_CORRECT = "correct"
COMPARISON_BOTH_CORRECT = "Both correct"
COMPARISON_BOTH_INCORRECT_SAME = "Both incorrect - same prediction"
COMPARISON_BOTH_INCORRECT_DIFF = "Both incorrect - different predictions"