"""
General settings for the Streamlit app
"""

from typing import Final

HOME_FILENAME: Final[str] = "home"
DATASET_FILENAME: Final[str] = "dataset"
INSTANCES_FILENAME: Final[str] = "instances"
INSTANCE_DETAIL_FILENAME: Final[str] = "instancedetail"


# settings for the interactive table of instances (iTable)
ITABLE_MAX_BYTES: Final[int] = 5000000
ITABLE_MAX_ROWS: Final[int] = 5000
ITABLE_ROWS_PER_PAGE: Final[int] = 25
