import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from xailens import runner
from xailens.exceptions import InvalidConfigurationError, MissingDataError
from xailens.entities.context import ModelContext, ModelData, DataSchema

MODULE_UNDER_TEST = "xailens.runner"


# Test with no context provided
def test_runner_no_context() -> None:
    with pytest.raises(TypeError):
        runner.run()

# Test with empty context
def test_runner_empty_context() -> None:
    with pytest.raises(InvalidConfigurationError, match="Context"):
        runner.run(None)
