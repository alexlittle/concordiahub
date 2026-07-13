"""
Base abstract class for all Model adapters
"""

from abc import ABC

from xailens.config import TreeModel

class ModelAdapter(ABC):
    """
    Abstract class for all Model adapters
    """
    display_name: str | None = None

    def __init__(self, model: TreeModel =None):
        self.model = model

    def prepare(self):
        """
        Returns the model

        Returns:
            TreeModel: The model
        """
        return self.model



