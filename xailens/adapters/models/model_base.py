"""
Base abstract class for all Model adapters
"""

from abc import ABC

from typing import Optional, ClassVar
from xailens.config import ScikitLearnModel

class ModelAdapter(ABC):
    """
    Abstract class for all Model adapters
    """
    display_name: str | None = None
    key: ClassVar[str] = None

    def __init__(self, model: Optional[ScikitLearnModel] = None):
        self.model = model

    def prepare(self) -> ScikitLearnModel:
        """
        Returns the model

        Returns:
            ScikitLearnModel: The model
        """
        return self.model



