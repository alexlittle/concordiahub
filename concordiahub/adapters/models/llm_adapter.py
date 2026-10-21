"""
LLMAdapter class - implementation of ModelAdapter
"""


from concordiahub.registry.model_registry import register_model
from concordiahub.adapters.models.model_base import ModelAdapter
from concordiahub import strings
from concordiahub.config import ScikitLearnModel

@register_model("llm")
class LLMAdapter(ModelAdapter):
    """
    LLMAdapter class - implementation of ModelAdapter
    """

    display_name = strings.LLM_DISPLAY_NAME

    def get_model(self) -> ScikitLearnModel:
        """
        Not required/implemented for this adapter

        Returns:
            ScikitLearnModel: model object
        """
        # no model to prepare/return for LLM
        pass