"""
LLMAdapter class - implementation of ModelAdapter
"""


from xailens.registry.model_registry import register_model
from xailens.adapters.models.model_base import ModelAdapter
from xailens import strings

@register_model("llm")
class LLMAdapter(ModelAdapter):
    """
    LLMAdapter class - implementation of ModelAdapter
    """

    display_name = strings.LLM_DISPLAY_NAME

    def prepare(self):
        """
        Not required/implemented for this adapter
        """
        # no model to prepare/return for LLM
        pass