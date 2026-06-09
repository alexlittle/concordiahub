from xailens.registry.model_registry import register_model
from xailens.adapters.models.model_base import ModelAdapter
from xailens import strings

@register_model("llm")
class LLMAdapter(ModelAdapter):

    display_name = strings.LLM_DISPLAY_NAME

    def prepare(self):
        # no model to prepare/return for LLM
        pass