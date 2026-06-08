from xai_compare.registry.model_registry import register_model
from xai_compare.adapters.models.model_base import ModelAdapter
from xai_compare import strings

@register_model("llm")
class LLMAdapter(ModelAdapter):

    display_name = strings.LLM_DISPLAY_NAME

    def prepare(self):
        # no model to prepare/return for LLM
        pass