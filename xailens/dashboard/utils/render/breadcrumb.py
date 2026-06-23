from xailens.entities.modeleval import ModelEval
from xailens.dashboard import settings


def display_breadcrumbs(dataset: str = None,
                        modeleval_a: ModelEval=None,
                        modeleval_b: ModelEval=None,
                        instance_id:str = None) -> str:
    parts = []
    parts.append("<a href='/' target='_self'>Home</a>")

    if dataset:
        parts.append(f"<a href='/{settings.DATASET_FILENAME}?dataset={dataset}' target='_self'>{dataset}</a>")

    if modeleval_a and modeleval_b:
        if instance_id:
            parts.append(f"""
            <a href='/{settings.PW_OVERVIEW_FILENAME}?modelevals={modeleval_a.get('model_uuid')},{modeleval_b.get('model_uuid')}' target='_self'>
            {modeleval_a.get("model_display_name")} &amp; {modeleval_b.get("model_display_name")}</a>
            """)
        else:
            parts.append(f"""
                        {modeleval_a.get("model_display_name")} &amp; {modeleval_b.get("model_display_name")}
                        """)

    if instance_id:
        parts.append(f"{instance_id}")
    
    return f"<div class='breadcrumbs'>{' &nbsp;&gt;&nbsp; '.join(parts)}</div>"
