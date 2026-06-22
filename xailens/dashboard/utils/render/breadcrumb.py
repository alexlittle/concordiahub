from xailens.entities.modeleval import ModelEval
from xailens.formatting import format_model_name



def display_breadcrumbs(modeleval_a: ModelEval=None, modeleval_b: ModelEval=None, instance_id:str = None) -> str:
    parts = []
    parts.append("<a href='/' target='_self'>Home</a>")

    if modeleval_a and modeleval_b:
        if instance_id:
            parts.append(f"""
            <a href='/overview?runs={modeleval_a.get('model_uuid')},{modeleval_b.get('model_uuid')}' target='_self'>
            {format_model_name(modeleval_a)} &amp; {format_model_name(modeleval_b)}</a>
            """)
        else:
            parts.append(f"""
                        {format_model_name(modeleval_a)} &amp; {format_model_name(modeleval_b)}
                        """)

    if instance_id:
        parts.append(f"{instance_id}")

    return " &nbsp;&gt;&nbsp; ".join(parts)