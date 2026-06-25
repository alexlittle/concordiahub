from xailens.dashboard import settings


def display_breadcrumbs(dataset: str = None,
                        instance_id:str = None) -> str:
    parts = ["<a href='/' target='_self'>Home</a>"]

    if dataset:
        parts.append(f"<a href='/{settings.DATASET_FILENAME}?dataset={dataset}' target='_self'>{dataset}</a>")


    if instance_id:
        parts.append(f"{instance_id}")
    
    return f"<div class='breadcrumbs'>{' &nbsp;&gt;&nbsp; '.join(parts)}</div>"
