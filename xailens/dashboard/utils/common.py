import streamlit as st

from xailens.dashboard import settings
from xailens import strings
from xailens.dashboard.assets.icons import help_icon


def display_breadcrumbs(dataset: str = None,
                        instance_group: str = None,
                        instance_uuid:str = None) -> str:
    parts = ["<a href='/' target='_self'>Home</a>"]

    models = st.query_params.get('models')
    if models:
        model_param = f"&models={models}"
    else:
        model_param = ""

    if dataset:
        if instance_group:
            parts.append(f"<a href='/{settings.DATASET_FILENAME}?dataset={dataset}{model_param}' target='_self'>{dataset}</a>")
        else:
            parts.append(f"{dataset}")

    if instance_group:
        if instance_uuid:
            parts.append(f"<a href='/{settings.INSTANCES_FILENAME}?dataset={dataset}{model_param}&instances={instance_group}' target='_self'>{strings.INSTANCE_GROUPS[instance_group]}</a>")
        else:
            parts.append(strings.INSTANCE_GROUPS[instance_group])

    if instance_uuid:
        parts.append(f"{instance_uuid}")
    
    return f"<div class='breadcrumbs'>{' &nbsp;&gt;&nbsp; '.join(parts)}</div>"


def title_with_help_icon(title: str, help_text:str) -> str:
    return f"""
            <div class="card-header-row">
                <h3>{title}</h3>
                {help_icon(help_text)}
            </div>
        """
