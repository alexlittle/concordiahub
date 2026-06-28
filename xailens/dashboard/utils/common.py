import streamlit as st

from xailens.dashboard import settings
from xailens import config


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
            parts.append(f"<a href='/{settings.INSTANCES_FILENAME}?dataset={dataset}{model_param}&instances={instance_group}' target='_self'>{config.INSTANCE_GROUPS[instance_group]}</a>")
        else:
            parts.append(config.INSTANCE_GROUPS[instance_group])

    if instance_uuid:
        parts.append(f"{instance_uuid}")
    
    return f"<div class='breadcrumbs'>{' &nbsp;&gt;&nbsp; '.join(parts)}</div>"



def render_metric_card(title, value, link):
    return f"""
    <div class="xl-metric-card">
        <a href="{link}" target="_self">
            <div class="xl-metric-label">{title}</div>
            <div class="xl-metric-value">{value}</div>
        </a>
    </div>
"""
