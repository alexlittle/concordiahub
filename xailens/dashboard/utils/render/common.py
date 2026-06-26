from xailens.dashboard import settings
from xailens import constants


def display_breadcrumbs(dataset: str = None,
                        instance_group: str = None,
                        instance_id:str = None) -> str:
    parts = ["<a href='/' target='_self'>Home</a>"]

    if dataset:
        parts.append(f"<a href='/{settings.DATASET_FILENAME}?dataset={dataset}' target='_self'>{dataset}</a>")

    if instance_group:
        parts.append(constants.INSTANCE_GROUPS[instance_group])

    if instance_id:
        parts.append(f"{instance_id}")
    
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
