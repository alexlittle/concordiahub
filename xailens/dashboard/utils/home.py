

from xailens.entities.model_explorer import ModelExplorer
from xailens import strings


def render_datasets(explorer: ModelExplorer):
    datasets = explorer.get_datasets()

    parts = ["<div class='dataset-grid'>"]

    for dataset in datasets:
        num_models = len(explorer.get_models_for_dataset(dataset))
        num_text = strings.HOME_DATASET_NUM_MODELS.format(count=num_models)

        parts.append(f"""
            <a href="dataset?dataset={dataset}" target="_self" class="dataset-card">
                <span class="dataset-card-title">{dataset}</span>
                <span class="dataset-card-count">{num_text}</span>
            </a>
        """)
        explorer.get_models_for_dataset(dataset)
    parts.append("</div>")
    return "\n".join(parts)


