


def render_datasets(datasets: [str]):

    parts = []

    parts.append("<div class='dataset-grid'>")
    for dataset in datasets:

        parts.append(f"""
            <a href="dataset?dataset={dataset}" target="_self" class="dataset-card" >{dataset}</a>
        """)
    parts.append("</div>")
    return "\n".join(parts)