

def build_filter_pills_html(active: str = "all",
                            param: str = "pill_filter",
                            pill_options: dict[str, str] | None = None,
                            variant: str = "model",
                            current_params: dict[str, str] | None = None) -> str:

    if pill_options is None:
        pill_options = {"all": "All model_data"}
    if active not in pill_options:
        active = "all"

    pills =[]
    for key, label in pill_options.items():
        params = dict(current_params or {})
        params[param] = key
        query_string = "&".join(f"{k}={v}" for k, v in params.items())

        css_class = f"pill-{variant} pill-{variant}--active" if key == active else f"pill-{variant}"
        pills.append(
            f'<a href="?{query_string}" target="_parent" class="{css_class}">{label}</a>'
        )
    return f'<div class="pill-bar">{"".join(pills)}</div>'

def apply_dataset_filter(pairs: list[list], dataset_key: str) -> list[list]:
    if dataset_key == "all":
        return pairs
    return [pair for pair in pairs if pair[0].get("dataset_id") == dataset_key]

def apply_model_filter(pairs: list[list], pill_key: str) -> list[list]:
    """Filter comparable run pairs by model type pairing."""
    if pill_key == "all":
        return pairs
    a, _, b = pill_key.partition("_vs_")
    return [
        pair for pair in pairs
        if {pair[0].get("model_type"), pair[1].get("model_type")} == {a, b}
    ]