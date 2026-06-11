

from xailens import config

def build_filter_pills_html(active: str = "all", param: str = "pill_filter",
    pill_options: dict[str, str] | None = None) -> str:

    if pill_options is None:
        pill_options = {config.ALL_RUN_OPTIONS[0]: config.ALL_RUN_OPTIONS[1]}

    if active not in pill_options:
        active = "all"

    pills = []
    for key, label in pill_options.items():
        css_class = "pill pill--active" if key == active else "pill"
        pills.append(
            f'<a href="?{param}={key}" target="_parent" class="{css_class}">{label}</a>'
        )

    return f"<div class='pill-bar'>{''.join(pills)}</div>"

def apply_pill_filter(pairs: list[list], pill_key: str) -> list[list]:
    """Filter comparable run pairs by model type pairing."""
    if pill_key == "all":
        return pairs
    a, _, b = pill_key.partition("_vs_")
    return [
        pair for pair in pairs
        if {pair[0].get("model_type"), pair[1].get("model_type")} == {a, b}
    ]