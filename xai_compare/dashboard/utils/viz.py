


def select_individual_viz(viz_id: str, current: str, options: dict, existing_params: dict) -> str:
    option_list = []
    for k, v in options.items():
        if k == current:
            option_list.append(v.display_name)
        else:
            link_params = {**existing_params, viz_id: k}
            query_string = "&".join(
                f"{pk}={pv}" for pk, pv in link_params.items())
            option_list.append(
                f'<a href="detail?{query_string}" target="_self">{v.display_name}</a>'
            )
    return " | ".join(option_list)

def select_comparison_viz(current: str, options: dict, existing_params: dict) -> str:
    option_list =[]
    for k,v in options.items():
        if k == current:
            option_list.append(v.display_name)
        else:
            link_params = {**existing_params, "comp_viz": k}
            query_string = "&".join(f"{pk}={pv}" for pk, pv in link_params.items())
            option_list.append(
                f'<a href="detail?{query_string}" target="_self">{v.display_name}</a>'
            )
    return " | ".join(option_list)
