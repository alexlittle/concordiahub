

from xailens.dashboard.services import get_explorer
from xailens.exceptions import InvalidUrlParamsError
from xailens import strings

def get_runs_from_url_param(query_params):
    explorer = get_explorer()
    runs_param = query_params.get("model_data")
    if not runs_param:
        raise InvalidUrlParamsError(strings.ERR_NO_RUNS)

    run_ids = runs_param.split(",")

    runs = explorer.get_runs_by_uuids(run_ids)

    if len(runs) != 2:
        raise InvalidUrlParamsError(strings.ERR_NOT_2_RUNS)

    run_a, run_b = runs
    return run_a, run_b