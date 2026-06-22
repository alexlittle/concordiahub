

from xailens.dashboard.services import get_explorer
from xailens.exceptions import InvalidUrlParamsError
from xailens import strings

def get_modelevals_from_url_param(query_params):
    explorer = get_explorer()
    modelevals_param = query_params.get("model_data")
    if not modelevals_param:
        raise InvalidUrlParamsError(strings.ERR_NO_RUNS)

    modeleval_ids = modelevals_param.split(",")

    modelevals = explorer.get_modelevals_by_uuids(modeleval_ids)

    if len(modelevals) != 2:
        raise InvalidUrlParamsError(strings.ERR_NOT_2_RUNS)

    return modelevals