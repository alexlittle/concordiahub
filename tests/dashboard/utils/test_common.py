
from xailens.dashboard.utils.common import display_breadcrumbs

def test_no_params():
    breadcrumbs = display_breadcrumbs()
    assert "<div class='breadcrumbs'><a href='/' target='_self'>Home</a></div>" == breadcrumbs
