"""
Tests dashboard/utils/home_helpers.py

Covers:
- render_datasets
"""

from xailens.dashboard.utils.home_helpers import render_datasets
from xailens import strings

def test_render_datasets(mock_explorer):
    """Checks that multiple dataset cards are created"""
    mock_explorer.get_datasets.return_value = ["cleveland", "medsynth"]
    num_text = strings.HOME_DATASET_NUM_MODELS.format(count=3)

    result = render_datasets(mock_explorer)

    assert result.count('class="dataset-card"') == 2
    assert '<span class="dataset-card-title">cleveland</span>' in result
    assert f'<span class="dataset-card-count">{num_text}</span>' in result