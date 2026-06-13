from xailens.formatting import format_datetime


def test_format_datetime():
    result = format_datetime("20240525-143000")
    assert result == "14:30 25 May 2024"