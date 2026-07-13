"""
Custom exceptions for xailens module
"""

class InvalidConfigurationError(Exception):
    """
    Exception raised for invalid configuration
    """
    pass

class MissingDataError(Exception):
    """
    Exception raised for missing data
    """
    pass

class InvalidSelectionError(Exception):
    """
    Exception raised for invalid selection, eg
    """
    pass

class InvalidUrlParamsError(Exception):
    """
    Exception raised for invalid url params, eg missing parameters in dashboard url
    """
    pass

class DataProcessingError(Exception):
    """
    Exception raised when data processing fails, eg with SHAP calculations
    """
    pass