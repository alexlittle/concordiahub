# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html



import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from concordiahub._version import __version__
release = __version__
version = ".".join(__version__.split(".")[:2])


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'concordiahub'
copyright = '2026, Alex Little'
author = 'Alex Little'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme =  'furo'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_css_files = ['custom.css']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0d4f4f",
        "color-brand-content": "#0d4f4f",
        "color-highlight-on-target": "#fdf0e6",
        "color-sidebar-background": "#D4E4E477",
        "color-link": "#0d4f4f",
        "color-link--hover": "#0d4f4f",
        "color-link--visited": "#0d4f4f",
        "color-link--visited--hover": "#0d4f4f",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4C8C8C",
        "color-brand-content": "#4C8C8C",
        "color-highlight-on-target": "#3a3020",
        "color-link": "#6BA8A8",
        "color-link--hover": "#6BA8A8",
        "color-link--visited": "#6BA8A8",
        "color-link--visited--hover": "#6BA8A8",
        "color-sidebar-background": "#1a2626",
        "color-sidebar-background-border": "#2e3d3d",
        "color-sidebar-brand-text": "#6BA8A8",
        "color-sidebar-caption-text": "#6BA8A8",
        "color-sidebar-link-text--top-level": "#6BA8A8",
        "color-sidebar-item-background--hover": "#243333",
    }
}
