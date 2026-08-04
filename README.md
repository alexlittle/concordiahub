# XAI Lens

Developed as part of my MSc project, XAI Lens allows model developers to compare
Explainable AI methods and explanations between models, including scikit-learn 
models and LLMs.

As this is a proof of concept framework, not all model, XAI method or 
visualization types have been developed. However, the architecture allows new 
model types, XAI methods and visualizations to be added without large changes to
the codebase.

## Demo

Demo available at: https://xailens.alexlittle.net/

## Install

* Install with `pip install xailens`
* Launch the dashboard with `xailens-dash`


## Repository Structure

An overview of this repository's directory and file structure. Note that not 
every directory/file is included here.

```
.
├── docs                     # Documentation in Sphinx RST format
├── examples                 # Example models used in the demo              
├── tests                    # Tests - dir/file structure mirrors the xailens 
│                            # core package structure
├── xailens                    # Core XAI Lens code
│   ├── adapters                 # Adapters
│   │   ├── explainers             # Adapters for XAI methods (eg TreeShap)
│   │   ├── models                 # Adapters for model types (eg LLM, XGBoost)
│   │   └── viz                    # Adapters for visualizations (eg bar chart)
│   ├── dashboard                # Streamlit dashboard application
│   │   ├── assets                 # CSS, logo/icons etc
│   │   ├── pages                  # Streamlit pages
│   │   ├── utils                  # Helper functions/utilities/components for pages
│   │   ├── app.py                 # Dashboard start up
│   │   ├── explorer.py            # Memory caching of model_explorer class
│   │   └── settings.py            # Dashboard settings
│   ├── entities                 # Helper classes for library and dashboard
│   │   ├── context.py             # Dataclasses for sending model to framework
│   │   ├── dataset.py             # Dataset class (represents a dataset)
│   │   ├── model_explorer.py      # Explorer class (for dataset, models available)
│   │   ├── model.py               # Model class
│   │   └── schema.py              # Dataclass for model data description
│   ├── registry                 # Adapter registries
│   │   ├── autodiscover.py        # Autodiscovers the adapters available
│   │   ├── model_registry.py      # Registry for model adapters
│   │   ├── viz_registry.py        # Registry for visualization adapters
│   │   └── xai_registry.py        # Registry for XAi method adapters
│   ├── storage                  # Storage classes
│   │   ├── filesystem_store.py    # Filesystem store class
│   │   └── store_base.py          # Base (abstract) store class
│   ├── artifacts.py             # Artifact store - manages models and storage
│   ├── cli.py                   # Command line interface for launching dashboard
│   ├── config.py                # XAI Lens configuration
│   ├── exceptions.py            # Custom exceptions
│   ├── runner.py                # Orchestrator/entry point for loading a model
│   ├── strings.py               # Strings used throughout library
│   └── _version.py              # library version number
├── LICENSE                    # GNU 3 open source license
├── pyproject.toml             # Project/pip install file
├── README.md                  # This readme file
├── requirements-dev.txt       # Required packages if developing more features 
├── requirements-examples.txt  # Required packages for running the example models
├── requirements.txt           # Core required packages
├── setup.py                   # Project/pip install file
└── sonar-project.properties   # SonarQube configuration file   
```

## Source code/Development

This repo uses Git LFS for large files. Before cloning, install Git LFS:
```bash
sudo apt install git-lfs 
git lfs install
```


## CI & Code Quality

### Continuous Integration

[![CI](https://github.com/alexlittle/xailens/actions/workflows/ci.yml/badge.svg)](https://github.com/alexlittle/xailens/actions)

### SonarCloud - Quality Overview

[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xailens)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=coverage)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)

### Detailed Metrics

[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xailens)
[![Security](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xailens)

### Issues Breakdown

[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=bugs)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xailens)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xailens)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xailens&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xailens)