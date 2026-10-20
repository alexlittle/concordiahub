# Concordia Hub

Developed as part of my MSc project, Concordia Hub allows model developers to compare
Explainable AI methods and explanations between models, including scikit-learn 
models and LLMs.

As this is a proof-of-concept framework, not all model, XAI method or 
visualization types have been developed. However, the architecture allows new 
model types, XAI methods and visualizations to be added without large changes to
the codebase.

## Demo

Demo available at: https://concordiahub.alexlittle.net/

## Install

* Install with `pip install concordiahub`
* Launch the dashboard with `concordiahub-dash`


## Repository Structure

An overview of this repository's directory and file structure. Note that not 
every directory/file is included here. All code required for model registration and dashboard display is in the 
`concordiahub` directory, other top-level directories are supporting scripts, examples, tests and documentation.

```
.
├── docs                     # Placeholder for documentation in Sphinx RST format
├── evaluation               # Scripts used to measure feature alignment            
├── tests                    # Tests - dir/file structure mirrors the concordiahub 
│                            # core package structure
├── concordiahub               # Core Concordia Hub code
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
│   │   └── xai_registry.py        # Registry for XAI method adapters
│   ├── storage                  # Storage classes
│   │   ├── filesystem_store.py    # Filesystem store class
│   │   └── store_base.py          # Base (abstract) store class
│   ├── artifacts.py             # Artifact store - manages models and storage
│   ├── cli.py                   # Command line interface for launching dashboard
│   ├── config.py                # Concordia Hub configuration
│   ├── exceptions.py            # Custom exceptions
│   ├── runner.py                # Orchestrator/entry point for loading a model
│   ├── strings.py               # Strings used throughout library
│   └── _version.py              # library version number
├── LICENSE                    # GNU 3 open source license
├── pyproject.toml             # Project/pip install file
├── README.md                  # This readme file
├── setup.py                   # Project/pip install file
└── sonar-project.properties   # SonarQube configuration file   
```


## CI & Code Quality

### Continuous Integration

[![CI](https://github.com/alexlittle/concordiahub/actions/workflows/ci.yml/badge.svg)](https://github.com/alexlittle/concordiahub/actions)

### SonarCloud - Quality Overview

[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_concordiahub)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=coverage)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)

### Detailed Metrics

[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_concordiahub)
[![Security](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_concordiahub)

### Issues Breakdown

[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=bugs)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_concordiahub)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_concordiahub)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_concordiahub&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_concordiahub)