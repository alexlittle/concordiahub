# XAI Lens

Developed as part of my MSc project, XAI Lens allows model developers to compare Explainable AI methods and 
explanations between models, including scikit-learn models and LLMs.

As this is a proof of concept framework, not all model, XAI method or visualisation types have been developed. However,
the architecture (adapter pattern) allows new model types, XAI methods and visualisations to be added without large 
changes to the codebase.

## Demo & Documentation

Demo available at: https://xailens.alexlittle.net/

Documentation available at: https://xailens-docs.alexlittle.net/ 

## Install

* Install with `pip install xailens`
* Launch the dashboard with `xailens-dash`


## Repository Structure

```
.
├── docs                          # Documentation in Sphinx RST format
├── examples                      # Example models used in the demo              
├── LICENSE                       # GNU 3 open Source license
├── pyproject.toml                # Project/pip install file
├── README.md                     # This readme file
├── requirements-dev.txt          # Required packages if developing more features  
├── requirements-examples.txt     # Required packages for running the example models
├── requirements.txt              # Core required packages
├── setup.py
├── sonar-project.properties      # SonarQube configuration file
├── tests                         # Tests - dir/file structure mirrors the xailens core package structure
└── xailens
    ├── adapters
    │   ├── explainers
    │   ├── models
    │   └── viz
    ├── artifacts.py
    ├── cli.py
    ├── config.py
    ├── dashboard
    │   ├── app.py
    │   ├── assets
    │   ├── explorer.py
    │   ├── pages
    │   ├── settings.py
    │   └── utils
    ├── entities
    │   ├── context.py
    │   ├── dataset.py
    │   ├── model_explorer.py
    │   ├── model.py
    │   ├── schema.py
    │   └── viz_option.py
    ├── exceptions.py
    ├── registry
    │   ├── autodiscover.py
    │   ├── model_registry.py
    │   ├── viz_registry.py
    │   └── xai_registry.py
    ├── runner.py
    ├── storage
    │   ├── filesystem_store.py
    │   └── store_base.py
    ├── strings.py
    └── _version.py
    
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

### SonarCloud — Quality Overview

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