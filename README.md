# XAI Lens

Under development as part of my MSc project, XAI Lens allows model developers to compare Explainable AI methods and 
explanations between models, including scikit-learn models and LLMs.



As this is a proof of concept framework, not all model, XAI method or visualisation type has been developed. However,
the architecture (adapter pattern) allows new model types, XAI methods and visualisations to be added without large 
changes to the codebase.

## Live Demo

A demo is available at: https://xai.alexlittle.net/ 

## Install

* Install with `pip install xailens`
* Launch the dashboard with `xailens-dash`


## Source code/Development
* This repo uses Git LFS for large files. Before cloning, install Git LFS:
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