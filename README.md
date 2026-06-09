# XAILens

Under development as part of my MSc project, XAI Compare allows model developers to compare Explainable AI methods and 
explanations between models, including scikit-learn models and LLMs.



As this is a proof of concept framework, not all model, XAI method or visualisation type has been developed. However,
the architecture (adapter pattern) allows new model types, XAI methods and visualisations to be added without large 
changes to the codebase.


## Install

* Install with `pip install xailens`
* Lanuch the dashboard with `xailens-dash`


## Source code/Development
* This repo uses Git LFS for large files. Before cloning, install Git LFS:
    ```bash
    sudo apt install git-lfs  # Ubuntu
    brew install git-lfs      # macOS
    git lfs install
      ```




## 🚀 CI & Code Quality

### 🧪 Continuous Integration
[![CI](https://github.com/alexlittle/xaicompare/actions/workflows/ci.yml/badge.svg)](https://github.com/alexlittle-github/xaicompare/actions)

### 🔎 SonarCloud — Quality Overview
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=coverage)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)

### 📈 Detailed Metrics
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)
[![Security](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)

### 🐛 Issues Breakdown
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=bugs)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=alexlittle-github_xaicompare&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=alexlittle-github_xaicompare)