#  *"Technical Debt Prediction and Simulation"*

This package contains all the Python code to conduct the data preprocessing and Bayesian analysis using different modesl which is used in our study 'Technical debt prediction using Bayesian Analysis'.

## Overview

 This package focuses on technical debt prediction through Bayesian analysis, examining how different models and estimators can effectively forecast technical debt in software projects. The primary objective of this research is to assess the reliability and effectiveness of various Bayesian models in predicting technical debt, drawing on historical data from the Technical Debt Dataset. By investigating the application of these models across different project datasets, this study aims to offer developers and project managers predictive tools for better decision-making in managing software quality. 

### Prerequisites

Running the code requires Python3.9. See installation instructions [here](https://www.python.org/downloads/).
You might also want to consider using [virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).
Ensure you have the following tools installed and configured on your system:
## Orbit-ML
**Orbit-ML** is a library for Bayesian time series forecasting and inference.
Install from PyPi:
```bash
pip install orbit-ml
```

Install from Github:

```bash
git clone https://github.com/uber/orbit.git
cd orbit
pip install -r requirements.txt
pip install .
```

## PyBats
PyBATS (Python for Bayesian Time Series) is used for dynamic modeling of time series data.

PyBATS is hosted on PyPI and can be installed with pip:

```bash
pip install pybats
```
Install from Github

```bash
git clone https://github.com/uber/orbit.git
cd orbit
pip install -r requirements.tx
```

## pyDLM
pyDLM (Dynamic Linear Models with Bayesian Inference) is useful for Bayesian time series analysis with dynamic linear models.

You can  get the package from PyPI:

```bash
pip install pydlm
```

You can also get the latest from github:

```bash
git clone git@github.com:wwrechard/pydlm.git pydlm
cd pydlm
sudo python setup.py install
```

## pyBSTS
pyBSTS is a Python implementation of Bayesian Structural Time Series (BSTS) models.

You can  get the package from PyPI:

```bash
pip install pybst
```


## Structure of the replication package

The replication package provides two different directories. The first one provides the codes for running the study and the second one
provides the initial data files obtained from the dataset specified in the paper.

### Codes

The folder `../codes/` should contain the following files:
```commons.py```
```main.py```
```modules.py```
```bayesian_prediction_orbit_DLT.py```
```bayesian_prediction_orbit_ETS.py```
```bayesian_pybats_dglm.py```
```bayesian_prediction_pybsts.py```
```bayesian_prediction_pyDLM.py```
```preprocessing.py```
```tsDataPreparation.py```

### Data
The folder `../data/` should contain the following files:
```biweekly_data/```
```monthly_data/```
```complete_data/```
```raw-data/```





## Running the code

NOTE 1: Please, find the `DATA_PATH` global variable in the `commons.py` script and define the path where the program should create all the needed results.

NOTE 2: The different stages of the study execution are splitted in the ```main.py``` script, from the boolean definitions in
```commons.py``` practitioners can decide which stages want to be manipulated or re-executed again without affecting the other stages. Single or multiple models can be run depending on boolean variable in commons.py.
For a complete execution, set all the boolean global variables to ```True```


### Stage 1: ```PREPROCESSING```

- Executes scrips ```preprocessing.py``` and ```tsDataPreparation.py```.
- In these scripts the raw data from the Technical Debt dataset are converted into project-divided csv files repeated in 
```biweekly```, ```monthly``` and ```complete``` format by first performing data cleaning and preprocessing using the techniques
described in the paper.

### Stage 2: ```Orbit-ML DLT```

- Executes script ```bayesian_prediction_orbit_DLT.py``` for executing bayesian Analysis with DLT model for biweekly, monthly and complete dataset:

### Stage 3: ```Orbit-ML ETS```

- Executes script ```bayesian_prediction_orbit_ETS.py```  for executing bayesian Analysis with ETS model for biweekly, monthly and complete dataset:

### Stage 4: ```PyBats DGLM```

- Executes script ```bayesian_pybats_dglm.py``` for executing bayesian Analysis with DGLM model for biweekly, monthly and complete dataset:

### Stage 5: ```pyBSTS DLM```

- Executes script ```rbayesian_prediction_pybsts.p``` for executing bayesian Analysis with pyBSTS DLM model for biweekly, monthly and complete dataset:

### Stage 5: ```pyDLM DLM```

- Executes script ```bayesian_prediction_pyDLM.py``` for executing bayesian Analysis with pyDLM DLM model for biweekly, monthly and complete dataset:

