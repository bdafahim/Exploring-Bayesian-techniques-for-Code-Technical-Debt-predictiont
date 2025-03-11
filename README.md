#  *"Exploring Bayesian techniques for Code Technical Debt predictiont"*

This package contains all the Python codes to conduct the data preprocessing and Bayesian analysis using different modesl which is used in our study 'Exploring Bayesian techniques for Code Technical Debt predictiont'.
Here is the paper link: [Exploring Bayesian techniques for Code Technical Debt Prediction](https://oulurepo.oulu.fi/handle/10024/53208).

## Overview

This package focuses on technical debt prediction through Bayesian analysis, examining how different models and estimators can effectively forecast technical debt in software projects.The primary objective of this research is to assess the reliability and effectiveness of various Bayesian models in predicting Code Technical Debt(TD), drawing on historical data from the Technical Debt Dataset. By investigating the application of these models across different project datasets, this study aims to offer developers and project managers predictive tools for better decision-making in managing software quality. 

## Prerequisites

Running the code requires Python3.9. See installation instructions [here](https://www.python.org/downloads/).
You might also want to consider using [virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).
Ensure you have the following packages installed and configured on your system:
### Orbit-ML
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

### PyBats
PyBATS is used for dynamic modeling of time series data.

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

### pyDLM
pyDLM is useful for Bayesian time series analysis with dynamic linear models.

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



## Structure of the replication package

The replication package provides two different directories. The first one provides the codes for running the study and the second one
provides the initial data files obtained from the dataset specified in the paper.

### Codes

The folder `../codes/` contain the following files:
```commons.py```
```main.py```
```modules.py```
```bayesian_prediction_orbit_DLT.py```
```bayesian_prediction_orbit_ETS.py```
```bayesian_pybats_dglm.py```
```bayesian_prediction_pyDLM.py```
```preprocessing.py```
```tsDataPreparation.py```

### Data
The folder `../data/`  contain the following files:
```biweekly_data/```
```monthly_data/```
```complete_data/```
```raw-data/```
```Decomposition Plot/```
```ORBIT_ML_DLT_Result/```
```ORBIT_ML_ETS_Result/```
```PYBATS_DGLM_Results/```
```PyDLM_Results/```

### Results
Results for individual model are available in the directory `../data/{model_name}_Result/Periodicity/Results`. Here {model_name}_Result can be
```ORBIT_ML_DLT_Result```,
```ORBIT_ML_ETS_Result```,
```PYBATS_DGLM_Results``` or
```PyDLM_Results```
The Periodicity can be biweekly, monthly, or original, depending on the dataset used.
Additionally, confidence intervals for all the models are provided in the directory `../data/{model_name}_Result/Periodicity/Confidence Intervals`

### Data Visualization
All the plots for individual models are available in the directory `../data/Result_Folder/Periodicity/Plots`. The Result_Folder  names for each model are detailed above in the 'Data' section. For instance, to access all plots for the Orbit-ML DLT model with biweekly data, navigate to `../data/
ORBIT_ML_DLT_Result/biweekly/Plots`

Also, we have provided decomposition plots for DLT and ETS models. Decomposition plots can be found ../data/Decomposition Plot/model/Periodicity`. where model corresponds to either DLT or ETS, and Periodicity refers to biweekly, monthly, or complete, depending on the desired plots.



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

### Stage 2: ```Orbit-ML Damped Local Trend(DLT)```

- Executes script ```bayesian_prediction_orbit_DLT.py``` for executing bayesian Analysis with DLT for biweekly, monthly and complete dataset.

### Stage 3: ```Orbit-ML Exponential Smoothing(ETS)```

- Executes script ```bayesian_prediction_orbit_ETS.py```  for executing bayesian Analysis with ETS for  biweekly, monthly and complete dataset.

### Stage 4: ```PyBats Dynamic Generalized Linear Model(DGLM)```

- Executes script ```bayesian_pybats_dglm.py``` for executing bayesian Analysis with DGLM for biweekly, monthly and complete dataset.


### Stage 5: ```pyDLM Dynamic Linear Model(DLM)```

- Executes script ```bayesian_prediction_pyDLM.py``` for executing bayesian Analysis with DLM for biweekly, monthly and complete dataset.
  


