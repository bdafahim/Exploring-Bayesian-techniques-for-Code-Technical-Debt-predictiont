#  *"Technical Debt Prediction and Simulation"*

This package contains all the Python code to conduct the data preprocessing and Bayesian analysis using different modesl which is used in our study 'Technical debt prediction using Bayesian Analysis'.

## Overview

 This package focuses on technical debt prediction through Bayesian analysis, examining how different models and estimators can effectively forecast technical debt in software projects. The primary objective of this research is to assess the reliability and effectiveness of various Bayesian models in predicting technical debt, drawing on historical data from the Technical Debt Dataset. By investigating the application of these models across different project datasets, this study aims to offer developers and project managers predictive tools for better decision-making in managing software quality. 

### Prerequisites

Running the code requires Python3.9. See installation instructions [here](https://www.python.org/downloads/).
You might also want to consider using [virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).
Ensure you have the following tools installed and configured on your system:
## Orbit-ML
**Orbit-ML** is a library for Bayesian time series forecasting and inference. Install it using:
Install from PyPi:
```bash
pip install orbit-ml```

Install from Github:
```bash
git clone https://github.com/uber/orbit.git
cd orbit
pip install -r requirements.txt
pip install .
```

##PyBats
PyBATS (Python for Bayesian Time Series) is used for dynamic modeling of time series data. Install it using:
pip install pybats



## Structure of the replication package

The replication package provides two different directories. The first one provides the codes for running the study and the second one
provides the initial data files obtained from the dataset specified in the paper.

### Codes

The folder `../codes/` should contain the following files:
```commons.py```
```main.py```
```modules.py```
```bayesian_prediction_orbit_DLT.oy```
```bayesian_prediction_orbit_ETS.py```
```bayesian_prediction_pybsts.py```
```bayesian_prediction_pyDLM.py```
```ts_simulation_chunks.py```
```ts_simulation_regressor_combination.py```
```bayesian_prediction_pymc.py```
```bayesian_pybats_dglm.py```
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
```commons.py``` practitioners can decide which stages want to be manipulated or re-executed again without affecting the other stages.
For a complete execution, set all the boolean global variables to ```True```

Note 3: Python files ```ml_modelling.py```, ```related_work.py``` and ```ts_modelling.py``` execute the same logic behind the implemented python files but instead of using the built-in in functions to execute backward variable selection and parameter-tuning, they perform it hard-coded, therefore consider longer run times if you consider using them.

### Stage 1: ```PREPROCESSING```

- Executes scrips ```preprocessing.py``` and ```tsDataPreparation.py```.
- In these scripts the raw data from the Technical Debt dataset are converted into project-divided csv files repeated in 
```biweekly```, ```monthly``` and ```complete``` format by first performing data cleaning and preprocessing using the techniques
described in the paper.

### Stage 2: ```SARIMAX```

- Executes script ```ts_modelling_speed.py``` for executing the multivariate Time Series Analysis approach proposed:
  - ```seasonality=True```: Addresses the seasonality effect within the data through SARIMAX
  - ```seasonality=False```: Does not address the seasonality effect and implements just ARIMAX.

### Stage 3: ```RELATED_WORK```

- Executes script ```related_work_speed.py``` for executing the multivariate Time Series Analysis provided from the related work:
  - ```seasonality=True```: Addresses the seasonality effect within the data by implementing univariate SARIMA+LM.
  - ```seasonality=False```: Does not address the seasonality effect and implements just ARIMAX+LM.

### Stage 4: ```ML_MODELS```

- Executes script ```ml_modelling_backward.py``` for executing the considered ML models with backward variable selection procedure.

### Stage 5: ```COMBINE_RESULTS```

- Executes script ```result_combiner.py``` for combining the performance results from all the resulting projects.
  - Generates the final results from all the models by collecting their average results. The ```csv``` as well as ```LaTex``` tables can be found in ```../data/final_results/```.

### Stage 6: ```Visualization of the results```

- For the sake of flexibility, multiple visualization options apart from the ones displayed in the paper can be obtained by running all the cells existing in the Jupyter Notebook ```visualization.ipynb```.


