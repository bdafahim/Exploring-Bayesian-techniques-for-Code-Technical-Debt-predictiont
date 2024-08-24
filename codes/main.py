"""
This project executes a series of TSA techniques as well as ML models

NOTE: The data collection and preprocessing has been already performed.
"""

import os

from preprocessing import preprocessing
from tsDataPreparation import data_prepare
# from ml_modelling import ml_models
from ml_modelling_backward import ml_models
#from ts_modelling import ts_models
#from related_work import related_models
from ts_modelling_speed import ts_models
from related_work_speed import related_models
from result_combiner import combine_results
#from ts_simulation_future import ts_simulation
#from ts_simulation_chunks import ts_simulation_seasonal_f
from ts_simulation_regressor_combination import ts_simulation
#from ts_simulation_chunks_regressors import ts_simulation_seasonal_f

from commons import SARIMAX, SIMULATION, SIMULATION_FUTURE_POINTS,  RELATED_WORK, ML_MODELS, COMBINE_RESULTS, PREPROCESSING


def main():

    # Preprocess the data and generate the clean tables for analysis with biweekly, monthly and complete data
    if PREPROCESSING:
        #preprocessing()
        data_prepare()

    # SARIMAX modelling stage execution
    if SARIMAX:
        ts_models(seasonality=True)
        ts_models(seasonality=False)  # Replicated work
    
    if SIMULATION:
        ts_simulation(seasonality=True)
    

    # SARIMA + LM related work stage execution
    '''if RELATED_WORK:
        related_models(seasonality=True)
        related_models(seasonality=False)  # Replicated work'''

    # ML stage
    if ML_MODELS:
        ml_models()

    # Combining model assessment results.
    if COMBINE_RESULTS:
        combine_results()

    # Run cells in visualization Jupyter Notebook


if __name__ == '__main__':
    main()