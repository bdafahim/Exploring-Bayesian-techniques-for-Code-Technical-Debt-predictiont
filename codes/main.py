"""
This project executes a series of TSA techniques as well as ML models

NOTE: The data collection and preprocessing has been already performed.
"""

import os

from preprocessing import preprocessing
from tsDataPreparation import data_prepare
from bayesian_prediction_orbit_DLT import bayesian_orbit_DLT
from bayesian_prediction_orbit_ETS import bayesian_orbit_ets
from bayesian_pybats_dglm import bayesian_dglm
from bayesian_prediction_pybsts import bayesian_pybsts
from bayesian_prediction_orbit_KTR import bayesian_orbit_ktr
#from bayesian_prediction_pymc import bayesian_pymc
from bayesian_prediction_pyDLM import bayesian_pyDLM
from change_point_detection_pyDLM import bayesian_change_point_detection_pyDLM
from bayesian_change_point_detection_online import bayesian_change_point_detection


from commons import PREPROCESSING, DGLM, ORBIT, PYBSTS, PYMC, PYDLM, CHANGEPOINT


def main():

    # Preprocess the data and generate the clean tables for analysis with biweekly, monthly and complete data
    if PREPROCESSING:
        #preprocessing()
        data_prepare()
    if DGLM:
        bayesian_dglm()
    if ORBIT:
        bayesian_orbit_DLT()
        #bayesian_orbit_ets()
        #bayesian_orbit_ktr()
    if PYBSTS:
        bayesian_pybsts()
    if PYDLM:
        bayesian_pyDLM()
    if CHANGEPOINT:
        bayesian_change_point_detection()
        #bayesian_change_point_detection_pyDLM()



if __name__ == '__main__':
    main()