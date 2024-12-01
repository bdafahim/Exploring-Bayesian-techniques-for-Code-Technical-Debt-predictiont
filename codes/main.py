"""
This file executes various bayesian models and a changepoint detection in Technical debt dataset

"""

import os

from preprocessing import preprocessing
from tsDataPreparation import data_prepare
from bayesian_prediction_orbit_DLT import bayesian_orbit_DLT
from bayesian_prediction_orbit_ETS import bayesian_orbit_ets
from bayesian_pybats_dglm import bayesian_dglm
from bayesian_prediction_pyDLM import bayesian_pyDLM


from commons import PREPROCESSING, DGLM, ORBIT, PYDLM


def main():

    if PREPROCESSING:
        preprocessing()
        data_prepare()
    if DGLM:
        bayesian_dglm()
    if ORBIT:
        bayesian_orbit_DLT()
        bayesian_orbit_ets()
    if PYDLM:
        bayesian_pyDLM()



if __name__ == '__main__':
    main()