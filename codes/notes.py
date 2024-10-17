import os
import pandas as pd
import numpy as np
from commons import DATA_PATH
from modules import check_encoding
import matplotlib.pyplot as plt
import logging
from modules import MAPE, RMSE, MAE, MSE
from scipy.stats import multivariate_normal

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Bayesian Changepoint Detection Helper Functions
class BayesianChangepointDetection:
    def __init__(self, hazard_func, obs_likelihood):
        self.hazard_func = hazard_func
        self.obs_likelihood = obs_likelihood
        self.run_length_probs = []

    def initialize(self, data):
        self.data = data
        self.T = data.shape[0]
        self.run_length_probs = np.zeros(self.T)
        self.run_length_probs[0] = 1.0  # Initialize the first run length probability

    def detect(self):
        changepoints = []
        for t in range(1, self.T):
            # Compute predictive probability
            pred_prob = self.obs_likelihood(self.data[:t])
            
            # Update run length probabilities using recursive message passing
            self.run_length_probs[t] = self.hazard_func(t) * (1 - pred_prob)
            
            # Normalize probabilities
            self.run_length_probs[t] /= np.sum(self.run_length_probs[t])

            # If there's a high probability of changepoint, log it
            if self.run_length_probs[t] > 0.9:  # Example threshold
                changepoints.append(t)

        return changepoints  # Return detected changepoints

# Define a hazard function (geometric distribution as an example)
def hazard_function(t, lambda_=200):
    return 1.0 / lambda_

# Define a simple observation likelihood (e.g., multivariate normal) with regularization
def observation_likelihood(data):
    mean = np.mean(data, axis=0)

    # Compute the covariance matrix
    cov = np.cov(data, rowvar=False)

    # Check if the covariance matrix is square
    if cov.shape[0] != cov.shape[1]:
        logger.error(f"Covariance matrix shape mismatch: {cov.shape}")
        return 1e-10  # Return a small likelihood if covariance matrix is ill-defined

    # Add a small regularization to the diagonal of the covariance matrix to make it positive definite
    regularization_term = 1e-6
    cov += np.eye(cov.shape[0]) * regularization_term
    
    try:
        mvn = multivariate_normal(mean=mean, cov=cov, allow_singular=False)
        likelihoods = mvn.pdf(data)
        return np.prod(likelihoods)
    except np.linalg.LinAlgError as e:
        logger.error(f"Error in computing multivariate normal likelihood: {str(e)}")
        return 1e-10  # Return a small likelihood if there's an error in covarian


def trigger_detection(df_path, project_name, periodicity):
    try:
        # Load dataset
        encoding = check_encoding(df_path)
        df = pd.read_csv(df_path, encoding=encoding)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df.set_index('COMMIT_DATE', inplace=True)
        df = df.dropna()

        # Assuming independent variables are all columns except COMMIT_DATE and SQALE_INDEX
        independent_vars = df.drop(columns=["SQALE_INDEX"]).values

        # Ensure the data has sufficient dimensions
        if independent_vars.ndim != 2 or independent_vars.shape[0] < independent_vars.shape[1]:
            logger.error(f"Error in {project_name}: Insufficient data or too many features. Data shape: {independent_vars.shape}")
            return None

        # Initialize and apply Bayesian changepoint detection
        bocpd = BayesianChangepointDetection(hazard_function, observation_likelihood)
        bocpd.initialize(independent_vars)

        changepoints = bocpd.detect()
        logger.info(f"Detected changepoints for {project_name} in {periodicity} data at indices: {changepoints}")

        return changepoints  # Return the detected changepoints

    except Exception as e:
        logger.error(f"Error processing {project_name}: {str(e)}")
        return None


def bayesian_changepoint_detection():
    # Define paths for biweekly and monthly data
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data_1")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data_1")
    complete_data_path = os.path.join(DATA_PATH, "complete_data_1")

    # List existing data files
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)
    complete_files = os.listdir(complete_data_path)

    for i in range(len(biweekly_files)):
        if biweekly_files[i] == '.DS_Store':
            continue
        project = biweekly_files[i][:-4]

        logger.info(f"Processing {project}")

        # Process biweekly data
        logger.info(f"> Processing {project} for biweekly data")
        changepoints_biweekly = trigger_detection(
            df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
            project_name=project,
            periodicity="biweekly",
        )

    '''for i in range(len(monthly_files)):
        if monthly_files[i] == '.DS_Store':
            continue
        project = monthly_files[i][:-4]

        logger.info(f"Processing {project}")

        # Process monthly data
        logger.info(f"> Processing {project} for monthly data")
        changepoints_monthly = trigger_detection(
            df_path=os.path.join(monthly_data_path, monthly_files[i]),
            project_name=project,
            periodicity="monthly",
        )

    for i in range(len(complete_files)):
        if complete_files[i] == '.DS_Store':
            continue
        project = complete_files[i][:-4]

        logger.info(f"Processing {project}")

        # Process complete data
        logger.info(f"> Processing {project} for complete data")
        changepoints_complete = trigger_detection(
            df_path=os.path.join(complete_data_path, complete_files[i]),
            project_name=project,
            periodicity="complete",
        )'''
        

    logger.info("> Changepoint detection performed!")