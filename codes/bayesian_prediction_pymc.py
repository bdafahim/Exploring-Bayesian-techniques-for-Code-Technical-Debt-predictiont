import os
import pandas as pd
import numpy as np
import pymc as pm
from commons import DATA_PATH
from modules import check_encoding
import matplotlib.pyplot as plt
import logging
from modules import MAPE, RMSE, MAE, MSE

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



def add_seasonality(x, period):
    return pm.math.sin(2 * np.pi * x / period)

def trigger_prediction(df_path, project_name, periodicity):
    try:
        # Load dataset
        encoding = check_encoding(df_path)
        df = pd.read_csv(df_path, encoding=encoding)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df.set_index('COMMIT_DATE', inplace=True)
        df = df.dropna()

        # Check for missing values in each column
        missing_values = df.isnull().sum()
        print(f'Missing values in project {project_name}')
        print(missing_values[missing_values > 0])  # Print columns with missing values and their count

        # Splitting data into training (80%) and testing (20%)
        split_point = round(len(df) * 0.8)
        training_df = df.iloc[:split_point, :]
        testing_df = df.iloc[split_point:, :]

        # Dependent and independent variables
        y_train = training_df['SQALE_INDEX'].values
        x_train = training_df.drop(columns=['SQALE_INDEX']).values
        y_test = testing_df['SQALE_INDEX'].values
        x_test = testing_df.drop(columns=['SQALE_INDEX']).values

        # Normalize the data for better model convergence
        x_train_scaled = np.log1p(x_train)
        x_test_scaled = np.log1p(x_test)

        # Define the PyMC model
        with pm.Model() as model:
            # Define priors for coefficients (linear regression model)
            intercept = pm.Normal("Intercept", mu=0, sigma=10)
            coefficients = pm.Normal("Coefficients", mu=0, sigma=10, shape=x_train_scaled.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Adding seasonality for biweekly data (period = 26) or monthly data (period = 12)
            if periodicity == "biweekly":
                seasonality = add_seasonality(np.arange(len(x_train_scaled)), 26)
            elif periodicity == "monthly":
                seasonality = add_seasonality(np.arange(len(x_train_scaled)), 12)

            time_index = np.arange(len(x_train_scaled))
            trend = pm.Normal("trend", mu=0, sigma=1) * time_index

            if  periodicity == "complete":
                # Define the expected value (mu) of the dependent variable
                mu = intercept + pm.math.dot(x_train_scaled, coefficients)
            else:
                # Define the expected value (mu) of the dependent variable
                mu = intercept + pm.math.dot(x_train_scaled, coefficients) + seasonality + trend

            # Define likelihood for SQALE_INDEX
            likelihood = pm.Normal("SQALE_INDEX", mu=mu, sigma=sigma, observed=y_train)

            # Inference
            trace = pm.sample(1000, return_inferencedata=True)

            # Plot the posterior distributions
            #pm.plot_trace(trace)
            #plt.show()

            print(f'pm summary: {pm.summary(trace)}')
            print(f"Variables in trace: {trace.posterior.keys()}")

            # Posterior predictive check (forecast)
            posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["SQALE_INDEX"])

            # Forecasting for testing data
            mu_test = intercept + np.dot(x_test_scaled, coefficients)
            test_posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["SQALE_INDEX"])

            print(f"Keys in test_posterior_predictive: {test_posterior_predictive.keys()}")

            print(test_posterior_predictive.posterior_predictive)
            print('-----Keys----')
            print(test_posterior_predictive.posterior_predictive.keys())

            if "SQALE_INDEX" in test_posterior_predictive.posterior_predictive.keys():
                print(f"Shape of SQALE_INDEX: {test_posterior_predictive.posterior_predictive['SQALE_INDEX'].shape}")
            else:
                print("SQALE_INDEX not found in posterior_predictive")

            # Calculate error metrics
            predicted_y_test = test_posterior_predictive.posterior_predictive["SQALE_INDEX"].mean(axis=0)
            predicted_y_test_np = predicted_y_test.mean(axis=0).values
            y_test_len = len(y_test)
            predicted_y_test_np = predicted_y_test_np[-y_test_len:]

            mae_value = round(MAE(y_test, predicted_y_test_np), 2)
            mape_value = round(MAPE(y_test, predicted_y_test_np), 2)
            mse = round(MSE(y_test, predicted_y_test_np), 2)
            rmse_value = round(RMSE(y_test, predicted_y_test_np), 2)

            # Prepare result data
            result_data = {
                'Project': project_name,
                "Model": 'Bayesian Linear Regression',
                'Estimator': 'MCMC',
                "MAE": mae_value,
                "MAPE": mape_value,
                "MSE": mse,
                "RMSE": rmse_value
            }

            print(result_data)

            # Output path to save the results
            base_path = os.path.join(DATA_PATH, "PYMC", "PYMC_Result", periodicity)
            os.makedirs(base_path, exist_ok=True)
            csv_output_path = os.path.join(base_path, "assessment.csv")

            # Save result_df as a CSV file
            results_df = pd.DataFrame([result_data])
            if not os.path.isfile(csv_output_path):
                results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
            else:
                results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

            return result_data

    except Exception as e:
        logger.error(f"Error processing {project_name}: {str(e)}")
        return None


def bayesian_pymc():
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
        pybats_dglm_biweekly = trigger_prediction(
            df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
            project_name=project,
            periodicity="biweekly",
        )

    for i in range(len(monthly_files)):
        if monthly_files[i] == '.DS_Store':
            continue
        project = monthly_files[i][:-4]

        logger.info(f"Processing {project}")

        # Process monthly data
        logger.info(f"> Processing {project} for monthly data")
        pybats_dglm_monthly = trigger_prediction(
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
        pybats_dglm_complete = trigger_prediction(
            df_path=os.path.join(complete_data_path, complete_files[i]),
            project_name=project,
            periodicity="complete",
        )
        

    logger.info("> PYMC Forecast stage performed!")
