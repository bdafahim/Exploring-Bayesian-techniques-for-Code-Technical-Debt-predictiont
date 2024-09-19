import os
import pandas as pd
import numpy as np
from pybats.dglm import dlm
from pybats.analysis import analysis
from pybats.plot import plot_data_forecast
from commons import DATA_PATH
from modules import check_encoding
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def bayesian_dglm(seasonality=True):
    # Define paths for biweekly and monthly data
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data_1")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data_1")

    # List existing data files
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)

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
            seasonality=seasonality
        )

        # Process monthly data
        logger.info(f"> Processing {project} for monthly data")
        pybats_dglm_monthly = trigger_prediction(
            df_path=os.path.join(monthly_data_path, monthly_files[i]),
            project_name=project,
            periodicity="monthly",
            seasonality=seasonality
        )

        # Check if processing was successful
        if pybats_dglm_biweekly is None or pybats_dglm_monthly is None:
            logger.warning(f"Skipping {project} due to simulation issues.")
            continue

        logger.info(f"> dglm prediction for project <{project}> performed - {i+1}/{len(biweekly_files)}")

    logger.info("> Bayesian DGLM simulation stage performed!")


def trigger_prediction(df_path, project_name, periodicity, seasonality):
    try:
        # Check encoding of the file
        encoding = check_encoding(df_path)
        
        # Load the dataset with the detected encoding
        df = pd.read_csv(df_path, encoding=encoding)
        
        logger.debug(f"Data loaded for {project_name}. Shape: {df.shape}")
        
        # Check if DataFrame is empty
        if df.empty:
            logger.error(f"Empty DataFrame for project {project_name}")
            return None

        # Check if required columns exist
        required_columns = ['COMMIT_DATE', 'SQALE_INDEX']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns for project {project_name}")
            return None

        df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)
        
        # Ensure SQALE_INDEX is numeric
        df['SQALE_INDEX'] = pd.to_numeric(df['SQALE_INDEX'], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        logger.debug(f"Data cleaned for {project_name}. New shape: {df.shape}")
        
        # Check if there's enough data after cleaning
        if len(df) < 10:  # arbitrary minimum number of rows
            logger.error(f"Not enough data for project {project_name} after cleaning")
            return None

        # Dependent variable
        sqale_index = df.SQALE_INDEX.to_numpy()
        
        # Initial data splitting: 80% for training and 20% for testing
        split_point = round(len(sqale_index) * 0.8)
        training_df = df.iloc[:split_point, :]
        testing_df = df.iloc[split_point:, :]

        logger.debug(f"Training data shape: {training_df.shape}, Testing data shape: {testing_df.shape}")

        # Prepare the training and testing data
        y_train = training_df['SQALE_INDEX'].values
        X_train = training_df.drop(columns=['COMMIT_DATE', 'SQALE_INDEX']).values
        y_test = testing_df['SQALE_INDEX'].values
        X_test = testing_df.drop(columns=['COMMIT_DATE', 'SQALE_INDEX']).values

        logger.debug(f"y_train shape: {y_train.shape}, X_train shape: {X_train.shape}")
        logger.debug(f"y_test shape: {y_test.shape}, X_test shape: {X_test.shape}")

        # Define correct forecast start and end points
        forecast_start = len(y_train)  # Forecast starts right after the training data ends
        forecast_end = forecast_start + len(y_test) - 1  # Forecasting ends at the last test data point

        # Define seasPeriods and seasHarmComponents
        if seasonality:
            if periodicity == "biweekly":
                seasPeriods = [26]
                seasHarmComponents = [[1]]  # List of lists
            elif periodicity == "monthly":
                seasPeriods = [12]
                seasHarmComponents = [[1]]  # List of lists
            else:
                seasPeriods = []
                seasHarmComponents = []
        else:
            seasPeriods = []
            seasHarmComponents = []

        logger.debug(f"seasPeriods: {seasPeriods}, seasHarmComponents: {seasHarmComponents}")

        # Perform the Bayesian analysis and forecast
        try:
            result = analysis(
                y_train, 
                X=X_train,
                k=len(y_test),
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                nsamps=500,
                family='normal',
                ntrend=1,
                dates=df['COMMIT_DATE'],
                seasPeriods=seasPeriods,
                seasHarmComponents=seasHarmComponents,
                ret=['model', 'forecast'],
                mean_only=False
            )
        except Exception as analysis_error:
            logger.error(f"Error in analysis function for {project_name}: {str(analysis_error)}")
            raise

        # Extract forecast results using integer indexing
        model = result[0]
        forecast_results = result[1]

        # Correct method call to obtain forecast results
        forecast_mean = forecast_results.mean() if callable(forecast_results.mean) else np.array(forecast_results.mean)
        forecast_var = forecast_results.var() if callable(forecast_results.var) else np.array(forecast_results.var)

        # Ensure forecast_mean and forecast_var are 1D arrays and match the length of y_test
        forecast_mean = forecast_mean.flatten()
        forecast_var = forecast_var.flatten()

        if forecast_mean.shape[0] != len(y_test):
            logger.error(f"Forecast length mismatch: expected {len(y_test)}, got {forecast_mean.shape[0]}")
            return None

        # Plot the forecast results manually using matplotlib
        fig, ax = plt.subplots()

        # Plot actual training data
        ax.plot(training_df['COMMIT_DATE'], y_train, label='Training Data', color='blue')

        # Plot actual test data
        ax.plot(testing_df['COMMIT_DATE'], y_test, label='Test Data', color='orange')

        # Plot forecasted mean and confidence intervals
        forecast_dates = testing_df['COMMIT_DATE'].values
        ax.plot(forecast_dates, forecast_mean, label='Forecasted Mean', color='green')
        ax.fill_between(forecast_dates,
                        forecast_mean - 1.96 * np.sqrt(forecast_var),
                        forecast_mean + 1.96 * np.sqrt(forecast_var),
                        color='green', alpha=0.2, label='95% Confidence Interval')

        # Customize the plot
        ax.set_title(f"Forecasting for {project_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("SQALE_INDEX")
        ax.legend()

        # Show the plot
        plt.show()
         
        return True  # Return a success flag

    except Exception as e:
        logger.exception(f"Error in processing {project_name}: {str(e)}")
        return None  # Indicate failure