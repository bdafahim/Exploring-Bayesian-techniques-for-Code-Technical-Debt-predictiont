import os
import pandas as pd
import numpy as np
from pybats.dglm import dlm
from pybats.plot import plot_data_forecast
from commons import DATA_PATH
from modules import check_encoding
import matplotlib.pyplot as plt
import logging
from pybats.analysis import analysis
from pybats.point_forecast import median
from pybats.dcmm import dcmm
from pybats.plot import plot_data_forecast

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def bayesian_dglm(seasonality=True):
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
            seasonality=seasonality
        )
        # Check if processing was successful
        if pybats_dglm_biweekly is None or pybats_dglm_monthly is None:
            logger.warning(f"Skipping {project} due to simulation issues.")
            continue
            logger.info(f"> dglm prediction for project <{project}> performed - {i+1}/{len(biweekly_files)}")

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
            seasonality=seasonality
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
            seasonality=False
        )
        

    logger.info("> Bayesian DGLM simulation stage performed!")


def trigger_prediction(df_path, project_name, periodicity, seasonality):
    try:
        # Load the dataset
        encoding = check_encoding(df_path)
        df = pd.read_csv(df_path, encoding=encoding)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df['SQALE_INDEX'] = pd.to_numeric(df['SQALE_INDEX'], errors='coerce')
        df = df.dropna()

        # Define the date index and data columns
        df.set_index('COMMIT_DATE', inplace=True)
        y = df['SQALE_INDEX']

        # Define seasonality
        if seasonality:
            if periodicity == 'biweekly':
                seasPeriods = 26
            elif periodicity == 'monthly':
                seasPeriods = 12
            else:
                seasPeriods = None
        else:
            seasPeriods = None

        # Define priors for DGLM
        prior_length = max(10, seasPeriods * 2 if seasPeriods else 0)
        k = min(len(y) - 1, prior_length)
        mod, prior = dcmm(y.iloc[:k].values, seasPeriods=seasPeriods)

        # Fit the model and forecast
        forecast_start = df.index[k]
        forecast_end = df.index[-1]
        result = analysis(y, mod, k, forecast_start, forecast_end,
                          forecast_samps=1000, prior_length=prior_length,
                          family="normal", seasPeriods=seasPeriods)

        # Save forecasts to DataFrame
        forecast = median(result)
        df_forecast = pd.DataFrame(data={
            'Forecast': forecast,
            'Actual': y[forecast_start:forecast_end+pd.DateOffset(days=1)]
        }, index=pd.date_range(start=forecast_start, periods=len(forecast), freq=df.index.freq))

        # Plotting forecast against actual data
        plot_data_forecast(y, forecast, df.index[k], forecast_start, forecast_end)

        return df_forecast

    except Exception as e:
        logger.error(f"Error processing {project_name}: {str(e)}")
        return None