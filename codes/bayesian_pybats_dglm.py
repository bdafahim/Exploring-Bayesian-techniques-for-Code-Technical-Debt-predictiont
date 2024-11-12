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
from modules import MAPE, RMSE, MAE, MSE


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



def bayes_forecast(iv, dv, periodicity, project_name, y_test):
    if iv is None:
        x = None
    else:
        x = iv.values

    y = dv.values
    k = 1
    forecast_end = len(y) - 1
    # Define the index where the forecast should start - beginning of the last 20% of the data
    forecast_start = int(len(y) * 0.8)

    # Set seasonal periods and components based on periodicity
    if periodicity == "biweekly":
        seas_periods = [26]
        seas_harm_components = [[1, 2, 3]]
    elif periodicity == "monthly":
        seas_periods = [12]
        seas_harm_components = [[1, 2]]
    else:  # No seasonality for 'complete' datasets
        seas_periods = []
        seas_harm_components = []


    try:

        # Define the model with PyBATS
        mod, samples = analysis(Y=y, X=x, family='normal',  # Change family as per your data distribution
                                forecast_start=forecast_start,
                                forecast_end=forecast_end,
                                k=k,
                                ntrend=1,
                                nsamps=5000,
                                seasPeriods=seas_periods,  # Adjust based on seasonality
                                seasHarmComponents=seas_harm_components,
                                prior_length=4,
                                deltrend=0.94,
                                delregn=0.90,
                                delVar=0.98,
                                delSeas=0.98,
                                rho=0.6)

        forecast = median(samples)
        credible_interval = 95
        alpha = (100 - credible_interval) / 2
        upper = np.round(np.percentile(samples, 100 - alpha, axis=0).reshape(-1), 2)
        lower = np.round(np.percentile(samples, alpha, axis=0).reshape(-1), 2)


        # Define index for the last 20% of the data
        test_start = int(len(y) * 0.8)
        # Get the last 20% of the data
        y_test = y[test_start:]

        # Error metrics are calculated for the forecast period
        mae = round(MAE(y[forecast_start:], forecast), 2)
        mape_value = round(MAPE(y[forecast_start:], forecast), 2)
        mse = round(MSE(y[forecast_start:], forecast), 2)
        rmse = round(RMSE(y[forecast_start:], forecast), 2)

        # Plot and save the forecast
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax = plot_data_forecast(fig, ax, y[forecast_start:forecast_end + k], forecast, samples,
                                dates=np.arange(forecast_start, forecast_end + 1, dtype='int'))
        ax.set_xlabel('Time')
        ax.set_ylabel('SQALE Index')
        ax.legend(['Forecast', 'Actual', '95% Credible Interval'])
        ax.set_title(f'Forecast for {project_name} ({periodicity})')

        # Define the path to save the plot
        plot_path = os.path.join(DATA_PATH, 'PYBATS_DGLM_Results', periodicity, 'Plots')
        os.makedirs(plot_path, exist_ok=True)
        plot_output_path = os.path.join(plot_path,f"{project_name}_forecast.pdf")
        plt.savefig(plot_output_path)
        plt.close(fig)  # Close the figure to save memory

        print(f"Forecast plot saved at {plot_path}")
    except Exception as e:
        logger.error(f"Error during forecasting for {project_name}: {str(e)}")

    # Log the metrics 
    print(f"Final MAE: {mae:.2f}")
    print(f"Final MAPE: {mape_value:.2f}%")
    print(f"Final RMSE: {rmse:.2f}")
    print(f"Final MSE: {mse:.2f}")

    # Store the results in a dictionary
    result_data = {
        'Project': project_name,
        'Model': 'DGLM',
        'MAE': mae,
        'MAPE': mape_value,
        'RMSE': rmse,
        'MSE': mse
    }

    # Output path to save the results
    base_path = os.path.join(DATA_PATH, 'PYBATS_DGLM_Results', periodicity, 'Results')
    os.makedirs(base_path, exist_ok=True)
    csv_output_path = os.path.join(base_path, "assessment.csv")

    # Save result_df as a CSV file
    results_df = pd.DataFrame([result_data])
    if not os.path.isfile(csv_output_path):
        results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
    else:
        results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

    print(f"Results for {project_name} saved in {base_path}")

    # Save the upper and lower confidence intervals
    interval_path = os.path.join(DATA_PATH, 'PYBATS_DGLM_Results', periodicity, 'Confidence Intervals')
    os.makedirs(interval_path, exist_ok=True)
    interval_output_path = os.path.join(interval_path, f"{project_name}_confidence_interval.csv")
        
    # Create a DataFrame for the confidence intervals
    interval_df = pd.DataFrame({
        'Index': np.arange(forecast_start, forecast_end + 1),
        'Lower': lower,
        'Upper': upper,
        'Predicted': np.round(np.ravel(forecast), 2),
        'Actual': np.round(np.ravel(y_test), 2)
    })

    # Save intervals to CSV
    interval_df.to_csv(interval_output_path, index=False)
    print(f"Confidence intervals saved at {interval_output_path}")

    return mod, forecast, samples, y


def trigger_prediction(df_path, project_name, periodicity):
    try:
        encoding = check_encoding(df_path)
        df = pd.read_csv(df_path, encoding=encoding)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df.set_index('COMMIT_DATE', inplace=True)
        df = df.dropna()

        # Check for missing values in each column
        missing_values = df.isnull().sum()
        print(f'Missing values in project{project_name}')
        print(missing_values[missing_values > 0])  # This will print columns with missing values and their count

        # Splitting data into training (80%) and testing (20%)
        split_point = round(len(df) * 0.8)
        training_df = df.iloc[:split_point, :]
        testing_df = df.iloc[split_point:, :]

        #Dependent and independent variables
        y_train = training_df['SQALE_INDEX'].values
        x_train = training_df.drop(columns=['SQALE_INDEX'])
        y_test = testing_df['SQALE_INDEX'].values
        x_test = testing_df.drop(columns=['SQALE_INDEX'])

        mv_mod, mv_for, mv_samp, mv_y = bayes_forecast(None, df['SQALE_INDEX'], periodicity, project_name, y_test)


        return mv_for

    except Exception as e:
        logger.error(f"Error processing {project_name}: {str(e)}")
        return None

def bayesian_dglm():
    # Define paths for biweekly and monthly data
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    complete_data_path = os.path.join(DATA_PATH, "complete_data")

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
        

    logger.info("> Bayesian DGLM simulation stage performed!")
