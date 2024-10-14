import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE
import json
from pydlm import dlm, trend, seasonality, dynamic
import logging
import matplotlib.pyplot as plt



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Main method to trigger the prediction process
def trigger_changepoint_detection(df_path, project_name, periodicity=None):
    try:
        # Read the data from CSV
        df = pd.read_csv(df_path)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df.set_index('COMMIT_DATE', inplace=True)
        df = df.dropna()

        #Dependent and independent variables
        y_train = df['SQALE_INDEX'].values
        x_train = df.drop(columns=['SQALE_INDEX']).values

        # Convert x_train to list format for pyDLM
        x_train = x_train.tolist()

        # Create the DLM model with a trend component
        dlm_model = dlm(y_train) + trend(degree=1, discount=0.98, name='linear_trend')

        # Add seasonality based on the periodicity
        if periodicity == 'biweekly':
            dlm_model = dlm_model + seasonality(period=26, discount=0.98, name='biweekly_seasonality')
        elif periodicity == 'monthly':
            dlm_model = dlm_model + seasonality(period=12, discount=0.98, name='monthly_seasonality')

        # Add the exogenous variables as a dynamic component
        dlm_model = dlm_model + dynamic(features=x_train, discount=0.98, name='exogenous_features')

        # Fit the model
        dlm_model.fit()

        # Get the smoothed result using backwardSmoother
        smoothed_result = dlm_model.getMean(filterType='backwardSmoother')

        # Detect change points by analyzing abrupt changes in the smoothed result
        change_points = []
        threshold = 1000  # Set a threshold for detecting an abrupt change
        for i in range(1, len(smoothed_result)):
            if abs(smoothed_result[i] - smoothed_result[i - 1]) > threshold:
                change_points.append(i)

        # Log and save the change points detected in the time series
        print(f"Detected change points for {project_name}: {change_points}")

        # Store the results in a JSON file
        base_path = os.path.join(DATA_PATH, 'Changepoint_Result', periodicity)
        os.makedirs(base_path, exist_ok=True)
        plot_path = os.path.join(base_path, 'Changepoinyt_Plots')
        os.makedirs(plot_path, exist_ok=True)


        # Plot the SQALE_INDEX values (y_train)
        plt.figure(figsize=(10, 6))
        plt.plot(y_train, label="SQALE_INDEX", color='blue', linewidth=2)

        # Highlight change points with red vertical lines
        for cp in change_points:
            plt.axvline(x=cp, color='red', linestyle='--', label=f'Change Point at {cp}' if cp == change_points[0] else "")

        plt.title(f"SQALE_INDEX with Detected Change Points, project: {project_name}, periodicty:{periodicity}")
        plt.xlabel("Time Index")
        plt.ylabel("SQALE_INDEX")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_path, f"{project_name}.png"))
        #plt.show()

        
        output_path = os.path.join(base_path, 'Changepoints_JSON', f"{project_name}_changepoints.json")
        os.makedirs(output_path, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump({"change_points": change_points}, f)

        # Optionally, return change points for further analysis
        return change_points

    except Exception as e:
        logger.error(f"Error processing {project_name}: {str(e)}")
        return None




# Example function call
def bayesian_change_point_detection_pyDLM():
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    complete_data_path = os.path.join(DATA_PATH, "complete_data")

    biweekly_files = os.listdir(biweekly_data_path) 
    monthly_files = os.listdir(monthly_data_path)
    complete_files = os.listdir(complete_data_path)

    # Process biweekly data
    for biweekly_file in biweekly_files:
        if biweekly_file == '.DS_Store':
            continue
        project = biweekly_file[:-4]
        print(f"> Processing {project} for biweekly data")
        method_biweekly = trigger_changepoint_detection(
            df_path=os.path.join(biweekly_data_path, biweekly_file),
            project_name=project,
            periodicity="biweekly",
        )

    # Process monthly data
    for monthly_file in monthly_files:
        if monthly_file == '.DS_Store':
            continue
        project = monthly_file[:-4]
        print(f"> Processing {project} for monthly data")
        method_monthly = trigger_changepoint_detection(
            df_path=os.path.join(monthly_data_path, monthly_file),
            project_name=project,
            periodicity="monthly",
        )

    # Process complete data
    for complete_file in complete_files:
        if complete_file == '.DS_Store':
            continue
        project = complete_file[:-4]
        print(f"> Processing {project} for complete data")
        method_complete = trigger_changepoint_detection(
            df_path=os.path.join(complete_data_path, complete_file),
            project_name=project,
            periodicity="complete",
        )