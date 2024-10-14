import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE
import json
import pybsts




# Main method to trigger the prediction process
def trigger_changepoint_detection(df_path, project_name, periodicity=None, seasonality=None):
    # Read the data from CSV
    df = pd.read_csv(df_path)

    # Convert the COMMIT_DATE column to datetime format
    df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
    df['SQALE_INDEX'] = pd.to_numeric(df['SQALE_INDEX'], errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    # Extract the target time series data (SQALE_INDEX)
    y = df['SQALE_INDEX'].values

    # Initialize the pyBSTS model with desired components
    model = BSTS(
        response=y,
        niter=1000,  # Number of MCMC iterations
        components=["trend"],  # Add more components if necessary, e.g., 'seasonality' for periodic data
    )

    # If seasonality is provided, add it to the model
    if seasonality:
        model.add_seasonality(seasonality, name="seasonality", period=seasonality)

    # Fit the model to the data
    model.fit()

    # Detect change points
    change_points = model.detect_changepoints()

    # Log and save the change points detected in the time series
    print(f"Detected change points for {project_name}: {change_points}")

    # Store the results in a JSON file
    base_path = os.path.join(DATA_PATH, 'Changepoint_Result', periodicity)
    os.makedirs(base_path, exist_ok=True)
    output_path = os.path.join(base_path, "changepoints", f"{project_name}_changepoints.json")

    with open(output_path, "w") as f:
        json.dump({"change_points": change_points.tolist()}, f)

    # Optionally, return change points for further analysis
    return change_points



# Example function call
def bayesian_change_point_detection():
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data_1")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data_1")
    complete_data_path = os.path.join(DATA_PATH, "complete_data_1")

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
            seasonality=26
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
            seasonality=12
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
            seasonality=None
        )