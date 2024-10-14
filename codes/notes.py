# Load the best model configuration and regressors from the appropriate file
    if(seasonality):
         model_config_path = os.path.join(DATA_PATH, "best_sarimax_models", periodicity, f"{project_name}.json")
    else:
        model_config_path = os.path.join(DATA_PATH, "best_arimax_models", periodicity, f"{project_name}.json")

    if not os.path.exists(model_config_path):
        print(f"Model configuration file not found for project {project_name} and periodicity {periodicity}.")
        return None

    with open(model_config_path, 'r') as file:
        model_data = json.load(file)
        best_model_cfg = model_data.get('model_params')
        best_aic = model_data.get('best_aic')
        best_regressors = model_data.get('best_regressors')
        output_flag = True

    if not best_model_cfg or not best_regressors:
        print(f"Model configuration or regressors missing for project {project_name}.")
        output_flag = False
        return None

  '''residuals = actual - predicted
        plt.figure(figsize=(12, 6))
        plt.plot(testing_df['COMMIT_DATE'], residuals, label='Residuals', marker='o')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Commit Date')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} Residuals Over Time')
        plt.legend()
        plt.show()'''
import os
import pandas as pd

# Function to save results to CSV and avoid duplication
def save_results_to_csv(result_data, csv_output_path, project_name, model_name):
    # Create a DataFrame for the new result
    results_df = pd.DataFrame([result_data])

    # Check if the CSV file exists
    if os.path.isfile(csv_output_path):
        # Load the existing CSV
        existing_results = pd.read_csv(csv_output_path)

        # Check if this project and model combination already exists
        existing_rows = existing_results[(existing_results['Project'] == project_name) & 
                                         (existing_results['Model'] == model_name)]

        if not existing_rows.empty:
            # If the project and model combination exists, update that row
            print(f"Updating {project_name} with {model_name} as it already exists in the CSV.")
            existing_results.update(results_df)
        else:
            # If it doesn't exist, append the new result to the DataFrame
            print(f"Adding new entry for {project_name} with {model_name}.")
            existing_results = pd.concat([existing_results, results_df], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        existing_results.to_csv(csv_output_path, mode='w', index=False, header=True)
    else:
        # If the CSV doesn't exist, create it and write the new result
        results_df.to_csv(csv_output_path, mode='w', index=False, header=True)

    print(f"Results for {project_name} saved in {csv_output_path}")

# Example usage in your `trigger_prediction` function
def trigger_prediction(df_path, project_name, periodicity=None, seasonality=None):
    df = pd.read_csv(df_path)

    # Convert dates
    df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
    df['SQALE_INDEX'] = pd.to_numeric(df['SQALE_INDEX'], errors='coerce')
    df = df.dropna()

    # Splitting data into training (80%) and testing (20%)
    split_point = round(len(df) * 0.8)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]

    # Dependent and independent variables
    y_train = training_df['SQALE_INDEX'].values
    x_train = training_df.drop(columns=['COMMIT_DATE', 'SQALE_INDEX'])
    y_test = testing_df['SQALE_INDEX'].values
    x_test = testing_df.drop(columns=['COMMIT_DATE', 'SQALE_INDEX'])

    # Hypertune the DLT model
    best_model, best_config, important_features = hypertune_dlt_model(
        training_df=training_df, 
        y_train=y_train, 
        x_train=x_train, 
        y_test=y_test, 
        testing_df=testing_df, 
        seasonality=seasonality
    )

     # Ensure COMMIT_DATE is included in the DataFrame used for prediction along with important features
    test_df_for_prediction = testing_df[['COMMIT_DATE']].copy()  # Ensure COMMIT_DATE is present
    test_df_for_prediction = pd.concat([test_df_for_prediction, testing_df[important_features]], axis=1)

    # Use the best model for final predictions
    predicted_df = best_model.predict(df=test_df_for_prediction)
    
    actual = y_test
    predicted = predicted_df['prediction'].values
    
    # Calculate error metrics
    mae = round(MAE(predicted, actual), 2)
    mape_value = round(MAPE(predicted, actual), 2)
    mse = round(MSE(predicted, actual), 2)
    rmse = round(RMSE(predicted, actual), 2)

    # Log the metrics
    print(f"Final MAE: {mae:.2f}")
    print(f"Final MAPE: {mape_value:.2f}%")
    print(f"Final RMSE: {rmse:.2f}")
    print(f"Final MSE: {mse:.2f}")

    # Store the results in a dictionary
    result_data = {
        'Project': project_name,
        'Model': 'DLT',
        'Trend': best_config['trend'],
        'Estimator': best_config['estimator'],
        'MAE': mae,
        'MAPE': mape_value,
        'RMSE': rmse,
        'MSE': mse
    }

    # Output path to save the results
    base_path = os.path.join(DATA_PATH, 'ORBIT_ML', 'DLT_Result', periodicity)
    os.makedirs(base_path, exist_ok=True)
    csv_output_path = os.path.join(base_path, "assessment.csv")

    # Save result_df as a CSV file, updating if duplicate exists
    save_results_to_csv(result_data, csv_output_path, project_name, 'DLT')

    return result_data

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

    # Define the specification and configuration for the BSTS model
    specification = {
        "ar_order": 1,
        "local_trend": {"local_level": True},
        "sigma_prior": np.std(df['SQALE_INDEX'], ddof=1),
        "initial_value": df['SQALE_INDEX'].iloc[0]
    }

    config = {
        "ping": 10,
        "niter": 200,
        "seed": 1,
        "burn": 10
    }


    # Model fitting
    model = pybsts.PyBsts("gaussian", specification, config)
    model.fit(df['SQALE_INDEX'].values, seed=1)

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