import os
import pandas as pd
import numpy as np
from orbit.models import KTR
from sklearn.metrics import mean_absolute_error
from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE




# Define the hypertuning function for DLT model
def hypertune_ktr_model(training_df, y_train, x_train, y_test, testing_df, seasonality, project_name):

    estimators = ['stan-map', 'stan-mcmc']

    best_mae = float('inf')
    best_model = None
    best_config = None

    # Hardcoded the estimator as pyro-svi
    estimator = 'pyro-svi'
    
    print(f"Training with estimator={estimator} for project={project_name}")
    
    # Define the model with the selected hyperparameters
    model = KTR(
        seasonality=seasonality,
        response_col='SQALE_INDEX',
        date_col='COMMIT_DATE',
        estimator=estimator,
        seed=8888,
        n_bootstrap_draws=1000,
        # pyro training config
        num_steps=301,
        message=100,
    )
    
    # Fit the model
    model.fit(df=training_df)
    
    # Predict and calculate MAE
    predicted_df = model.predict(df=testing_df)
    predicted = predicted_df['prediction'].values
    
    mae = mean_absolute_error(y_test, predicted)
    
    print(f"MAE for project={project_name}, estimator={estimator}: {mae:.2f}")
    
    # Check if the current model is better
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_config = {
            'estimator': estimator
        }

    print(f"Best configuration: {best_config} with MAE: {best_mae:.2f}")
    
    # Return the best model and configuration
    return best_model, best_config



# Main method to trigger the prediction process
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
    estimator = 'pyro-svi'

    # Define the model with the selected hyperparameters
    model = KTR(
        seasonality=seasonality,
        response_col='SQALE_INDEX',
        date_col='COMMIT_DATE',
        estimator=estimator,
        seed=8888,
        n_bootstrap_draws=1000,
        # pyro training config
        num_steps=301,
        message=100,
    )
    # Fit the model
    model.fit(df=training_df)
    # Use the best model for final predictions
    predicted_df = model.predict(df=testing_df)
    
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
        'Model': 'ETS',
        'Estimator': best_config['estimator'],
        'MAE': mae,
        'MAPE': mape_value,
        'RMSE': rmse,
        'MSE': mse
    }

    # Output path to save the results
    base_path = os.path.join(DATA_PATH, 'ORBIT_ML', 'ETS_Result', periodicity)
    os.makedirs(base_path, exist_ok=True)
    csv_output_path = os.path.join(base_path, "assessment.csv")

    # Save result_df as a CSV file
    results_df = pd.DataFrame([result_data])
    if not os.path.isfile(csv_output_path):
        results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
    else:
        results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

    print(f"Results for {project_name} saved in {base_path}")
    return result_data


# function call
def bayesian_orbit_ktr():
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
        method_biweekly = trigger_prediction(
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
        method_monthly = trigger_prediction(
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
        method_complete = trigger_prediction(
            df_path=os.path.join(complete_data_path, complete_file),
            project_name=project,
            periodicity="complete",
            seasonality=None
        )