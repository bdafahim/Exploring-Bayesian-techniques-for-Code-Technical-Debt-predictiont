import os
import pandas as pd
import numpy as np
from commons import DATA_PATH
import matplotlib.pyplot as plt
import pybsts
from modules import MAPE, RMSE, MAE, MSE, check_encoding

def bayesian_pybsts(seasonality=True):
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


        pybsts_biweekly = trigger_prediction(
            df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
            project_name=project,
            periodicity="biweekly",
            seasonality=seasonality
        )

    for i in range(len(monthly_files)):
        if monthly_files[i] == '.DS_Store':
            continue
        project = monthly_files[i][:-4]


        # Process monthly data
        print(f"> Processing {project} for monthly data")
        pybsts_monthly = trigger_prediction(
            df_path=os.path.join(monthly_data_path, monthly_files[i]),
            project_name=project,
            periodicity="monthly",
            seasonality=seasonality
        )

    for i in range(len(complete_files)):
        if complete_files[i] == '.DS_Store':
            continue
        project = complete_files[i][:-4]


        # Process complete data
        print(f"> Processing {project} for complete data")
        pybsts_complete = trigger_prediction(
            df_path=os.path.join(complete_data_path, complete_files[i]),
            project_name=project,
            periodicity="complete",
            seasonality=False
        )
        

    print("> Bayesian pybsts simulation stage performed!")


def plot_results(actual_data, forecast_means, forecast_index, project_name):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data.index, actual_data['SQALE_INDEX'], label='Observed', color='blue')
    plt.plot(forecast_index, forecast_means, label='Forecast', color='red')
    plt.fill_between(forecast_index,
                     [mean - 1.96 * np.std(forecast_means) for mean in forecast_means],
                     [mean + 1.96 * np.std(forecast_means) for mean in forecast_means],
                     color='pink', alpha=0.3)
    plt.title(f'Forecast vs Actual for {project_name}')
    plt.xlabel('Date')
    plt.ylabel('SQALE Index')
    plt.legend()
    plt.grid(True)
    plt.show()

def trigger_prediction(df_path, project_name, periodicity, seasonality):
    try:
        # Load the dataset
        encoding = check_encoding(df_path)
        df = pd.read_csv(df_path, encoding=encoding)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df.set_index('COMMIT_DATE', inplace=True)
        df['SQALE_INDEX'] = pd.to_numeric(df['SQALE_INDEX'], errors='coerce').astype(np.float64)
        df = df.dropna()

        # Splitting data into training (80%) and testing (20%)
        split_point = round(len(df) * 0.8)
        training_df = df.iloc[:split_point, :]
        testing_df = df.iloc[split_point:, :]

        # Define the specification and configuration for the BSTS model
        specification = {
            "ar_order": 1,
            "local_trend": {"local_level": True},
            "sigma_prior": np.std(training_df['SQALE_INDEX'], ddof=1),
            "initial_value": training_df['SQALE_INDEX'].iloc[0]
        }

        config = {
            "ping": 10,
            "niter": 200,
            "seed": 1,
            "burn": 10
        }

        y_test = testing_df['SQALE_INDEX'].values

        # Model fitting
        bsts_model = pybsts.PyBsts("gaussian", specification, config)
        bsts_model.fit(training_df['SQALE_INDEX'].values, seed=1)

        # Forecasting
        '''forecast_result = bsts_model.predict(len(testing_df), seed=1)
        forecast_means = np.array([point[0] for point in forecast_result])  # Ensure it's a flat array
        forecast_index = testing_df.index  # Ensure the index matches the forecast data'''

        # Forecasting: generate predictions for each test point and compute the average across all iterations
        forecast_means = []
        for _ in range(len(testing_df)):
            prediction = bsts_model.predict(1, seed=1)
            # Average over all iterations for this forecast step, ignoring burn-in
            mean_prediction = np.mean(prediction, axis=0)
            forecast_means.append(mean_prediction)

        # Flatten the list to create a 1D array of means
        forecast_means = np.array(forecast_means).flatten()

        # Calculate error metrics
        mae = MAE(y_test, forecast_means)
        mse = MSE(y_test, forecast_means)
        rmse = RMSE(y_test, forecast_means)
        mape = MAPE(y_test, forecast_means)

        # Log and store the results
        results_data = {
            'Project': project_name,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse
        }
        print('Result data >----- ', results_data)
        
        # Output path to save the results
        base_path = os.path.join(DATA_PATH, 'PYBSTS_Result', periodicity)
        os.makedirs(base_path, exist_ok=True)
        csv_output_path = os.path.join(base_path, "assessment.csv")

        # Save result_df as a CSV file
        results_df = pd.DataFrame([results_data])
        if not os.path.isfile(csv_output_path):
            results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
        else:
            results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

        print(f"Results for {project_name} saved in {base_path}")

        # Plot the results using the new function
        #plot_results(df, forecast_means, forecast_index, project_name)

        return bsts_model

    except Exception as e:
        print(f"Failed to process {project_name}: {str(e)}")
        return None