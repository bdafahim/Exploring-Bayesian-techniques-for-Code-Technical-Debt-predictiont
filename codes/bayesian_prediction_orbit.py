import os
import pandas as pd
import numpy as np
from orbit.models import DLT, ETS, LGT, KTR
from orbit.diagnostics.metrics import smape
from orbit.diagnostics.plot import plot_predicted_data
from orbit.forecaster import Forecaster
from commons import DATA_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from modules import MAPE, RMSE, MAE, MSE, check_encoding, detect_existing_output


# Define paths for biweekly and monthly data
def bayesian_orbit(seasonality=True):
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data_1")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data_1")
    complete_data_path = os.path.join(DATA_PATH, "complete_data_1")

    biweekly_files = os.listdir(biweekly_data_path) 
    monthly_files = os.listdir(monthly_data_path)
    complete_files = os.listdir(complete_data_path)

    for i in range(len(biweekly_files)):
        if biweekly_files[i] == '.DS_Store':
            continue
        project = biweekly_files[i][:-4]

         # Process biweekly data
        print(f"> Processing {project} for biweekly data")
        method_biweekly = trigger_prediction(
            df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
            project_name=project,
            periodicity="biweekly",
            seasonality=26
        )

        # Process monthly data
        print(f"> Processing {project} for monthly data")
        method_monthly = trigger_prediction(
            df_path=os.path.join(monthly_data_path, monthly_files[i]),
            project_name=project,
            periodicity="monthly",
            seasonality=12
        )

        # Process monthly data
        print(f"> Processing {project} for complete data")
        method_monthly = trigger_prediction(
            df_path=os.path.join(complete_data_path, complete_files[i]),
            project_name=project,
            periodicity="complete",
            seasonality=None
        )

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

    x_train_scaled = x_train.map(np.log1p)

    ESTIMATOR_MAP = 'stan-map'
    ESTIMATOR_MCMC = 'stan-mcmc'
    ESTIMATOR_SVI = 'pyro-svi'


    # Apply different models and compare their performance
    models = {
        'DLT': DLT(seasonality=seasonality, response_col='SQALE_INDEX', date_col='COMMIT_DATE', estimator=ESTIMATOR_MAP, seed=8888, 
        global_trend_option='linear',prediction_percentiles=[5, 95], regressor_col=x_train_scaled.columns.tolist(), n_bootstrap_draws=1000,),
        'ETS': ETS(seasonality=seasonality, response_col='SQALE_INDEX', date_col='COMMIT_DATE', estimator='stan-map', seed=1, prediction_percentiles=[5, 95],),
        #'LGT': LGT(seasonality=seasonality, response_col='SQALE_INDEX', date_col='COMMIT_DATE', estimator='stan-map', seed=8888),
    }

    all_results = []

    # Create the output folder
    base_path = os.path.join(DATA_PATH, 'ORBIT_ML')

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        if(model_name == 'ETS'):
            training_df = training_df[['COMMIT_DATE', 'SQALE_INDEX']].copy()
        model.fit(df=training_df)
        predicted_df = model.predict(df=testing_df, predict_interval=True)
        p = predicted_df.copy()
        #valid = testing_df[['COMMIT_DATE', 'SQALE_INDEX']].copy()
        p = p.merge(testing_df, on='COMMIT_DATE', how='left')

        # Actual and predicted values
        actual = y_test
        predicted = predicted_df['prediction'].values

        # Calculate error metrics
        mae = round(MAE(predicted, actual),2)
        mape_value = round(MAPE(predicted, actual),2)
        mse = round(MSE(predicted, actual),2)
        rmse = round(RMSE(predicted, actual),2)

        print(f"{model_name} MAE: {mae:.2f}")
        print(f"{model_name} MAPE: {mape_value:.2f}%")
        print(f"{model_name} RMSE: {rmse:.2f}")
        print(f"{model_name} mse: {mse:.2f}%")

        # Store results for the current model
        result_data = {
            'Project': project_name,
            'Model': model_name,
            'MAE': mae,
            'MAPE': mape_value,
            'RMSE': rmse,
            'MSE': mse
        }

        # Append the result to the global list for cumulative results
        all_results.append(result_data)

        output_folder = os.path.join(DATA_PATH, base_path, f"{model_name}_Result", periodicity)
        os.makedirs(output_folder, exist_ok=True)

        # Create a DataFrame to display all results
        results_df = pd.DataFrame([result_data])
        print(f'Results for {periodicity} data')
        print(results_df)

        # Save result_df as a CSV file
        csv_output_path = os.path.join(output_folder, "assessment.csv")

        # Append to CSV instead of overwriting
        if not os.path.isfile(csv_output_path):
            # If the file doesn't exist, write with header
            results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
        else:
            # If the file exists, append without writing the header
            results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

        print(f"Results for {model_name} saved in {output_folder}")


        '''fig, ax = plt.subplots(1,1, figsize=(1280/96, 720/96))
        ax.plot(p['COMMIT_DATE'], p['SQALE_INDEX'], label='actual')
        ax.plot(p['COMMIT_DATE'], p['prediction'], label='prediction')
        if 'prediction_5' in p.columns and 'prediction_95' in p.columns:
            ax.fill_between(p['COMMIT_DATE'], p['prediction_5'], p['prediction_95'], alpha=0.2, color='orange', label='prediction percentiles')
        else:
            print(f"Percentile columns are not available for {model_name}. Skipping fill_between.")
        ax.set_title(f'Error, Trend, Seasonality {model_name} for {periodicity} data')
        ax.set_ylabel('Sqale Index')
        ax.set_xlabel('Date')
        ax.legend()
        plt.show()

      
        # Plot forecast vs actuals
        plot_predicted_data(
            training_actual_df=training_df,
            predicted_df=predicted_df,
            date_col='COMMIT_DATE',
            actual_col='SQALE_INDEX',
            title=f'Prediction with {model_name}'
        )'''

    if all_results:
        final_results_df = pd.DataFrame(all_results)
        print("Final cumulative results:")
        print(final_results_df)
        output_folder = os.path.join(DATA_PATH, base_path, f"All model Result", periodicity)
        os.makedirs(output_folder, exist_ok=True)
        

        # Save result_df as a CSV file
        csv_output_path = os.path.join(output_folder, "assessment.csv")
        if not os.path.isfile(csv_output_path):
            final_results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
        else:
            final_results_df.to_csv(csv_output_path, mode='a', index=False, header=False)
    else:
        print("No results were generated.")
    
    return all_results, result_data