import os
import pandas as pd
import numpy as np
from orbit.models import DLT
from orbit.models import ETS
from sklearn.metrics import mean_absolute_error
from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE

from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments


# Function to save results for each estimator
def save_results(result_data, training_df, predicted_df,lower_bounds,upper_bounds, periodicity, project_name, estimator):
    base_path = os.path.join(DATA_PATH,'ORBIT_ML_ETS_Result', periodicity, 'Results')
    os.makedirs(base_path, exist_ok=True)
    csv_output_path = os.path.join(base_path, f"{estimator}_assessment.csv")

    results_df = pd.DataFrame([result_data])
    if not os.path.isfile(csv_output_path):
        results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
    else:
        results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

    print(f"Results for {project_name} with estimator {estimator} saved in {base_path}")

    # Define path for saving plot
    plot_path = os.path.join(DATA_PATH, 'ORBIT_ML_ETS_Result', periodicity, 'Plots')
    os.makedirs(plot_path, exist_ok=True)

    response_col='SQALE_INDEX'
    date_col='COMMIT_DATE'
    
    # Plot the results and save the plot
    ax = plot_predicted_data(training_df, predicted_df, date_col, response_col, title=f'ETS plot for {project_name}_{estimator}')
    fig = ax.get_figure()
    plt.close(fig)
    
    # Save the plot image
    plot_file_name = f"{project_name}_{estimator}_plot.png"
    plot_output_path = os.path.join(plot_path, plot_file_name)
    fig.savefig(plot_output_path)
    print(f"Plot saved at {plot_output_path}")

    # Save the confidence intervals to CSV
    interval_path = os.path.join(DATA_PATH, 'ORBIT_ML_ETS_Result', periodicity, 'Confidence Intervals')
    os.makedirs(interval_path, exist_ok=True)
    interval_output_path = os.path.join(interval_path, f"{project_name}_{estimator}_confidence_interval.csv")

    # Create a DataFrame for the confidence intervals
    forecast_len = len(predicted_df)  # Length of the forecasted data
    split_point = len(training_df)    # Starting index for the forecasted data
    interval_df = pd.DataFrame({
        'Index': np.arange(split_point, split_point + forecast_len),
        'Lower': np.round(lower_bounds, 2),
        'Upper': np.round(upper_bounds, 2)
    })

    # Save intervals to CSV
    interval_df.to_csv(interval_output_path, index=False)
    print(f"Confidence intervals saved at {interval_output_path}")

# Define the function for ETS model training and evaluation
def evaluate_ets_model(training_df, testing_df, seasonality, project_name, periodicity):
    estimators = ['stan-map', 'stan-mcmc']
    y_test = testing_df['SQALE_INDEX'].values

    for estimator in estimators:
        print(f"Training with estimator={estimator} for project={project_name}")
        
        if(estimator=='stan-map'):
            model = ETS(
            seasonality=seasonality,
            response_col='SQALE_INDEX',
            date_col='COMMIT_DATE',
            estimator=estimator,
            seed=8888,
            n_bootstrap_draws=1000,
        )
        else:
            model = ETS(
            seasonality=seasonality,
            response_col='SQALE_INDEX',
            date_col='COMMIT_DATE',
            estimator=estimator,
            seed=8888,
            num_warmup=400,
            num_sample=1000,
        )

        
        model.fit(df=training_df)
        predicted_df = model.predict(df=testing_df)
        predicted = predicted_df['prediction'].values
        lower_bounds = predicted_df['prediction_5'].values
        upper_bounds = predicted_df['prediction_95'].values

        metrics = {
            'MAE': round(MAE(predicted, y_test), 2),
            'MAPE': round(MAPE(predicted, y_test), 2),
            'RMSE': round(RMSE(predicted, y_test), 2),
            'MSE': round(MSE(predicted, y_test), 2)
        }

        result_data = {
            'Project': project_name,
            'Model': 'ETS',
            'Estimator': estimator,
            **metrics
        }
        save_results(result_data,training_df, predicted_df,lower_bounds,upper_bounds, periodicity, project_name, estimator)

# Main method to trigger predictions for all datasets
def trigger_prediction(df_path, project_name, periodicity=None, seasonality=None):
    df = pd.read_csv(df_path)
    df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
    df['SQALE_INDEX'] = pd.to_numeric(df['SQALE_INDEX'], errors='coerce').dropna()

    split_point = round(len(df) * 0.8)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]
    
    evaluate_ets_model(training_df, testing_df, seasonality, project_name, periodicity)


# function call
def bayesian_orbit_ets():
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