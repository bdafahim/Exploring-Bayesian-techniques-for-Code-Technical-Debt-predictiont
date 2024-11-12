import os
import pandas as pd
import numpy as np
from orbit.models import DLT
from sklearn.metrics import mean_absolute_error
from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE
import json
from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments

# Define the hypertuning function for DLT model
def evaluate_dlt_model(training_df, y_train, x_train, y_test, testing_df, seasonality, project_name, periodicity):
    trend_options = ['linear', 'loglinear']
    estimators = ['stan-map', 'stan-mcmc']

    # Output path to save the results
    base_path = os.path.join(DATA_PATH, 'ORBIT_ML_DLT_Result', periodicity, 'Results')
    os.makedirs(base_path, exist_ok=True)
    plot_path = os.path.join(DATA_PATH, 'ORBIT_ML_DLT_Result', periodicity, 'Plots')
    os.makedirs(plot_path, exist_ok=True)
    # = os.path.join(DATA_PATH, 'ORBIT_ML_DLT_Result', periodicity, 'Coefficient_Interval')
    #os.makedirs(coefficient_interval_path, exist_ok=True)
    interval_path = os.path.join(DATA_PATH, 'ORBIT_ML_DLT_Result', periodicity, 'Confidence Intervals')
    os.makedirs(interval_path, exist_ok=True)
    

    # Iterate over each combination of trend and estimator
    for trend in trend_options:
        for estimator in estimators:
            results = []
            # Define the model with the current set of hyperparameters
            print(f"Training with trend={trend}, estimator={estimator}")
            date_col='COMMIT_DATE'
            response_col='SQALE_INDEX'
            
            if(estimator=='stan-map'):
                model = DLT(
                seasonality=seasonality,
                response_col='SQALE_INDEX',
                date_col='COMMIT_DATE',
                estimator=estimator,
                global_trend_option=trend,
                seed=8888,
                regressor_col=x_train.columns.tolist(),
                n_bootstrap_draws=1000
            )
            else:
                model = DLT(
                seasonality=seasonality,
                response_col='SQALE_INDEX',
                date_col='COMMIT_DATE',
                estimator=estimator,
                global_trend_option=trend,
                seed=8888,
                regressor_col=x_train.columns.tolist(),
                num_warmup=1000,
                num_sample=1000,
            )
            
            # Fit the model
            model.fit(df=training_df)
            
            # Predict and calculate error metrics
            predicted_df = model.predict(df=testing_df, decompose=True)
            predicted = predicted_df['prediction'].values
            lower_bounds = predicted_df['prediction_5'].values
            upper_bounds = predicted_df['prediction_95'].values

            # Plot the results and save the plot
            ax = plot_predicted_data(training_df, predicted_df, date_col, response_col,  title=f'DLT plot for {project_name}_{trend}_{estimator}')
            fig = ax.get_figure()
            plt.close(fig)
            
            #plot = plot_predicted_data(training_df, predicted_df, date_col, response_col,  title=f'DLT plot for {project_name}_{trend}_{estimator}')
            plot_file_name = f"{project_name}_{trend}_{estimator}_plot.pdf"
            plot_output_path = os.path.join(plot_path, plot_file_name)
            fig.savefig(plot_output_path)
            print(f"Plot saved at {plot_output_path}")

            # Define path for saving decomposition plot
            plot_path_d = os.path.join(DATA_PATH, 'Decomposition Plot', 'DLT', periodicity)
            os.makedirs(plot_path_d, exist_ok=True)
            axes_d = plot_predicted_components(predicted_df, date_col,
                                    plot_components=['prediction', 'trend', 'seasonality', 'regression'])
            # Create a new figure for the combined plot with 3 subplots in a single row
            fig, axs = plt.subplots(4, 1, figsize=(12, 16))

            # Plot each component on a separate subplot in the combined figure
            for i, ax in enumerate(axs):
                # Extract data from the original plot and re-plot on the new axis
                for line in axes_d[i].lines:  # Transfer line objects
                    ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
                # Set custom title and labels
                if i == 0:
                    ax.set_title("SQALE INDEX")  # Rename 'prediction' to 'SQALE INDEX'
                else:
                    ax.set_title(axes_d[i].get_title())
                ax.set_xlabel(axes_d[i].get_xlabel())
                ax.set_ylabel(axes_d[i].get_ylabel())
                ax.legend()  # Show legend if any
            # Close the original figure to prevent display issues
            plt.close(axes_d[0].get_figure())
            
            # Save the plot image
            plot_file_name_d = f"{project_name}_{trend}_{estimator}_plot.pdf"
            plot_output_path_d = os.path.join(plot_path_d, plot_file_name_d)
            fig.savefig(plot_output_path_d)
            print(f"Plot saved at {plot_output_path_d}")


            mae = round(MAE(predicted, y_test), 2)
            mape_value = round(MAPE(predicted, y_test), 2)
            mse = round(MSE(predicted, y_test), 2)
            rmse = round(RMSE(predicted, y_test), 2)
            
            if estimator == 'stan-map':
                bic_value = model.get_bic()
                print(f"BIC for trend={trend}, estimator={estimator}: {bic_value:.2f}")
            else:
                bic_value = None
            if estimator == 'stan-mcmc':
                wbic_value = model.fit_wbic(df=training_df)
                print(f"WBIC for trend={trend}, estimator={estimator}: {wbic_value:.2f}")
            else:
                wbic_value = None

            print(f"Final MAE: {mae:.2f}")
            print(f"Final MAPE: {mape_value:.2f}%")
            print(f"Final RMSE: {rmse:.2f}")
            print(f"Final MSE: {mse:.2f}")
            
            # Store the result for this combination
            result_data = {
                'Project': project_name,
                'Model': 'DLT',
                'Trend': trend,
                'Estimator': estimator,
                'MAE': mae,
                'MAPE': mape_value,
                'RMSE': rmse,
                'MSE': mse
            }
            # Conditionally add BIC or WBIC to result_data
            if bic_value is not None:
                result_data['BIC'] = round(bic_value, 2)
            if wbic_value is not None:
                result_data['WBIC'] = round(wbic_value,2)
            results.append(result_data)

            # Save each result in a separate CSV file based on trend and estimator
            file_name = f"{trend}_{estimator}_assesment.csv"
            csv_output_path = os.path.join(base_path, file_name)
            # Save the result as a single-row DataFrame to the specific CSV file
            result_df = pd.DataFrame(results)
            if not os.path.isfile(csv_output_path):
                result_df.to_csv(csv_output_path, mode='w', index=False, header=True)
            else:
                result_df.to_csv(csv_output_path, mode='a', index=False, header=False)

            print(f"Results for trend={trend}, estimator={estimator} saved in {csv_output_path}")

            forecast_len = len(predicted_df)  # Length of the forecasted data
            split_point = len(training_df)    # Starting index for the forecasted data

            # Create and save the confidence intervals DataFrame
            interval_output_path = os.path.join(interval_path, f"{project_name}_{trend}_{estimator}_confidence_intervals.csv")
            interval_df = pd.DataFrame({
                'Index': np.arange(split_point, split_point + forecast_len),
                'Lower': np.round(lower_bounds, 2),
                'Upper': np.round(upper_bounds, 2),
                'Predicted': np.round(predicted, 2),
                'Actual':  np.round(y_test, 2)
            })

            # Save intervals to CSV
            interval_df.to_csv(interval_output_path, index=False)
            print(f"Confidence intervals saved at {interval_output_path}")

    
    # Save all results to a CSV file
    '''results_df = pd.DataFrame(results)
    if not os.path.isfile(csv_output_path):
        results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
    else:
        results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

    print(f"All combination results for {project_name} saved in {csv_output_path}")'''
    return results


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

    # Run hypertuning to get results for all combinations
    results = evaluate_dlt_model(
        training_df=training_df, 
        y_train=y_train, 
        x_train=x_train, 
        y_test=y_test, 
        testing_df=testing_df, 
        seasonality=seasonality,
        project_name=project_name,
        periodicity=periodicity
    )

    return results


# function call
def bayesian_orbit_DLT():
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