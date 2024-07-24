import pandas as pd
import numpy as np
import os
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from commons import DATA_PATH
from modules import check_encoding, detect_existing_output
import json

# Helper functions for metrics calculation
def MAPE(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def MSE(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def MAE(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def RMSE(actual, predicted):
    return np.sqrt(MSE(actual, predicted))


def backward_modelling(df, periodicity, seasonality, output_flag=True):
    """
    Finds the best modelling order for the SARIMAX model and stores its parameters, AIC value, and useful regressors in a JSON file
    """
    sqale_index = df.SQALE_INDEX.to_numpy()
    split_point = round(len(sqale_index) * 0.8)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]

    s = 12 if periodicity == "monthly" else 26
    best_aic = np.inf
    best_model_cfg = None
    best_regressors = None

    try:
        current_regressors = training_df.iloc[:, 2:].columns.tolist()
        while current_regressors:
            print(f"> REMAINING REGRESSORS: {len(current_regressors)}")
            if len(current_regressors) > 1:
                aic_with_regressor_removed = []
                i = 0
                for regressor in current_regressors:
                    print(f">Regressor {regressor}")
                    try_regressors = current_regressors.copy()
                    try_regressors.remove(regressor)
                    tmp_X_try = training_df[try_regressors].to_numpy()
                    tmp_X_try_scaled = np.log1p(tmp_X_try)
                    print(f">Try Regressor {tmp_X_try}")
                    print(f">Try Regressor scaled {tmp_X_try_scaled}")

                    try:
                        auto_arima_model = auto_arima(
                            training_df['SQALE_INDEX'].to_numpy(),
                            X=tmp_X_try_scaled,
                            m=s,
                            seasonal=seasonality,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=True,
                            information_criterion='aic',
                            test='adf'
                        )
                        P, D, Q = (auto_arima_model.seasonal_order[0], auto_arima_model.seasonal_order[1],
                                       auto_arima_model.seasonal_order[2])
                        p, d, q = auto_arima_model.order[0], auto_arima_model.order[1], auto_arima_model.order[2]

                        if seasonality:
                            model_try = SARIMAX(
                                training_df['SQALE_INDEX'].to_numpy(),
                                exog=tmp_X_try_scaled,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                enforce_stationarity=True,
                                enforce_invertibility=True
                            )
                        else:
                            model_try = SARIMAX(
                                training_df['SQALE_INDEX'].to_numpy(),
                                exog=tmp_X_try_scaled,
                                order=(p, d, q),
                                enforce_stationarity=True,
                                enforce_invertibility=True
                            )

                        results_try = model_try.fit(disp=0)
                        aic_with_regressor_removed.append((results_try.aic, regressor))

                        if results_try.aic < best_aic:
                            best_aic = results_try.aic
                            best_model_cfg = ((p, d, q), (P, D, Q, s))
                            best_regressors = current_regressors.copy()
                    except ConvergenceWarning:
                        print(f"> Failed to converge for model excluding {regressor}. Skipping...")
                        continue

                aic_with_regressor_removed.sort()
                current_regressors.remove(aic_with_regressor_removed[0][1])
            else:
                break
    except Exception as e:
        print(f"> Error with configuration: {str(e)}")
        output_flag = False

    if seasonality:
        print(f"> Best SARIMAX{best_model_cfg} - AIC:{best_aic} with regressors {best_regressors}")
    else:
        print(f"> Best ARIMAX{best_model_cfg} - AIC:{best_aic} with regressors {best_regressors}")

    return best_model_cfg, round(best_aic, 2), best_regressors, output_flag


def simulate_sqale_index(training_df, best_model_cfg, simulations=100, steps=10):
    """
    Simulates the SQALE_INDEX based on the ARIMA model.
    
    :param training_df: Training DataFrame with actual SQALE_INDEX and regressors
    :param best_model_cfg: Best ARIMA model configuration obtained from backward_modelling
    :param simulations: Number of simulations to perform
    :param steps: Number of steps to forecast beyond the training data length
    :return: DataFrame with actual and simulated SQALE_INDEX
    """
    actual_sqale_index = training_df['SQALE_INDEX']
    arima_order = best_model_cfg[0]
    simulated_results = pd.DataFrame(index=range(len(training_df), len(training_df) + steps), 
                                     columns=[f'Simulated_{i}' for i in range(simulations)])

    for i in range(simulations):
        model = ARIMA(actual_sqale_index, order=arima_order)
        fitted_model = model.fit()
        forecast = fitted_model.get_forecast(steps=steps)
        simulated_values = forecast.predicted_mean
        simulated_results[f'Simulated_{i}'] = simulated_values

    return simulated_results



def assess_simulations(simulated_df):
    """
    Assesses the simulations by computing the mean and standard deviation of the simulated SQALE_INDEX.
    
    :param simulated_df: DataFrame containing simulated SQALE_INDEX
    :return: DataFrame with metrics for each simulation
    """
    mean_simulation = simulated_df.mean(axis=1)

    print(f'Mean of the simulation: {mean_simulation}')
    std_simulation = simulated_df.std(axis=1)
    metrics = []
    
    for column in simulated_df.columns:
        mean_value = mean_simulation
        std_deviation = std_simulation
        mse_val = MSE(mean_simulation, simulated_df[column])
        mae_val = MAE(mean_simulation, simulated_df[column])
        rmse_val = RMSE(mean_simulation, simulated_df[column])
        metrics.append([column, mean_value, std_deviation, mse_val, mae_val, rmse_val])
    
    metrics_df = pd.DataFrame(metrics, columns=['Simulation', 'MEAN', 'STD_DEV', 'MSE', 'MAE', 'RMSE'])
    return metrics_df
    
def trigger_simulation(df_path, project_name, periodicity, seasonality):

    # DATA PREPARATION (Splitting)
    encoding = check_encoding(df_path)
    df = pd.read_csv(df_path, encoding=encoding)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)
    sqale_index = df.SQALE_INDEX.to_numpy()  # Dependent variable
    split_point = round(len(sqale_index)*0.8)  # Initial data splitting. (80% training 20% testing)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]
    # Assuming training_df is already loaded and prepared

    print(f'Backward modeleling started for project>>>>>--- {project_name}')

    best_model_cfg, best_aic, best_regressors, output_flag = backward_modelling(
        df=training_df, periodicity=periodicity, seasonality=seasonality
    )

    if seasonality:
        best_model_path = os.path.join(DATA_PATH, "best_sarimax_simulations")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))
    else:
        best_model_path = os.path.join(DATA_PATH, "best_arimax_simulations")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))

    json_dict = {'model_params': best_model_cfg, 'best_aic': best_aic, "best_regressors": best_regressors}
    json_object = json.dumps(json_dict, indent=4)
    with open(os.path.join(best_model_path, periodicity, f"{project_name}.json"), 'w+') as out:
        out.write(json_object)

    if output_flag:
        simulated_df = simulate_sqale_index(training_df, best_model_cfg)
        metrics_df = assess_simulations(simulated_df)

        print(f"> Metrics df {metrics_df}")

        for index, row in metrics_df.iterrows():
            print(f"Simulation: {row['Simulation']}")
            print(f"  MEAN: {row['MEAN']}")
            print(f"  STD_DEV: {row['STD_DEV']}")
            print(f"  MAE: {row['MAE']}")
            print(f"  MSE: {row['MSE']}")
            print(f"  RMSE: {row['RMSE']}\n")
        

        # Print the metrics    
        # Plotting the first simulation vs actual values
        plt.figure(figsize=(10, 6))
        plt.plot(simulated_df.index, simulated_df['Simulated_99'], label='Simulated_99')
        plt.legend()
        plt.title('Simulated SQALE_INDEX for next 100 points')
        plt.show()
        return [project_name, metrics_df['Simulation'], metrics_df['MEAN'], metrics_df['STD_DEV'], metrics_df['MSE'], metrics_df['RMSE']]
    else:
        print("Model fitting failed. Please check the data and parameters.")
    

def ts_simulation_future(seasonality):
    """
    Executes the tsa simulatioin process
    """

    # Check if Seasonality is taken into consideration
    if seasonality == True:
        output_directory = "sarimax_simulation_results"
    else:
        output_directory = "arimax_simulation_results"

    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data_1")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data_1")
    output_path = os.path.join(DATA_PATH, output_directory)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "monthly_results"))
        os.mkdir(os.path.join(output_path, "biweekly_results"))

    # List existing data files:
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)

    assessment_statistics = ['Simulation', 'MEAN', 'STD_DEV', 'MSE', 'MAE', 'RMSE']
    for i in range(len(biweekly_files)):
        if biweekly_files[i] == '.DS_Store':
            continue
        project = biweekly_files[i][:-4]
        monthly_results_path = os.path.join(output_path, "monthly_results", f"{project}.csv")
        biweekly_results_path = os.path.join(output_path, "biweekly_results", f"{project}.csv")

        biweekly_assessment = pd.DataFrame(columns=assessment_statistics)
        monthly_assessment = pd.DataFrame(columns=assessment_statistics)

        # Check if the project has already been processed
        if detect_existing_output(project=project, paths=[monthly_results_path, biweekly_results_path],
                                  flag_num=i, files_num=len(biweekly_files), approach=f"{seasonality}-ARIMAX"):
            print(f"> Project {project} already procesed for SARIMAX simulation")
            continue

        # Runs the SARIMAX execution for the given project in biweekly format
        print(f">   {project} for biweekly data")
        biweekly_statistics = trigger_simulation(df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
                                           project_name=project,
                                           periodicity="biweekly",
                                           seasonality=seasonality)
        
        print(f'Final biweekly_statistics {biweekly_statistics}')

        biweekly_assessment.loc[len(biweekly_assessment)] = biweekly_statistics
        biweekly_assessment.to_csv(biweekly_results_path, index=False)
        print(f"> Processing {project} for monthly data")
        monthly_statistics = trigger_simulation(df_path=os.path.join(monthly_data_path, monthly_files[i]),
                                          project_name=project,
                                          periodicity="monthly",
                                          seasonality=seasonality)
        
        print(f'Final biweekly_statistics {monthly_statistics}')

        monthly_assessment.loc[len(monthly_assessment)] = monthly_statistics
        monthly_assessment.to_csv(monthly_results_path, index=False)

        if seasonality:
            print(f"> SARIMAX simulation for project <{project}> performed - {i+1}/{len(biweekly_files)}")
        else:
            print(f"> ARIMAX simulation for project <{project}> performed - {i+1}/{len(biweekly_files)}")

    if seasonality:
        print("> SARIMAX simulationstage performed!")
    else:
        print("> ARIMAX simulation stage performed!")