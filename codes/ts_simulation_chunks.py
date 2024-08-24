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
from get_best_model_params import get_best_model_params

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


def simulate_sqale_index_arima_future_points(training_df, testing_df, best_model_cfg, best_regressors, steps, index, simulations=30):
    """
    Simulates the SQALE_INDEX based on the SARIMA model.
    
    :param training_df: Training DataFrame with actual SQALE_INDEX and regressors
    :param best_model_cfg: Best SARIMA model configuration obtained from backward_modelling
    :param simulations: Number of simulations to perform
    :param steps: Number of steps to forecast beyond the training data length
    :return: DataFrame with actual and simulated SQALE_INDEX
    """
    results = {}
    y_train = training_df['SQALE_INDEX'].astype(float)
    arima_order = best_model_cfg[0]
    seasonal_order = best_model_cfg[1]
    X_train = training_df[best_regressors].astype(float)
    X_train_scaled = X_train.map(np.log1p)


    for steps in steps:
        last_index = training_df.index[-1]
        simulation_index = range(last_index + 1, last_index + 1 + steps)
        simulated_results = pd.DataFrame(index=simulation_index, 
                                     columns=[f'Simulated_{i}' for i in range(simulations)])

        # Generate future exogenous values for all simulations
        future_exog_dict = build_future_exog(best_model_cfg, training_df, steps, simulations, best_regressors)

        

        for i in range(simulations):
            try:
                model = SARIMAX(y_train, exog=X_train_scaled, order=arima_order,
                            enforce_stationarity=True, enforce_invertibility=True)
                fitted_model = model.fit(disp=False)

                '''last_values = y_train.values[-1:]
                simulated_values = fitted_model.simulate(nsimulations=steps, anchor='end', initial_state=fitted_model.predicted_state[:, -1], exog=future_exog)
                simulated_values = last_values + np.cumsum(simulated_values)  # Accumulate the simulation to follow the trend
                simulated_results.iloc[:, i] = simulated_values'''
                #Extract values from future_exog
                future_exog = future_exog_dict[i].values


                simulated_values = fitted_model.simulate(nsimulations=steps, anchor='end', initial_state=fitted_model.predicted_state[:, -1], exog=future_exog)
                simulated_results.iloc[:, i] = simulated_values
                #simulated_results[f'Simulated_{i}'] = simulated_values
                print(f"> Simulation Values: index:{i} {simulated_values}")
                print(f"> Simulation results: {simulated_results[f'Simulated_{i}']}")
        
            except Exception as e:
                print(f"> Error during simulation {i}: {str(e)}")
                simulated_results[f'Simulated_{i}'] = np.nan

        actual_df = pd.DataFrame({'Actual': y_train}, index=index)

        results[steps] = (actual_df, simulated_results, future_exog_dict)
    
    #actual_df = pd.DataFrame({'Actual': y_train}, index=range(len(y_train)))

    return results


def build_future_exog(best_model_cfg, training_df, steps, simulations, best_regressors):
    future_exog_dict = {}

    # Apply np.log1p to the training data for the best regressors
    training_df_log_transformed = training_df.copy()
    training_df_log_transformed[best_regressors] = np.log1p(training_df[best_regressors])
    
    for i in range(simulations):
        future_exog = pd.DataFrame(index=range(len(training_df), len(training_df) + steps))
            
        for column in best_regressors:  # Assuming the first two columns are not regressors
            y_train = training_df_log_transformed[column].dropna()
            if len(y_train) > 0:
                model = ARIMA(y_train, enforce_stationarity=True, enforce_invertibility=True)
                fitted_model = model.fit()
                
                simulated_values = fitted_model.simulate(nsimulations=steps, anchor='end', initial_state=fitted_model.predicted_state[:, -1])
                
                future_exog[column] = simulated_values

        future_exog[future_exog < 0] = np.nan  # Optionally replace negative values with NaN
        future_exog = future_exog.fillna(0)  # Optionally fill NaN values with 0

   
        future_exog = future_exog[best_regressors]
        future_exog_dict[i] = future_exog


    print(f"Future exogenous values for simulation {i}:")
    print(future_exog_dict)
    print(f':{steps}: <------------------>')
    return future_exog_dict


def simulate_sqale_index_sarima_future_points(training_df, testing_df, best_model_cfg, best_regressors, steps, index, simulations=30):
    """
    Simulates the SQALE_INDEX based on the SARIMA model.
    
    :param training_df: Training DataFrame with actual SQALE_INDEX and regressors
    :param best_model_cfg: Best SARIMA model configuration obtained from backward_modelling
    :param simulations: Number of simulations to perform
    :param steps: Number of steps to forecast beyond the training data length
    :return: DataFrame with actual and simulated SQALE_INDEX
    """
    results = {}
    y_train = training_df['SQALE_INDEX'].astype(float)
    arima_order = best_model_cfg[0]
    seasonal_order = best_model_cfg[1]
    X_train = training_df[best_regressors].astype(float)
    X_train_scaled = X_train.map(np.log1p)


    for steps in steps:
        last_index = training_df.index[-1]
        simulation_index = range(last_index + 1, last_index + 1 + steps)
        simulated_results = pd.DataFrame(index=simulation_index, 
                                     columns=[f'Simulated_{i}' for i in range(simulations)])

        # Generate future exogenous values for all simulations
        future_exog_dict = build_future_exog(best_model_cfg, training_df, steps, simulations, best_regressors)

        

        for i in range(simulations):
            try:
                model = SARIMAX(y_train, exog=X_train_scaled, order=arima_order, seasonal_order=seasonal_order,
                            enforce_stationarity=True, enforce_invertibility=True)
                fitted_model = model.fit(disp=False)

                '''last_values = y_train.values[-1:]
                simulated_values = fitted_model.simulate(nsimulations=steps, anchor='end', initial_state=fitted_model.predicted_state[:, -1], exog=future_exog)
                simulated_values = last_values + np.cumsum(simulated_values)  # Accumulate the simulation to follow the trend
                simulated_results.iloc[:, i] = simulated_values'''
                #Extract values from future_exog
                future_exog = future_exog_dict[i].values


                simulated_values = fitted_model.simulate(nsimulations=steps, anchor='end', initial_state=fitted_model.predicted_state[:, -1], exog=future_exog)
                simulated_results.iloc[:, i] = simulated_values
                #simulated_results[f'Simulated_{i}'] = simulated_values
                print(f"> Simulation Values: index:{i} {simulated_values}")
                print(f"> Simulation results: {simulated_results[f'Simulated_{i}']}")
        
            except Exception as e:
                print(f"> Error during simulation {i}: {str(e)}")
                simulated_results[f'Simulated_{i}'] = np.nan

        actual_df = pd.DataFrame({'Actual': y_train}, index=index)

        results[steps] = (actual_df, simulated_results, future_exog_dict)
    
    #actual_df = pd.DataFrame({'Actual': y_train}, index=range(len(y_train)))

    return results


def build_future_exog(best_model_cfg, training_df, steps, simulations, best_regressors):
    future_exog_dict = {}

    # Apply np.log1p to the training data for the best regressors
    training_df_log_transformed = training_df.copy()
    training_df_log_transformed[best_regressors] = np.log1p(training_df[best_regressors])
    
    for i in range(simulations):
        future_exog = pd.DataFrame(index=range(len(training_df), len(training_df) + steps))
            
        for column in best_regressors:  # Assuming the first two columns are not regressors
            y_train = training_df_log_transformed[column].dropna()
            if len(y_train) > 0:
                model = ARIMA(y_train, enforce_stationarity=True, enforce_invertibility=True)
                fitted_model = model.fit()
                
                simulated_values = fitted_model.simulate(nsimulations=steps, anchor='end', initial_state=fitted_model.predicted_state[:, -1])
                
                future_exog[column] = simulated_values

        future_exog[future_exog < 0] = np.nan  # Optionally replace negative values with NaN
        future_exog = future_exog.fillna(0)  # Optionally fill NaN values with 0

   
        future_exog = future_exog[best_regressors]
        future_exog_dict[i] = future_exog


    print(f"Future exogenous values for simulation {i}:")
    print(future_exog_dict)
    print(f':{steps}: <------------------>')
    return future_exog_dict


def trigger_simulation(df_path, project_name, periodicity, seasonality,steps):

    # DATA PREPARATION (Splitting)
    '''encoding = check_encoding(df_path)
    df = pd.read_csv(df_path, encoding=encoding)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)
    sqale_index = df.SQALE_INDEX.to_numpy()  # Dependent variable
    split_point = round(len(sqale_index)*0.8)  # Initial data splitting. (80% training 20% testing)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]'''
    # Assuming training_df is already loaded and prepared

    encoding = check_encoding(df_path)
    df = pd.read_csv(df_path, encoding=encoding)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)
    sqale_index = df.SQALE_INDEX.to_numpy()  # Dependent variable
    split_point = round(len(sqale_index)*0.25)  # Initial data splitting. (75% for training)
    training_df = df.iloc[split_point:, :]
    print(training_df.tail(10))
    #training_df = df.iloc[:split_point, :]
    testing_df = pd.DataFrame()
    index = df.index[split_point:]  # Get the corresponding indices

    print(f'Backward modeleling started for project>>>>>--- {project_name}')

    '''best_model_cfg, best_aic, best_regressors, output_flag = backward_modelling(
        df=training_df, periodicity=periodicity, seasonality=seasonality
    )'''

    best_model_cfg, best_aic, best_regressors, output_flag = get_best_model_params(
    project_name=project_name,
    periodicity=periodicity,
    training_df=training_df,
    seasonality=seasonality,
    backward_modelling=backward_modelling  # Passing the backward_modelling function as an argument
)



    if seasonality:
        best_model_path = os.path.join(DATA_PATH, "best_sarimax_simulations_models")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))
    else:
        best_model_path = os.path.join(DATA_PATH, "best_arimax_simulations_models")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))

    '''json_dict = {'model_params': best_model_cfg, 'best_aic': best_aic, "best_regressors": best_regressors}
    json_object = json.dumps(json_dict, indent=4)
    with open(os.path.join(best_model_path, periodicity, f"{project_name}.json"), 'w+') as out:
        out.write(json_object)'''

    if output_flag:
        if(seasonality):
            results = simulate_sqale_index_sarima_future_points(training_df, testing_df, best_model_cfg, best_regressors, steps, index)
        else:
            results = simulate_sqale_index_arima_future_points(training_df, testing_df, best_model_cfg, best_regressors, steps, index)

        
        return results, best_regressors
    else:
        print("Model fitting failed. Please check the data and parameters.")
        return None, None
    


def save_and_plot_results(results, project, seasonality, closest_simulations, best_regressors, df_path, periodicity):
    encoding = check_encoding(df_path)
    df = pd.read_csv(df_path, encoding=encoding)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)
    sqale_index = df.SQALE_INDEX.to_numpy()  # Dependent variable
    split_point = round(len(sqale_index) * 0.75)  # Initial data splitting (50% for training)
    training_df = df.iloc[split_point:, :]
    testing_df = pd.DataFrame()
    index = df.index[split_point:]  # Get the corresponding indices

    # Apply np.log1p to the training data for the best regressors
    training_df_log_transformed = training_df.copy()
    training_df_log_transformed[best_regressors] = np.log1p(training_df[best_regressors])

    for steps, (actual_df, simulated_df, future_exog_dict) in results.items():

        # Save to CSV
        if seasonality:
            best_model_path = os.path.join(DATA_PATH, "sarimax_simulations_output", f"{project}")
        else:
            best_model_path = os.path.join(DATA_PATH, "arimax_simulations_output", f"{project}")

        # Ensure all directories exist
        output_folder = os.path.join(best_model_path, 'results', periodicity)
        plots_folder = os.path.join(best_model_path, 'plots', periodicity)
        exog_folder = os.path.join(best_model_path, 'exog_data', periodicity)

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(plots_folder, exist_ok=True)
        os.makedirs(exog_folder, exist_ok=True)

        # Slice the actual_df to include only the last 'steps' values
        sliced_actual_df = actual_df.iloc[-5:]
        simulated_df = simulated_df.round(2)

        sliced_actual_df.rename(columns={'Actual': 'Actual_last_5_vals'}, inplace=True)

        combined_df = pd.concat([sliced_actual_df, simulated_df], axis=1)
        combined_df.to_csv(os.path.join(output_folder, f"{project}_simulations_steps_{steps}.csv"))

        # Create and save future exogenous variables for each simulation
        for sim_index, future_exog in future_exog_dict.items():
            # Extract the simulated future exog values
            future_exog_df = pd.DataFrame(future_exog)

            # Get actual exogenous values for the corresponding steps
            actual_exog_values = training_df_log_transformed[best_regressors].iloc[-steps:]

            # Create a new DataFrame to organize actual and simulated values under each regressor
            combined_exog_df = pd.DataFrame()
            for regressor in best_regressors:
                combined_exog_df[f'{regressor}_Actual'] = actual_exog_values[regressor].reset_index(drop=True)
                combined_exog_df[f'{regressor}_Simulated'] = future_exog_df[regressor].reset_index(drop=True)

            # Save the combined DataFrame with both actual and simulated values
            combined_exog_df.to_csv(
                os.path.join(exog_folder, f"{project}_future_exog_sim_{sim_index}_steps_{steps}.csv"),
                index=False
            )

        # Plotting the actual and simulated data
        plt.figure(figsize=(12, 6))
        plt.plot(actual_df.index, actual_df['Actual'], label='Actual', color='red')
        for column in simulated_df.columns:
            plt.plot(simulated_df.index, simulated_df[column], alpha=0.3)
        plt.plot(simulated_df.index, simulated_df.mean(axis=1), label='Mean of Simulations', color='black', linewidth=2)

        # Uncomment the following lines if you want to highlight the closest simulations
        # for simulation in closest_simulations[steps]:
        #     plt.plot(simulated_df.index, simulated_df[simulation], alpha=0.3, label=f'Closest Simulation {simulation}')

        plt.legend()
        plt.title(f'All Simulations for SQALE_INDEX (Steps = {steps})')
        plt.savefig(os.path.join(plots_folder, f"{project}_simulations_steps_{steps}.png"))
        plt.close()


def assess_closest_simulations(results, seasonality, project, periodicity):
    closest_simulations = {}
    step_ranks = {}


    for steps, (actual_df, simulated_df, future_exog_dict) in results.items():


        if seasonality:
            best_model_path = os.path.join(DATA_PATH, "sarimax_simulations_output", f"{project}")
        else:
            best_model_path = os.path.join(DATA_PATH, "arimax_simulations_output", f"{project}")
        
        # Ensure all directories exist
        output_folder = os.path.join(best_model_path, 'closest_sim', periodicity)
        ranked_steps_output_folder = os.path.join(best_model_path, 'sim_windows_total_deviation', periodicity)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(ranked_steps_output_folder, exist_ok=True)        
        

        # Ensure all data in simulated_df is numeric
        simulated_df = simulated_df.apply(pd.to_numeric, errors='coerce')
        # Fill NaN values with a high number to avoid them being closest by mistake
        simulated_df = simulated_df.fillna(np.inf)
        # Round simulated columns to 2 decimal places
        simulated_df = simulated_df.round(2)


        # Calculate the mean of simulations for each time step
        mean_simulation = simulated_df.mean(axis=1).round(2)


        # Compute the MAE for each simulation from the mean simulation
        mae_values = simulated_df.apply(lambda col: round(MAE(col, mean_simulation), 2), axis=0)
        

        # Identify the simulations with the smallest total deviations
        # Rank the simulations based on the total deviation
        ranked_simulations = mae_values.sort_values().index.tolist()
        closest_simulations[steps] = ranked_simulations

        # Slice the actual_df to include only the last 'steps' values
        sliced_actual_df = actual_df.iloc[-5:]

        sliced_actual_df.rename(columns={'Actual': 'Actual_last_5_vals'}, inplace=True)

        # Save the closest simulations
        closest_sim_df = simulated_df[ranked_simulations]
        combined_df = pd.concat([sliced_actual_df, mean_simulation.rename('Mean_Simulation'), closest_sim_df], axis=1)

        '''# Add individual MAEs to the combined_df
        for simulation in ranked_simulations:
            combined_df[f'MAE_{simulation}'] = mae_values[simulation]'''


        # Prepare to append  MAE values 
        avg_mae_values = ['MAE', None]  # 'Actual' and 'Mean_Simulation' placeholders
        for col in combined_df.columns[2:]:  # Skip 'Actual' and 'Mean_Simulation'
            if 'Simulated_' in col:
                avg_mae_values.append(mae_values[col])
            else:
                avg_mae_values.append(None)  # Fill with None for non-simulation columns

        if len(avg_mae_values) != len(combined_df.columns):
            raise ValueError(f"Column mismatch: expected {len(combined_df.columns)}, got {len(avg_mae_values)}")

        # Append the average MAE row
        avg_mae_row = pd.DataFrame([avg_mae_values], columns=combined_df.columns)
      
        combined_df = pd.concat([combined_df, avg_mae_row], ignore_index=True)

        combined_df.to_csv(os.path.join(output_folder, f"{project}_closest_simulations_steps_{steps}.csv"))


        # Sum the total MAEs for this step and store it
        step_ranks[steps] = mae_values.sum()
    
    # Rank the steps based on the sum of their total MAEs
    ranked_steps = sorted(step_ranks.items(), key=lambda x: x[1])
    ranked_steps_df = pd.DataFrame(ranked_steps, columns=['Steps', 'Total MAE'])
    ranked_steps_df.to_csv(os.path.join(ranked_steps_output_folder, f"{project}_simulation_windows_deviation.csv"), index=False)

    print(f"Ranked steps for {project}: {ranked_steps}")


    return closest_simulations, ranked_steps



def ts_simulation(seasonality):
    """
    Executes the tsa simulatioin process
    """
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data_1")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data_1")


    # List existing data files:
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)

    for i in range(len(biweekly_files)):
        if biweekly_files[i] == '.DS_Store':
            continue
        project = biweekly_files[i][:-4]

        # Runs the SARIMAX execution for the given project in biweekly format
        print(f"> Processing {project} for biweekly data")
        biweekly_statistics, best_regressors_biweekly = trigger_simulation(df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
                                           project_name=project,
                                           periodicity="biweekly",
                                           seasonality=seasonality, steps=[1,2,6,12,24])

      
        print(f"> Processing {project} for monthly data")
        monthly_statistics, best_regressors_monthly = trigger_simulation(df_path=os.path.join(monthly_data_path, monthly_files[i]),
                                          project_name=project,
                                          periodicity="monthly",
                                          seasonality=seasonality, steps=[1,3,6,12])
        
        # Check if either biweekly_statistics or monthly_statistics is None
        if biweekly_statistics is None or monthly_statistics is None:
            print(f"Skipping {project} due to simulation issues.")
            continue


        closest_sim_biwwekly = assess_closest_simulations(biweekly_statistics, seasonality, project, periodicity="biweekly")
        closest_sim_monthly = assess_closest_simulations(monthly_statistics, seasonality, project, periodicity="monthly")

        #closest_sim_biwwekly = assess_and_rank_closest_simulations(biweekly_statistics, biweekly_files, seasonality, periodicity="biweekly")
        #closest_sim_monthly = assess_and_rank_closest_simulations(monthly_statistics, monthly_files, seasonality, periodicity="monthly")

        save_and_plot_results(biweekly_statistics, project, seasonality, closest_sim_biwwekly, best_regressors_biweekly,
                              df_path=os.path.join(biweekly_data_path, biweekly_files[i]), periodicity="biweekly")
        save_and_plot_results(monthly_statistics, project, seasonality, closest_sim_monthly, best_regressors_monthly,
                              df_path=os.path.join(monthly_data_path, monthly_files[i]), periodicity="monthly")

        

        if seasonality:
            print(f"> SARIMAX simulation for project <{project}> performed - {i+1}/{len(biweekly_files)}")
        else:
            print(f"> ARIMAX simulation for project <{project}> performed - {i+1}/{len(biweekly_files)}")

    if seasonality:
        print("> SARIMAX simulationstage performed!")
    else:
        print("> ARIMAX simulation stage performed!")