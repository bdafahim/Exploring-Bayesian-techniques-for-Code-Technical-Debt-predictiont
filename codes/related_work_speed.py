"""
This script launches the related work forecasting phase
"""

import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import statsmodels.api as sm
from collections import deque

from modules import check_encoding, detect_existing_output
from commons import DATA_PATH
from ts_modelling import assessment_metrics


def regressor_forecast(df, vals_to_predict, periodicity, regressor_name, best_model_cfg, seasonality):
    """
    Obtains the predicted values for the regressors in a greedy format through SARIMA modelling
    """

    arima_order = best_model_cfg[0]
    s_order = best_model_cfg[1]
    predictions = []
    X_train = df[regressor_name].tolist()

    for i in range(vals_to_predict):

        if seasonality:
            model = SARIMAX(X_train, order=arima_order, seasonal_order=s_order, enforce_stationarity=True)
        else:
            model = SARIMAX(X_train, order=arima_order, enforce_stationarity=True)

        try:
            fitted_model = model.fit(disp=0, seasonal=seasonality)
        except np.linalg.LinAlgError:
            print("> Couldn't predict new values for regressor {}".format(regressor_name))
            return [np.nan] * vals_to_predict

        y_pred = fitted_model.get_forecast(steps=1)
        predictions.append(y_pred.predicted_mean[0])
        X_train.append(y_pred.predicted_mean[0])

    print(f">>> Predictions made for regressor {regressor_name}")

    return predictions


def ols_prediction(train_df, test_df, prediction_regressors):
    """
    Performs linear regression prediction on the basis of the predicted values of the regressors with SARIMA modelling.

    train_df - the training data
    test_df - the test data
    prediction_regressors - The variables to use in the prediction phase
    """

    # Assign the data to X and Y.
    regressor_names = train_df.iloc[:, 2:].columns.tolist()
    X = train_df[regressor_names].to_numpy().astype(np.float64)
    # Convert the needed data from df to matrix format
    constant_train = np.ones(shape=(len(X), 1))
    # Explicitly adding the intercept of the Linear Regression
    X = np.concatenate((constant_train, X), axis=1)

    y = train_df["SQALE_INDEX"]
    # Create and fit the model
    model = sm.OLS(y, X).fit()
    real_y_vals = test_df["SQALE_INDEX"]
    X_test = prediction_regressors.to_numpy().astype(np.float64)
    constant_test = np.ones(shape=(len(X_test), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_test = np.concatenate((constant_test, X_test), axis=1)
    predictions = []

    # Perform the walk forward optimization for one step ahead
    for i in range(len(test_df)):

        # New observation for prediction
        new_observation = X_test[i:i+1, :]
        est = model.predict(new_observation)
        predictions.append(np.take(est, 0))

    mape_val, mse_val, mae_val, rmse_val = assessment_metrics(predictions=predictions, real_values=real_y_vals.tolist())

    return [mape_val, mse_val, mae_val, rmse_val, model.aic, model.bic]


def backward_modelling(df, periodicity, vals_to_predict, seasonality, project_name, output_flag=True):
    """
    Finds the best modelling order for the SARIMA model and stores its' parameters, AIC value and useful regressors in
    a JSON file
    """
    # Define the ranges for d and D since we are manually iterating over these
    """
    if seasonality:
        d_range = D_range = range(0, 4)
    else:
        d_range = range(0, 4)
        D_range = [0]  # We don't look into seasonal component
    """
    if periodicity == "monthly":
        s = 12  # Seasonal period
    else:
        s = 26  # Bi-weekly periodicity

    regressors = df.iloc[:, 2:].columns.tolist()

    # Create a dictionary with the predicted values for each predictor
    regressor_dict = {}

    # Stores the results in a json file
    json_dict = {}

    i = 0
    for regressor_name in regressors:

        best_aic = np.inf
        best_model_cfg = None
        best_regressors = None
        variable_array = df[regressor_name].astype(float)

        # Use auto_arima to find the best p, q, P, Q given d and D
        print(f"> Model fitting in process...")
        try:
            if seasonality:
                auto_arima_model = auto_arima(variable_array, m=s, seasonal=True,
                                              stepwise=True, suppress_warnings=True,
                                              error_action='ignore', trace=True,
                                              information_criterion='aic', test='adf')
                P, D, Q = auto_arima_model.seasonal_order[0], auto_arima_model.seasonal_order[1], auto_arima_model.seasonal_order[2]
            else:
                auto_arima_model = auto_arima(variable_array, m=s, seasonal=False,
                                              stepwise=True, suppress_warnings=True,
                                              error_action='ignore', trace=True,
                                              information_criterion='aic', test='adf')
                P, D, Q = np.nan , np.nan , np.nan

            print(f"> Model fitting for regressor {regressor_name} perfomed!")
            # Extract the best ARIMA order and seasonal order found by auto_arima
            p, d, q = auto_arima_model.order[0], auto_arima_model.order[1], auto_arima_model.order[2]

            scored_aic = auto_arima_model.aic()
            print(f"> Best p, q combination: {p} {q} - Seasonal: {P} {Q}")
            print(f"> d: {d}, D: {D}, aic: {round(scored_aic, 2)}")

            if scored_aic < best_aic:
                best_aic = auto_arima_model.aic()
                best_model_cfg = ((p, d, q), (P, D, Q, s))

        except Exception as e:
            print(f"> Error with configuration: {str(e)}")
            output_flag=False
            return output_flag

        # Meaning that the long iterative process for obtaining the best hyperparameter combination, we compute simple
        # auto_arima computation
        """
        if best_aic == np.inf:

            if seasonality:
                auto_arima_model = auto_arima(variable_array, m=s, seasonal=True,
                                              stepwise=True, suppress_warnings=True,
                                              error_action='ignore', trace=False,
                                              information_criterion='aic', maxiter=1000)
                P, Q = auto_arima_model.seasonal_order[0], auto_arima_model.seasonal_order[2]
            else:
                auto_arima_model = auto_arima(variable_array, seasonal=False,
                                              stepwise=True, suppress_warnings=True,
                                              error_action='ignore', trace=False,
                                              information_criterion='aic', maxiter=1000)
                P, Q = np.nan

            p, q = auto_arima_model.order[0], auto_arima_model.order[2]
            best_aic = auto_arima_model.aic()
            print(f"ATTENTION: Hyperparameter identification didn't work, running simple auto_arima...")
            print(f"Best p, q combination: {p} {q} - Seasonal: {P} {Q}")
            print(f"d: {d}, D: {D}")
        """
        if seasonality:
            print(f"> Best SARIMA{best_model_cfg} - AIC:{best_aic} for regressor {regressor_name}")
        else:
            print(f"> Best ARIMA{best_model_cfg} - AIC:{best_aic} for regressor {regressor_name}")

        # Monitorization of the results for each regressor
        parameter_dict = {'model_params': best_model_cfg, 'best_aic': best_aic, "best_regressors": best_regressors}
        json_dict[regressor_name] = parameter_dict

        # Forecasting the values of the regressors for the testing
        sarima_predictions = regressor_forecast(df=df, vals_to_predict=vals_to_predict, periodicity=periodicity,
                                                regressor_name=regressor_name, best_model_cfg=best_model_cfg,
                                                seasonality=seasonality)

        # If there is any nan value in the predicted values for the regressor we exclude it.
        if np.nan in sarima_predictions:
            continue

        # Store the predicted values for the regressors in the dict of predictions
        regressor_dict[regressor_name] = sarima_predictions

        print(f"> Variable {regressor_name} processed and predicted - {i+1}/{len(regressors)}")
        i+=1
    regressor_df = pd.DataFrame.from_dict(regressor_dict)

    if seasonality:
        best_model_path = os.path.join(DATA_PATH, "best_sarima_regressor_models")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))
    else:
        best_model_path = os.path.join(DATA_PATH, "best_arima_models")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))

    json_object = json.dumps(json_dict, indent=4)
    with open(os.path.join(best_model_path, periodicity, f"{project_name}.json"), 'w+') as out:
        out.write(json_object)

    return regressor_df


def relwork_model(df_path, project_name, periodicity, seasonality):
    """
    Performs the modelling of the ARIMA + LM model.

    :param df_path: Path of the existing csv file with project data
    :param project_name: Name of the project
    :param periodicity: Periodicity level between observations
    :return model assessment metrics
    """

    print(f"> Processing project {project_name}")

    # Read the dataframe
    encoding = check_encoding(df_path)
    df = pd.read_csv(df_path, encoding=encoding)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)

    # Dependent variable
    sqale_index = df.SQALE_INDEX.to_numpy()

    # Initial data splitting.
    split_point = round(len(sqale_index)*0.8)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]

    # SARIMA regressor predictors
    predicted_regressors = backward_modelling(df=training_df, periodicity=periodicity, vals_to_predict=len(testing_df),
                                              seasonality=seasonality, project_name=project_name)

    # If the predicted regressors are a boolean means that there was an error with the project for the calculation
    if type(predicted_regressors) is bool:
        print(f"> Regressors for project {project_name} couldn't be predicted for the ARIMA+LM approach")
        return [project_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    # LM prediction phase
    predicted_y_vals = ols_prediction(train_df=training_df, test_df=testing_df,
                                      prediction_regressors=predicted_regressors)

    # return the final output
    assessment_stats = deque(predicted_y_vals)
    assessment_stats.appendleft(project_name)
    assessment_stats = list(assessment_stats)

    return assessment_stats


def related_models(seasonality):
    """
    Executes the tsa process with the related work models
    """
    if seasonality:
        output_directory = "sarima_lm_results"
    else:
        output_directory = "arima_lm_results"

    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    output_path = os.path.join(DATA_PATH, output_directory)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "monthly_results"))
        os.mkdir(os.path.join(output_path, "biweekly_results"))

    # List existing data files:
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)

    assessment_statistics = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE", "AIC", "BIC"]

    for i in range(len(biweekly_files)):

        project = biweekly_files[i][:-4]
        monthly_results_path = os.path.join(output_path, "monthly_results", f"{project}.csv")
        biweekly_results_path = os.path.join(output_path, "biweekly_results", f"{project}.csv")

        biweekly_assessment = pd.DataFrame(columns=assessment_statistics)
        monthly_assessment = pd.DataFrame(columns=assessment_statistics)

        if detect_existing_output(project=project, paths=[monthly_results_path, biweekly_results_path],
                                  flag_num=i, files_num=len(biweekly_files), approach=f"{seasonality}-ARIMA+LM"):
            print(f"> Project {project} already procesed for ARIMA+LM modelling")
            continue

        print(f"> Processing {project} for biweekly data")
        biweekly_statistics = relwork_model(df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
                                            project_name=project, periodicity="biweekly", seasonality=seasonality)

        # Need to at results in the correct format to the DF
        biweekly_assessment.loc[len(biweekly_assessment)] = biweekly_statistics
        biweekly_assessment.to_csv(biweekly_results_path, index=False)

        print(f"> Processing {project} for monthly data")
        monthly_statistics = relwork_model(df_path=os.path.join(monthly_data_path, monthly_files[i]),
                                           project_name=project, periodicity="monthly", seasonality=seasonality)

        monthly_assessment.loc[len(monthly_assessment)] = monthly_statistics
        monthly_assessment.to_csv(monthly_results_path, index=False)

        if seasonality:
            print(f"> SARIMA+LM modelling for project <{project}> performed - {i+1}/{len(biweekly_files)}")
        else:
            print(f"> ARIMA+LM modelling for project <{project}> performed - {i+1}/{len(biweekly_files)}")

    if seasonality:
        print("> SARIMA + LM stage performed!")
    else:
        print("> ARIMA + LM stage performed!")
