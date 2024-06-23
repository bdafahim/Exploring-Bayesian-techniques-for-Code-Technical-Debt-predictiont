from modules import assessmentMetrics, check_encoding, change_encoding, detect_existing_output, RSS, AIC, MAE
from commons import DATA_PATH, assessment_statistics, INITIAL_VARS

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import xgboost as xgb
import statsmodels.api as sm
import json


def parameter_store_json(best_var_names, best_metric, pro_name, metric_name, model_name, periodicity):
    """
    Stores the best model parameters into a json file
    """

    ml_model_path = os.path.join(DATA_PATH, "best_ml_model_params")
    if not os.path.exists(ml_model_path):
        os.mkdir(ml_model_path)
    periodicity_path = os.path.join(ml_model_path, periodicity)
    if not os.path.exists(periodicity_path):
        os.mkdir(periodicity_path)
    ml_spec_path = os.path.join(periodicity_path, model_name)
    if not os.path.exists(ml_spec_path):
        os.mkdir(ml_spec_path)
    best_variables_path = os.path.join(ml_spec_path, f"{pro_name}_best_variables.json")
    json_dict = {"best_var_names": best_var_names, "best_aic": best_metric}

    with open(best_variables_path, 'w') as f:
        json.dump(json_dict, f)

    print(f"> Best parameter combination and AIC for project {pro_name}\n")
    if metric_name == "aic":
        print(f"> Best AIC: {best_metric}\n")
    else:
        print(f"> Best MAE: {best_metric}\n")
    print(f"> Best parameters: {best_var_names}")


def mlr_optimization(y_train, X_train):
    """
    Optimizes MLR model based on AIC parameter optimization
    """
    # Initial model with all variables
    model = sm.OLS(endog=y_train, exog=X_train).fit()
    best_aic = model.aic
    best_variables = list(range(X_train.shape[1]))  # List of current variables indexes

    print(f"> Starting parameter optimization for MLR model...")
    # Backward variable elimination based on AIC
    improved = True
    while improved:
        improved = False
        for i in best_variables[1:]:  # COnsidering the first one as the intercept
            try_variables = [v for v in best_variables if v != i]
            try_model = sm.OLS(endog=y_train, exog=X_train[:, try_variables]).fit()
            if try_model.aic < best_aic:
                best_aic = try_model.aic
                best_variables = try_variables  # Don't forget about the first index with the intercept
                improved = True

            print(f"> Parameter tuning at work - current best vars: {len(best_variables)} | best AIC: {best_aic}\n")

    print(f"> Parameter optimization for MLR model completed - best vars: {len(best_variables)} | best AIC: {best_aic}")
    # Convert the index into the variable names from the original dataframe to store the variable names in a json file
    initial_vars = ['Intercept']
    initial_vars.extend(INITIAL_VARS)
    best_var_names = [initial_vars[i-1] for i in best_variables[1:]]  # Adjusting for constant
    return best_var_names, best_aic, best_variables


def L_optimization(y_train, X_train, model_name):
    """
    Optimizes Ridge and Lasso model based on manual AIC parameter optimization
    """
    # Initial model with all variables

    if model_name == 'L2':  # Ridge
        model = Ridge(alpha=1.0, max_iter=1000)
    elif model_name == 'sgd':  # Stochastic Gradient Descent
        model = SGDRegressor(max_iter=1000, tol=0.0001)
    else:  # L1 Lasso
        model = Lasso(alpha=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    rss_val = RSS(y=y_train, X=X_train, model=model)
    best_aic = AIC(n=X_train.shape[0], k=X_train.shape[1]+1, rss=rss_val)
    best_variables = list(range(X_train.shape[1]))  # List of current variables indexes

    print(f"> Starting parameter optimization for {model_name} model...")
    # Backward variable elimination based on AIC
    improved = True
    while improved:
        improved = False
        for i in best_variables[1:]:  # COnsidering the first one as the intercept
            try_variables = [v for v in best_variables if v != i]
            if model_name == 'L2':
                try_model = Ridge(alpha=1.0, max_iter=1000)
            else:
                try_model = Lasso(alpha=1.0, max_iter=1000)
            try_model.fit(X_train[:, try_variables], y_train)
            try_rss = RSS(y=y_train, X=X_train[:, try_variables], model=try_model)
            try_aic = AIC(n=X_train.shape[0], k=len(try_variables)+1, rss=try_rss)
            if try_aic < best_aic:
                best_aic = try_aic
                best_variables = try_variables  # Don't forget about the first index with the intercept
                improved = True
            print(f"> Parameter tuning at work - current best vars: {len(best_variables)} | best AIC: {best_aic}\n")

    print(f"> Parameter optimization for {model_name} model completed - best vars: {len(best_variables)} | best AIC: {best_aic}")
    # Convert the index into the variable names from the original dataframe to store the variable names in a json file
    initial_vars = ['Intercept']
    initial_vars.extend(INITIAL_VARS)
    best_var_names = [initial_vars[i-1] for i in best_variables[1:]]  # Adjusting for constant
    return best_var_names, best_aic, best_variables


def mlr_regression(training_df, testing_df, pro_name, periodicity):
    """
    Performs the Multiple Linear Regression on the provided project.

    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """

    print(f"> Multiple Linear Regression for project {pro_name} - periodicity: [{periodicity}]")
    predictions = []

    # preparing the data
    X_train = training_df.iloc[:, 1:].to_numpy()  # Independent variable
    y_train = training_df.iloc[:, 0].to_numpy()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].to_numpy()
    y_test = testing_df.iloc[:, 0].to_numpy().astype(np.float64)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the needed data from df to matrix format
    constant_train = np.ones(shape=(len(X_train), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_train = np.concatenate((constant_train, X_train_scaled), axis=1)
    constant_test = np.ones(shape=(len(X_test), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_test = np.concatenate((constant_test, X_test_scaled), axis=1)

    # Backward parameter optimization
    best_var_names, best_aic, best_var_indices = mlr_optimization(y_train=y_train, X_train=X_train)
    # Save it into json
    parameter_store_json(best_var_names=best_var_names, best_metric=best_aic, pro_name=pro_name, metric_name='aic',
                         model_name='mlr', periodicity=periodicity)

    # Walk Forward train test validation
    for i in range(len(y_test)):

        model = sm.OLS(y_train, X_train[:, best_var_indices]).fit()

        # New observation for prediction
        new_observation = X_test[i:i+1, best_var_indices]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows
        print(f"> MLR testing process for project {pro_name} - {i+1}/{len(y_test)}")

    print(f"> PROCESSED Multivariate Linear Regression for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def svm_regression(training_df, testing_df, pro_name, periodicity):
    """
    Performs the Support Vector Regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Support Vector Machine Regression for project {pro_name} - periodicity: [{periodicity}]")

    # Empty array for the predictions
    predictions = []

    # Separating independent vars from dependent variable
    X_train = training_df.iloc[:, 1:].to_numpy()  # Independent variable
    y_train = training_df.iloc[:, 0].to_numpy()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].to_numpy()
    y_test = testing_df.iloc[:, 0].to_numpy()

    # Standardizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Backward parameter optimization
    best_var_names, best_mae, best_var_indices = nl_optimization(y_train=y_train, X_train=X_train,
                                                                 y_test=y_test, X_test=X_test,
                                                                 model_name='svr')
    # Save it into json
    parameter_store_json(best_var_names=best_var_names, best_metric=best_mae, pro_name=pro_name, metric_name='mae',
                         model_name='svr', periodicity=periodicity)

    for i in range(len(y_test)):

        # Pipeline to scale features and train the model
        model = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
        model.fit(X_train[:, best_var_indices], y_train)

        # Make prediction for the next observation
        # New observation for prediction
        new_observation = X_test[i:i+1, best_var_indices]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows
        print(f"> SVM testing process for project {pro_name} - {i+1}/{len(y_test)}")

    print(f"> PROCESSED Support Vector Machine Regression for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def ridge_regression(training_df, testing_df, pro_name, periodicity):
    """
    Performs the L2 Ridge regression on the provided project.

    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Ridge L2 Regression for project {pro_name} - periodicity: [{periodicity}]")

    # preparing the data
    X_train = training_df.iloc[:, 1:].to_numpy()  # Independent variable
    y_train = training_df.iloc[:, 0].to_numpy()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].to_numpy()
    y_test = testing_df.iloc[:, 0].to_numpy().astype(np.float64)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the needed data from df to matrix format
    constant_train = np.ones(shape=(len(X_train), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_train = np.concatenate((constant_train, X_train_scaled), axis=1)
    constant_test = np.ones(shape=(len(X_test), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_test = np.concatenate((constant_test, X_test_scaled), axis=1)

    # Backward parameter optimization
    best_var_names, best_aic, best_var_indices = L_optimization(y_train=y_train, X_train=X_train, model_name='L2')
    # Save it into json
    parameter_store_json(best_var_names=best_var_names, best_metric=best_aic, pro_name=pro_name, metric_name='aic',
                         model_name='L2', periodicity=periodicity)

    predictions = np.empty(len(X_test))

    for i in range(len(y_test)):
        model = Ridge(alpha=1.0, max_iter=1000)
        model.fit(X_train[:, best_var_indices], y_train)

        new_observation = X_test[i:i+1, best_var_indices]
        prediction = model.predict(new_observation)
        predictions[i] = np.take(prediction, 0)

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows
        print(f"> L2 testing process for project {pro_name} - {i+1}/{len(y_test)}")

    print(f"> PROCESSED L2 Ridge Regression for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def lasso_regression(training_df, testing_df, pro_name, periodicity):
    """
    Performs the L1 Lasso regression on the provided project.
    :param training_df
    :param testing_df
    :param pro_name
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Lasso L1 Regression for project {pro_name} - periodicity: [{periodicity}]")

    # preparing the data
    X_train = training_df.iloc[:, 1:].to_numpy()  # Independent variable
    y_train = training_df.iloc[:, 0].to_numpy()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].to_numpy()
    y_test = testing_df.iloc[:, 0].to_numpy().astype(np.float64)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the needed data from df to matrix format
    constant_train = np.ones(shape=(len(X_train), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_train = np.concatenate((constant_train, X_train_scaled), axis=1)
    constant_test = np.ones(shape=(len(X_test), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_test = np.concatenate((constant_test, X_test_scaled), axis=1)

    # Backward parameter optimization
    best_var_names, best_aic, best_var_indices = L_optimization(y_train=y_train, X_train=X_train, model_name='L1')
    # Save it into json
    parameter_store_json(best_var_names=best_var_names, best_metric=best_aic, pro_name=pro_name, metric_name='aic',
                         model_name='L1', periodicity=periodicity)

    predictions = np.empty(len(y_test))

    for i in range(len(X_test)):
        model = Lasso(alpha=1.0, max_iter=1000)
        model.fit(X_train[:, best_var_indices], y_train)

        new_observation = X_test[i:i+1, best_var_indices]
        prediction = model.predict(new_observation)
        predictions[i] = np.take(prediction, 0)

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows
        print(f"> L1 testing process for project {pro_name} - {i+1}/{len(y_test)}")

    print(f"> PROCESSED L1 Lasso Regression for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def xgb_optimization(y_train, X_train, y_test, X_test):
    """
    Optimizes parameter selection for xgboost model
    """

    # Using DMatrix for train and test with xgb package.
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define model params
    params = {
        "max_depth": 3,
        "eta": 0.1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse"
    }

    # Model metric
    watchlist = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist, early_stopping_rounds=100, verbose_eval=False)

    predictions_init = model.predict(dtest)
    best_mae = MAE(predicted_vals=predictions_init, testing_vals=y_test)
    best_variables = list(range(X_train.shape[1])) # List of current variables indexes
    # Backward variable elimination based on AIC
    # Backward variable elimination based on AIC
    improved = True
    print(f"> Starting parameter optimization for XGB model...")
    while improved:
        improved = False
        for i in best_variables[1:]:  # COnsidering the first one as the intercept
            try_variables = [v for v in best_variables if v != i]
            # Using DMatrix for train and test with xgb package.
            try_dtrain = xgb.DMatrix(X_train[:, try_variables], label=y_train)
            try_dtest = xgb.DMatrix(X_test[:, try_variables], label=y_test)

            # Define model params
            try_params = {
                "max_depth": 3,
                "eta": 0.1,
                "objective": "reg:squarederror",
                "eval_metric": "rmse"
            }

            # Model metric
            try_watchlist = [(try_dtrain, "train"), (try_dtest, "test")]
            try_model = xgb.train(try_params, try_dtrain, num_boost_round=500, evals=try_watchlist, early_stopping_rounds=100, verbose_eval=False)

            try_predictions = try_model.predict(try_dtest)
            try_mae = MAE(predicted_vals=try_predictions, testing_vals=y_test)
            if try_mae < best_mae:
                best_mae = try_mae
                best_variables = try_variables  # Don't forget about the first index with the intercept
                improved = True
            print(f"> Parameter tuning at work - current best vars: {len(best_variables)} | best AIC: {best_mae}\n")

    print(f"> Parameter optimization for MLR model completed - best vars: {len(best_variables)} | best AIC: {best_mae}")
    # Convert the index into the variable names from the original dataframe to store the variable names in a json file
    best_var_names = [INITIAL_VARS[i] for i in best_variables]  # Adjusting for constant
    return best_var_names, best_mae, best_variables


def rf_optimization(y_train, X_train, y_test, X_test):
    """
    Optimizes parameter selection for random forest model
    """

    model = RandomForestRegressor(n_estimators=100, random_state=None, n_jobs=-1, bootstrap=False)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    best_mae = MAE(predicted_vals=predictions, testing_vals=y_test)
    best_variables = list(range(X_train.shape[1]))  # List of current variables indexes

    # Backward variable elimination based on AIC
    improved = True
    print(f"> Starting parameter optimization for RF model...")
    while improved:
        improved = False
        for i in best_variables[1:]:  # COnsidering the first one as the intercept
            try_variables = [v for v in best_variables if v != i]
            try_model = RandomForestRegressor(n_estimators=100, random_state=None, n_jobs=-1, bootstrap=False)
            try_model.fit(X_train[:, try_variables], y_train)
            try_predictions = try_model.predict(X_test[:, try_variables])
            try_mae = MAE(predicted_vals=try_predictions, testing_vals=y_test)
            if try_mae < best_mae:
                best_mae = try_mae
                best_variables = try_variables  # Don't forget about the first index with the intercept
                improved = True
            print(f"> Parameter tuning at work - current best vars: {len(best_variables)} | best AIC: {best_mae}\n")

    print(f"> Parameter optimization for RF model completed - best vars: {len(best_variables)} | best AIC: {best_mae}")
    # Convert the index into the variable names from the original dataframe to store the variable names in a json file
    best_var_names = [INITIAL_VARS[i] for i in best_variables]
    return best_var_names, best_mae, best_variables


def svr_optimization(y_train, X_train, y_test, X_test):
    """
    Optimizes parameter selection for Support vector regression model
    """
    model = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    best_mae = MAE(predicted_vals=predictions, testing_vals=y_test)
    best_variables = list(range(X_train.shape[1]))  # List of current variables indexes

    # Backward variable elimination based on AIC
    print(f"> Starting parameter optimization for SVR model...")
    improved = True
    while improved:
        improved = False
        for i in best_variables[1:]:  # COnsidering the first one as the intercept
            try_variables = [v for v in best_variables if v != i]
            try_model = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
            try_model.fit(X_train[:, try_variables], y_train)
            try_predictions = try_model.predict(X_test[:, try_variables])
            try_mae = MAE(predicted_vals=try_predictions, testing_vals=y_test)
            if try_mae < best_mae:
                best_mae = try_mae
                best_variables = try_variables  # Don't forget about the first index with the intercept
                improved = True
            print(f"> Parameter tuning at work - current best vars: {len(best_variables)} | best AIC: {best_mae}\n")

    print(f"> Parameter optimization for MLR model completed - best vars: {len(best_variables)} | best AIC: {best_mae}")
    # Convert the index into the variable names from the original dataframe to store the variable names in a json file
    best_var_names = [INITIAL_VARS[i] for i in best_variables]
    return best_var_names, best_mae, best_variables


def nl_optimization(y_train, X_train, y_test, X_test, model_name):
    """
    Performs parameter optimization for non-linear regression ML models, which require use of the whole test set
    :param y_train: as name says
    :param X_train: as name says
    :param model_name: as name says
    """

    if model_name == 'xgb':  # XGBoost
        best_var_names, best_aic, best_var_indices = xgb_optimization(y_train, X_train, y_test, X_test)
    elif model_name == 'rf':  # Random Forest
        best_var_names, best_aic, best_var_indices = rf_optimization(y_train, X_train, y_test, X_test)
    else:  # Support vector regression
        best_var_names, best_aic, best_var_indices = svr_optimization(y_train, X_train, y_test, X_test)

    return best_var_names, best_aic, best_var_indices


def xgboost(training_df, testing_df, pro_name, periodicity):
    """
    Performs the XGBoost regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """

    print(f"> XGBoost for project {pro_name} - periodicity: [{periodicity}]")
    predictions = []
    X_train = training_df.iloc[:, 1:].to_numpy()
    y_train = training_df.iloc[:, 0].to_numpy()
    X_test = testing_df.iloc[:, 1:].to_numpy()
    y_test = testing_df.iloc[:, 0].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Backward parameter optimization
    best_var_names, best_mae, best_var_indices = nl_optimization(y_train=y_train, X_train=X_train,
                                                                 y_test=y_test, X_test=X_test,
                                                                 model_name='xgb')
    # Save it into json
    parameter_store_json(best_var_names=best_var_names, best_metric=best_mae, pro_name=pro_name, metric_name='mae',
                         model_name='xgb', periodicity=periodicity)

    for i in range(len(y_test)):
        # Using DMatrix for train and test with xgb package.
        dtrain = xgb.DMatrix(X_train[:, best_var_indices], label=y_train)
        dtest = xgb.DMatrix(X_test[i:i+1, best_var_indices].tolist(), label=[y_test[i]])

        # Define model params
        params = {
            "max_depth": 3,
            "eta": 0.1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }

        # Model metric
        watchlist = [(dtrain, "train"), (dtest, "test")]
        model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist, early_stopping_rounds=100, verbose_eval=False)

        prediction = model.predict(dtest)
        predictions.append(prediction[0])

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows
        print(f"> XGBoost testing process for project {pro_name} - {i+1}/{len(y_test)}")

    print(f"> PROCESSED XGBoost for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def rf_forest(training_df, testing_df, pro_name, periodicity):
    """
    Performs the Random Forest Regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Random Forest Regression for project {pro_name} - periodicity: [{periodicity}]")

    predictions = []
    # Separating independent vars from dependent variable
    X_train = training_df.iloc[:, 1:].to_numpy()  # Independent variable
    y_train = training_df.iloc[:, 0].to_numpy()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].to_numpy()
    y_test = testing_df.iloc[:, 0].to_numpy()

    # Standardizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Backward parameter optimization
    best_var_names, best_mae, best_var_indices = nl_optimization(y_train=y_train, X_train=X_train,
                                                                 y_test=y_test, X_test=X_test,
                                                                 model_name='rf')
    # Save it into json
    parameter_store_json(best_var_names=best_var_names, best_metric=best_mae, pro_name=pro_name, metric_name='mae',
                         model_name='rf', periodicity=periodicity)

    for i in range(len(y_test)):

        model = RandomForestRegressor(n_estimators=100, random_state=None, n_jobs=-1, bootstrap=False)
        model.fit(X_train[:, best_var_indices], y_train)

        new_observation = X_test[i:i+1, best_var_indices]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows
        print(f"> RF testing process for project {pro_name} - {i+1}/{len(y_test)}")

    print(f"> PROCESSED Random Forest Regression for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def sgd_regression(training_df, testing_df, pro_name, periodicity):
    """
     Performs the Stochastic Gradient Descent Regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """

    print(f"> Stochastic Gradient Descent for project {pro_name} - periodicity: [{periodicity}]")
    predictions = []

    # Vectorized format for model features.
    X_train = training_df.iloc[:, 1:].to_numpy()  # Independent variable
    y_train = training_df.iloc[:, 0].to_numpy()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].to_numpy()
    y_test = testing_df.iloc[:, 0].to_numpy().astype(np.float64)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the needed data from df to matrix format
    constant_train = np.ones(shape=(len(X_train), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_train = np.concatenate((constant_train, X_train_scaled), axis=1)
    constant_test = np.ones(shape=(len(X_test), 1))
    # Explicitly adding the intercept of the Linear Regression
    X_test = np.concatenate((constant_test, X_test_scaled), axis=1)

    # Backward parameter optimization
    best_var_names, best_aic, best_var_indices = L_optimization(y_train=y_train, X_train=X_train, model_name='sgd')
    # Save it into json
    parameter_store_json(best_var_names=best_var_names, best_metric=best_aic, pro_name=pro_name, metric_name='aic',
                         model_name='sgd', periodicity=periodicity)

    for i in range(len(y_test)):
        model = SGDRegressor(max_iter=1000, tol=0.0001)
        model.fit(X_train[:, best_var_indices], y_train)

        new_observation = X_test[i:i+1, best_var_indices]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows
        print(f"> SGD testing process for project {pro_name} - {i+1}/{len(y_test)}")

    print(f"> PROCESSED Stochastic Gradient Descent for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def ml_models():
    """
    Executes all the Ml models defined in the study
    :return:
    """

    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    complete_data_path = os.path.join(DATA_PATH, "complete_data")
    output_path = os.path.join(DATA_PATH, "ML_results")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ml_results_path = os.path.join(output_path, "results_ML_sarimax")
    if not os.path.exists(ml_results_path):
        os.mkdir(ml_results_path)

    # data files
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)
    complete_files = os.listdir(complete_data_path)

    ml_model_names = ["mlr", "svr", "L1", "L2", "xgb", "rf", "sgd"]
    models = [mlr_regression, svm_regression, ridge_regression, lasso_regression, xgboost, rf_forest, sgd_regression]

    for i in range(len(models)):

        model_results_path = os.path.join(ml_results_path, ml_model_names[i])

        # Creation of nidek results direcotry
        if not os.path.exists(model_results_path):
            os.mkdir(model_results_path)
        monthly_path = os.path.join(ml_results_path, f"{ml_model_names[i]}/monthly")
        biweekly_path = os.path.join(ml_results_path, f"{ml_model_names[i]}/biweekly")
        complete_path = os.path.join(ml_results_path, f"{ml_model_names[i]}/complete")

        # Checks if the directories for the given models and periodicity levels has already been implemented.
        if not os.path.exists(monthly_path):
            os.mkdir(monthly_path)
        if not os.path.exists(biweekly_path):
            os.mkdir(biweekly_path)
        if not os.path.exists(complete_path):
            os.mkdir(complete_path)

        # Creation of the dataframe to hold the model assessment results
        monthly_assessment_df = pd.DataFrame(columns=assessment_statistics)
        biweekly_assessment_df = pd.DataFrame(columns=assessment_statistics)
        complete_assessment_df = pd.DataFrame(columns=assessment_statistics)

        for j in range(len(biweekly_files)):

            project_name = biweekly_files[j][:-4]  # Removes the .csv extension from the project name
            # Removing extra index number and commit_date columns from the original dataset

            bi_encoding = check_encoding(path=os.path.join(biweekly_data_path, biweekly_files[j]))
            if not bi_encoding or bi_encoding == 'ascii':  # 'ascii' can be a sign of an uncertain detection
                bi_encoding = 'cp1252'  # Fallback encoding
            biweekly_df = (pd.read_csv(os.path.join(biweekly_data_path, biweekly_files[j]), encoding=bi_encoding).drop(columns=["COMMIT_DATE"]))
            mo_encoding = check_encoding(path=os.path.join(biweekly_data_path, biweekly_files[j]))
            if not mo_encoding or mo_encoding == 'ascii':  # 'ascii' can be a sign of an uncertain detection
                mo_encoding = 'cp1252'  # Fallback encoding
            monthly_df = (pd.read_csv(os.path.join(monthly_data_path, monthly_files[j]), encoding=mo_encoding).drop(columns=["COMMIT_DATE"]))
            
            # Due to unknown reasons (the extraction was the same) the encoding is not 'ascii' but 'windows1252'
            com_encoding = check_encoding(path=os.path.join(complete_data_path, complete_files[j]))

            if com_encoding == "Windows-1252":
                complete_df = change_encoding(path=os.path.join(complete_data_path, complete_files[j]))
                new_header = complete_df.iloc[0].tolist()  # First row has the actual headers
                complete_df = complete_df[1:]  # Consider only the dataframe from the first rows with values
                complete_df.columns = new_header
            else:
                complete_df = (pd.read_csv(os.path.join(complete_data_path, complete_files[j]), encoding=com_encoding).drop(columns=["COMMIT_DATE"]))
            
            # Splitting the data into train set (80%) and test set (20%)
            biweekly_train_size = round(0.8 * len(biweekly_df))
            monthly_train_size = round(0.8 * len(monthly_df))
            complete_train_size = round(0.8 * len(complete_df))

            biweekly_train = biweekly_df.iloc[:biweekly_train_size, :]
            monthly_train = monthly_df.iloc[:monthly_train_size, :]
            complete_train = complete_df.iloc[:complete_train_size, :]

            biweekly_test = biweekly_df.iloc[biweekly_train_size:, :]
            monthly_test = monthly_df.iloc[monthly_train_size:, :]
            complete_test = complete_df.iloc[complete_train_size:, :]

            # Model building & assessment
            biweekly_results = models[i](biweekly_train, biweekly_test, project_name, 'biweekly')
            monthly_results = models[i](monthly_train, monthly_test, project_name, 'monthly')
            complete_results = models[i](complete_train, complete_test, project_name, 'complete')

            # Updating biweekly results
            biweekly_assessment_df = pd.concat([biweekly_assessment_df, biweekly_results], axis=0)
            # Updating monthly results
            monthly_assessment_df = pd.concat([monthly_assessment_df, monthly_results], axis=0)
            # updating complete results
            complete_assessment_df = pd.concat([complete_assessment_df, complete_results], axis=0)

            print(f"> <{ml_model_names[i]}> ML modelling for project <{project_name}> performed - "
                  f"{j+1}/{len(biweekly_files)} projects - {i+1} of {len(models)} models")

        biweekly_assessment_df.to_csv(os.path.join(biweekly_path, f"assessment.csv"), index=False)
        monthly_assessment_df.to_csv(os.path.join(monthly_path, f"assessment.csv"), index=False)
        complete_assessment_df.to_csv(os.path.join(complete_path, f"assessment.csv"), index=False)

    print("> ML MODELLING STAGE COMPLETED!")