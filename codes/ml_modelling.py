from modules import assessmentMetrics, check_encoding, change_encoding, detect_existing_output
from commons import DATA_PATH, assessment_statistics

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

    for i in range(len(y_test)):

        model = sm.OLS(y_train, X_train).fit()

        # New observation for prediction
        new_observation = X_test[i:i+1, :]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows

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

    for i in range(len(y_test)):

        # Pipeline to scale features and train the model
        model = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
        model.fit(X_train, y_train)

        # Make prediction for the next observation
        # New observation for prediction
        new_observation = X_test[i:i+1, :]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows

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

    predictions = np.empty(len(X_test))

    for i in range(len(y_test)):
        model = Ridge(alpha=1.0, max_iter=10000)
        model.fit(X_train, y_train)

        new_observation = X_test[i:i+1, :]
        prediction = model.predict(new_observation)
        predictions[i] = np.take(prediction, 0)

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows

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

    predictions = np.empty(len(y_test))

    for i in range(len(X_test)):
        model = Lasso(alpha=1.0, max_iter=10000)
        model.fit(X_train, y_train)

        new_observation = X_test[i:i+1, :]
        prediction = model.predict(new_observation)
        predictions[i] = np.take(prediction, 0)

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows

    print(f"> PROCESSED L1 Lasso Regression for project {pro_name} - periodicity: [{periodicity}]")
    return assessmentMetrics(predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


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

    for i in range(len(y_test)):
        # Using DMatrix for train and test with xgb package.
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test[i:i+1, :].tolist(), label=[y_test[i]])

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

    for i in range(len(y_test)):

        model = RandomForestRegressor(n_estimators=100, random_state=None, n_jobs=-1, bootstrap=False)
        model.fit(X_train, y_train)

        new_observation = X_test[i:i+1, :]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows

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

    for i in range(len(y_test)):
        model = SGDRegressor(max_iter=1000, tol=0.0001)
        model.fit(X_train, y_train)

        new_observation = X_test[i:i+1, :]
        prediction = model.predict(new_observation)
        predictions.append(np.take(prediction, 0))

        # Update the training data
        X_train = np.concatenate((X_train, X_test[i:i+1, :]), axis=0)  # Rows
        y_train = np.append(y_train, y_test[i])  # Rows

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
            biweekly_df = (pd.read_csv(os.path.join(biweekly_data_path, biweekly_files[j]), encoding=bi_encoding)
                           .drop(columns=["COMMIT_DATE"]))
            mo_encoding = check_encoding(path=os.path.join(biweekly_data_path, biweekly_files[j]))
            monthly_df = (pd.read_csv(os.path.join(monthly_data_path, monthly_files[j]), encoding=mo_encoding)
                          .drop(columns=["COMMIT_DATE"]))

            # Due to unknown reasons (the extraction was the same) the encoding is not 'ascii' but 'windows1252'
            com_encoding = check_encoding(path=os.path.join(complete_data_path, complete_files[j]))

            if com_encoding == "Windows-1252":
                complete_df = change_encoding(path=os.path.join(complete_data_path, complete_files[j]))
                new_header = complete_df.iloc[0].tolist()  # First row has the actual headers
                complete_df = complete_df[1:]  # Consider only the dataframe from the first rows with values
                complete_df.columns = new_header
            else:
                complete_df = pd.read_csv(os.path.join(complete_data_path, complete_files[j]), encoding=com_encoding)

            complete_df = complete_df.drop(columns=["COMMIT_DATE"])
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