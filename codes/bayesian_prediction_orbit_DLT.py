import os
import pandas as pd
import numpy as np
from orbit.models import DLT
from sklearn.metrics import mean_absolute_error
from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import itertools





# Function to perform Lasso Regression for feature selection
def select_features_with_lasso(X, y, alpha=0.01):
    # Standardize the features (important for Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lasso Regression for feature selection
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    
    # Select features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    important_features = X.columns[selected_features]
    
    print(f"Selected Features: {important_features.tolist()}")
    
    return important_features


# Backward modeling for feature selection
def hyper_tune_with_backward_select_regressors(training_df, y_train, x_train, y_test, testing_df, seasonality):
    best_mae = np.inf
    best_model = None
    best_config = None
    best_regressors = None

    current_regressors = x_train.columns.tolist()

    while current_regressors:
        print(f"> REMAINING REGRESSORS: {len(current_regressors)}")

        if len(current_regressors) > 1:
            mae_with_regressor_removed = []

            for regressor in current_regressors:
                print(f"> Trying with regressor removed: {regressor}")
                try_regressors = current_regressors.copy()
                try_regressors.remove(regressor)

                try:
                    # Iterate over trend options and estimators
                    for trend in ['linear', 'loglinear', 'flat', 'logistic']:
                        for estimator in ['stan-map', 'stan-mcmc']:
                            # Define and fit the model
                            print(f"Training with trend={trend}, estimator={estimator}, regressors={try_regressors}")
                            model = DLT(
                                seasonality=seasonality,
                                response_col='SQALE_INDEX',
                                date_col='COMMIT_DATE',
                                estimator=estimator,
                                global_trend_option=trend,
                                seed=8888,
                                regressor_col=try_regressors,
                                n_bootstrap_draws=1000
                            )

                            model.fit(df=training_df)

                            # Predict and calculate MAE
                            predicted_df = model.predict(df=testing_df)
                            predicted = predicted_df['prediction'].values
                            mae = mean_absolute_error(y_test, predicted)

                            print(f"MAE for trend={trend}, estimator={estimator}, regressors={try_regressors}: {mae:.2f}")

                            mae_with_regressor_removed.append((mae, regressor))

                            # Check if this is the best model
                            if mae < best_mae:
                                best_mae = mae
                                best_model = model
                                best_config = {
                                    'trend': trend,
                                    'estimator': estimator,
                                }
                                best_regressors = try_regressors.copy()

                except Exception as e:
                    print(f"> Error when trying model excluding {regressor}: {str(e)}")

            # Sort based on MAE and remove the regressor with the lowest MAE impact
            mae_with_regressor_removed.sort()
            regressor_to_remove = mae_with_regressor_removed[0][1]
            current_regressors.remove(regressor_to_remove)
            print(f"Regressor {regressor_to_remove} removed. Remaining regressors: {current_regressors}")

        else:
            print("> Only one regressor left, stopping.")
            break

    print(f"Best model found with regressors: {best_regressors} and MAE: {best_mae:.2f}")
    return best_model, best_config, best_regressors


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

    # Hypertune the DLT model
    best_model, best_config, important_features = hyper_tune_with_backward_select_regressors(
        training_df=training_df, 
        y_train=y_train, 
        x_train=x_train, 
        y_test=y_test, 
        testing_df=testing_df, 
        seasonality=seasonality
    )

     # Ensure COMMIT_DATE is included in the DataFrame used for prediction along with important features
    test_df_for_prediction = testing_df[['COMMIT_DATE']].copy()  # Ensure COMMIT_DATE is present
    test_df_for_prediction = pd.concat([test_df_for_prediction, testing_df[important_features]], axis=1)

    # Use the best model for final predictions
    predicted_df = best_model.predict(df=test_df_for_prediction)
    
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
        'Model': 'DLT',
        'Trend': best_config['trend'],
        'Estimator': best_config['estimator'],
        'MAE': mae,
        'MAPE': mape_value,
        'RMSE': rmse,
        'MSE': mse
    }

    # Output path to save the results
    base_path = os.path.join(DATA_PATH, 'ORBIT_ML', 'DLT_Result', periodicity)
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


# Example function call
def bayesian_orbit():
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

        # Process complete data
        print(f"> Processing {project} for complete data")
        method_complete = trigger_prediction(
            df_path=os.path.join(complete_data_path, complete_files[i]),
            project_name=project,
            periodicity="complete",
            seasonality=None
        )