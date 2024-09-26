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
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests
import json



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
    
    return important_features, 'L1'

# Function to perform Recursive Feature Elimination (RFE) for feature selection
def select_features_with_rfe(X, y, num_features):
    # Use a basic linear model to perform RFE
    model = LinearRegression()
    
    # Perform RFE to select the top 'num_features' most important features
    rfe = RFE(model, n_features_to_select=num_features)
    rfe.fit(X, y)
    
    # Select features with non-zero importance
    selected_features = X.columns[rfe.support_]
    
    print(f"Selected Features using RFE: {selected_features.tolist()}")
    
    return selected_features, 'RFE'

# Bayesian Model Averaging (BMA) for Feature Selection
def select_features_with_bma(X, y, num_features=None):
    # Bayesian Ridge as a proxy for BMA-like behavior
    model = BayesianRidge()
    
    # Fit the model
    model.fit(X, y)
    
    # Extract feature coefficients (weights)
    coefficients = np.abs(model.coef_)
    
    # Sort features by their absolute coefficient values (important for selection)
    feature_importance = np.argsort(-coefficients)
    
    # Select top 'num_features' based on coefficients, or all non-zero coefficients if num_features is not provided
    if num_features is not None:
        selected_indices = feature_importance[:num_features]
    else:
        selected_indices = feature_importance[coefficients > 0]
    
    # Select the feature names
    selected_features = X.columns[selected_indices]
    
    print(f"Selected Features using BMA: {selected_features.tolist()}")
    
    return selected_features, 'BMA'


# XGBoost for Feature Selection
def select_features_with_xgboost(X, y, num_features=None):
    # Fit an XGBoost regressor
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=8888)
    model.fit(X, y)
    
    # Extract feature importance
    feature_importances = model.feature_importances_
    
    # Rank features based on importance
    feature_indices = np.argsort(-feature_importances)
    
    # Select the top 'num_features' if specified, or all non-zero features otherwise
    if num_features is not None:
        selected_indices = feature_indices[:num_features]
    else:
        selected_indices = feature_indices[feature_importances > 0]
    
    # Select the feature names
    selected_features = X.columns[selected_indices]
    
    print(f"Selected Features using XGBoost: {selected_features.tolist()}")
    
    return selected_features, 'XGBoost'

# Random Forest for Feature Selection
def select_features_with_random_forest(X, y, num_features=None):
    # Fit a RandomForest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=8888)
    model.fit(X, y)
    
    # Extract feature importance
    feature_importances = model.feature_importances_
    
    # Rank features based on importance
    feature_indices = np.argsort(-feature_importances)
    
    # Select the top 'num_features' if specified, or all non-zero features otherwise
    if num_features is not None:
        selected_indices = feature_indices[:num_features]
    else:
        selected_indices = feature_indices[feature_importances > 0]
    
    # Select the feature names
    selected_features = X.columns[selected_indices]
    
    print(f"Selected Features using Random Forest: {selected_features.tolist()}")
    
    return selected_features, 'RandomForest'


def remove_unwanted_features(features, unwanted=['Unnamed: 0']):
    """
    Removes unwanted features from the list of selected features.
    
    Args:
        features: List of selected features.
        unwanted: List of unwanted feature names to remove.
        
    Returns:
        List of features with unwanted features removed.
    """
    cleaned_features = [feature for feature in features if feature not in unwanted]
    print(f"Cleaned Features: {cleaned_features}")
    return cleaned_features

# Define the hypertuning function for DLT model
def hypertune_dlt_model(training_df, y_train, x_train, y_test, testing_df, seasonality):


    # Perform feature selection using Lasso
    important_features, name = select_features_with_lasso(x_train, y_train)

    # Perform feature selection using RFE
    #important_features, name = select_features_with_rfe(x_train, y_train, num_features=10)

    # Perform feature selection using BMA
    #important_features, name = select_features_with_bma(x_train, y_train)

    # Perform feature selection using XGBoost
    #important_features, name = select_features_with_xgboost(x_train, y_train)

    # Perform feature selection using Random Forest
    #important_features, name = select_features_with_random_forest(x_train, y_train)

    # Remove unwanted features like 'Unnamed: 0'
    important_features = remove_unwanted_features(important_features)



    # Define the hyperparameter grid (without penalties)
    trend_options = ['linear', 'loglinear', 'flat', 'logistic']
    estimators = ['stan-map', 'stan-mcmc']

    best_mae = float('inf')
    best_model = None
    best_config = None

    # Iterate over each combination of trend and estimator
    for trend in trend_options:
        for estimator in estimators:
            # Define the model with the current set of hyperparameters
            print(f"Training with trend={trend}, estimator={estimator}")
            
            model = DLT(
                seasonality=seasonality,
                response_col='SQALE_INDEX',
                date_col='COMMIT_DATE',
                estimator=estimator,
                global_trend_option=trend,
                seed=8888,
                regressor_col=important_features,
                n_bootstrap_draws=1000
            )
            
            # Fit the model
            model.fit(df=training_df)
            
            # Predict and calculate MAE
            predicted_df = model.predict(df=testing_df)
            predicted = predicted_df['prediction'].values
            
            mae = mean_absolute_error(y_test, predicted)
            
            print(f"MAE for trend={trend}, estimator={estimator}: {mae:.2f}")
            
            # Check if the current model is better
            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_config = {
                    'trend': trend,
                    'estimator': estimator
                }

    print(f"Best configuration: {best_config} with MAE: {best_mae:.2f}")
    
    # Return the best model and configuration
    return best_model, best_config, important_features, name


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
    best_model, best_config, important_features, name = hypertune_dlt_model(
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
    base_path = os.path.join(DATA_PATH, f"ORBIT_ML_{name}", 'DLT_Result', periodicity)
    os.makedirs(base_path, exist_ok=True)
    csv_output_path = os.path.join(base_path, "assessment.csv")

    # Save result_df as a CSV file
    results_df = pd.DataFrame([result_data])
    if not os.path.isfile(csv_output_path):
        results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
    else:
        results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

    # Save important features to JSON
    features_base_path = os.path.join(base_path, 'best_regressors')
    os.makedirs(features_base_path, exist_ok=True)
    features_output_path = os.path.join(features_base_path, f"{project_name}.json")
    features_data = {
        "Project": project_name,
        "best_regressors": important_features  # Convert important features to list for JSON format
    }
    
    with open(features_output_path, 'w') as json_file:
        json.dump(features_data, json_file, indent=4)  # Save the important features in JSON format with indentation

    print(f"Results for {project_name} saved in {base_path}")
    print(f"Important features saved in {features_output_path}")
    return result_data


# Example function call
def bayesian_orbit():
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