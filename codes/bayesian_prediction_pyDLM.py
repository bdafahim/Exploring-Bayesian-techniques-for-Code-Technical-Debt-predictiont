import os
import pandas as pd
import numpy as np
from commons import DATA_PATH
from modules import check_encoding
import matplotlib.pyplot as plt
import logging
from modules import MAPE, RMSE, MAE, MSE
from pydlm import dlm, trend, seasonality, dynamic
from sklearn.preprocessing import StandardScaler




logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)




def trigger_prediction(df_path, project_name, periodicity):
    try:
        encoding = check_encoding(df_path)
        df = pd.read_csv(df_path, encoding=encoding)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df.set_index('COMMIT_DATE', inplace=True)
        df = df.dropna()

        # Check for missing values in each column
        missing_values = df.isnull().sum()
        print(f'Missing values in project{project_name}')
        print(missing_values[missing_values > 0])  # This will print columns with missing values and their count

        # Splitting data into training (80%) and testing (20%)
        split_point = round(len(df) * 0.8)
        training_df = df.iloc[:split_point, :]
        testing_df = df.iloc[split_point:, :]

        #Dependent and independent variables
        y_train = training_df['SQALE_INDEX'].values
        x_train = training_df.drop(columns=['SQALE_INDEX']).values
        y_test = testing_df['SQALE_INDEX'].values
        x_test = testing_df.drop(columns=['SQALE_INDEX']).values

        # Scaling the exogenous variables
        '''scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)'''

        # Convert x_train and x_test to a list of lists format as required by pyDLM
        x_train = x_train.tolist()
        x_test = x_test.tolist()

        # Prepare the featureDict for prediction using the x_test DataFrame
        featureDict = {'exogenous_features': x_test}


        # Create the DLM model with a trend component
        dlm_model = dlm(y_train) + trend(degree=1, discount=0.98, name='linear_trend')

        # Add seasonality based on the periodicity
        if periodicity == 'biweekly':
            dlm_model = dlm_model + seasonality(period=26, discount=0.98, name='biweekly_seasonality')
        elif periodicity == 'monthly':
            dlm_model = dlm_model + seasonality(period=12, discount=0.98, name='monthly_seasonality')

        
        # Add the exogenous variables as a dynamic component
        dlm_model = dlm_model + dynamic(features=x_train, discount=0.98, name='exogenous_features')

        # Fit the model
        dlm_model.fit()

        # Forecast for the next steps (same length as the testing data)
        forecast_len = len(y_test)
        (predicted_mean, predicted_var) = dlm_model.predictN(forecast_len, featureDict=featureDict)

        # Evaluate the model's performance using the provided metrics
        mae = round(MAE(y_test, predicted_mean), 2)
        mape = round(MAPE(y_test, predicted_mean), 2)
        mse = round(MSE(y_test, predicted_mean), 2)
        rmse = round(RMSE(y_test, predicted_mean), 2)

        # Logging the results
        logger.info(f"Project: {project_name}, Periodicity: {periodicity}")
        logger.info(f"MAE: {mae}, MAPE: {mape}, MSE: {mse}, RMSE: {rmse}")

        # Store the results in a dictionary
        result_data = {
            'Project': project_name,
            'Model': 'DLM',
            'Trend': 'Linear',
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse
        }

        base_path = os.path.join(DATA_PATH, 'pyDLM', f'results_{periodicity}.csv')
        os.makedirs(base_path, exist_ok=True)
        csv_output_path = os.path.join(base_path, "assessment.csv")

        # Save result_df as a CSV file
        results_df = pd.DataFrame([result_data])
        if not os.path.isfile(csv_output_path):
            results_df.to_csv(csv_output_path, mode='w', index=False, header=True)
        else:
            results_df.to_csv(csv_output_path, mode='a', index=False, header=False)

        print(f"Results for {project_name} saved in {base_path}")
        

    except Exception as e:
        logger.error(f"Error processing {project_name}: {str(e)}")
        return None

def bayesian_pyDLM():
    # Define paths for biweekly and monthly data
    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    complete_data_path = os.path.join(DATA_PATH, "complete_data")

    # List existing data files
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)
    complete_files = os.listdir(complete_data_path)

    for i in range(len(biweekly_files)):
        if biweekly_files[i] == '.DS_Store':
            continue
        project = biweekly_files[i][:-4]

        logger.info(f"Processing {project}")

        # Process biweekly data
        logger.info(f"> Processing {project} for biweekly data")
        pybats_dglm_biweekly = trigger_prediction(
            df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
            project_name=project,
            periodicity="biweekly",
        )

    for i in range(len(monthly_files)):
        if monthly_files[i] == '.DS_Store':
            continue
        project = monthly_files[i][:-4]

        logger.info(f"Processing {project}")

        # Process monthly data
        logger.info(f"> Processing {project} for monthly data")
        pybats_dglm_monthly = trigger_prediction(
            df_path=os.path.join(monthly_data_path, monthly_files[i]),
            project_name=project,
            periodicity="monthly",
        )

    for i in range(len(complete_files)):
        if complete_files[i] == '.DS_Store':
            continue
        project = complete_files[i][:-4]

        logger.info(f"Processing {project}")

        # Process complete data
        logger.info(f"> Processing {project} for complete data")
        pybats_dglm_complete = trigger_prediction(
            df_path=os.path.join(complete_data_path, complete_files[i]),
            project_name=project,
            periodicity="complete",
        )
        

    logger.info("> pyDLM prediction stage performed!")
