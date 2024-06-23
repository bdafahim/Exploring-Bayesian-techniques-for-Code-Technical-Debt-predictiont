"""
Combines the needed results into aggregated values for global comparison
"""

from commons import DATA_PATH, assessment_statistics, final_table_columns, FINAL_TS_TABLE_COLS
from modules import transform_to_latex

import pandas as pd
import os
from statistics import mean


def check_results(periodicity_level):
    """
    Checks the format of the results obtain for each model in the provided periodicity level.
    NOTE: Only works for TS approaches
    :param periodicity_level: Given periodicity to examine
    """

    result_directories = ['sarimax_results', 'arimax_results', 'sarima_lm_results',
                          'arima_lm_results']

    for result_directory in result_directories:

        if not os.path.exists(os.path.join(DATA_PATH, f"{result_directory}_clean")):
            os.mkdir(os.path.join(DATA_PATH, f"{result_directory}_clean"))
            os.mkdir(os.path.join(DATA_PATH, f"{result_directory}_clean", "biweekly_results"))  # biweekly
            os.mkdir(os.path.join(DATA_PATH, f"{result_directory}_clean", "monthly_results"))  # Monthly

        # Check detected errors in output files
        project_list = os.listdir(os.path.join(DATA_PATH, result_directory, f"{periodicity_level}_results"))
        periodicity_results_dir = os.path.join(DATA_PATH, result_directory, f"{periodicity_level}_results")

        # Generate clean output files
        clean_output_path = os.path.join(DATA_PATH, f"{result_directory}_clean", f"{periodicity_level}_results")
        clean_dict = {col: [] for col in FINAL_TS_TABLE_COLS}

        for i in range(len(project_list)):
            project_name = project_list[i][:-4]
            project_path = os.path.join(periodicity_results_dir, project_list[i])
            df = pd.read_csv(project_path)
            results_cols = df.columns.tolist()

            # Check if the results file has the correct column names (due to unknown reasons some files got the names
            # inside the results)
            if results_cols != FINAL_TS_TABLE_COLS:
                df.columns = FINAL_TS_TABLE_COLS

            # Check length of the df, should be one project per csv, the correct df should work with this logic too.
            model_name_col = df['PROJECT']
            model_idx = list(model_name_col).index(project_name)
            output_row = df.loc[model_idx, :]
            for j in range(len(FINAL_TS_TABLE_COLS)):
                clean_dict[FINAL_TS_TABLE_COLS[j]].append(output_row[j])

            print("> Project [{}] cleaned for model results [{}] - {}/{}".format(project_name,
                                                                                 result_directory,
                                                                                 i+1, len(project_list)))

        # We merge all the outputs
        if not os.path.exists(os.path.join(DATA_PATH, f"{result_directory}_clean")):
            os.mkdir(os.path.join(DATA_PATH, f"{result_directory}_clean"))

        output_df = pd.DataFrame.from_dict(clean_dict)
        output_df.to_csv(os.path.join(clean_output_path, 'assessment.csv'), mode='w', index=False)


def merge_ml(periodicity, missing_values):
    """
    Merges the results from the completed
    """

    final_format_dict = {col: [] for col in final_table_columns}

    ml_results_path = os.path.join(DATA_PATH, 'results', 'results_ML_sarimax')
    models_list = os.listdir(ml_results_path)

    # Parse and collect all the assessment files of each of the ML models with the results of each project. Get the mean
    for i in range(len(models_list)):
        model = models_list[i]
        model_path = os.path.join(ml_results_path, model, periodicity)
        assessment_df = pd.read_csv(os.path.join(model_path, 'assessment.csv'))

        # NOTE: We filter out the rows of the projects which couldn't get calculated with the time series approach
        assessment_df = assessment_df[~assessment_df['PROJECT'].isin(missing_values)]

        final_format_dict['Approach'].append(model)
        if periodicity == 'complete':
            if model == 'svr' or model == 'xgb' or model == 'rf':
                final_format_dict['Type'].append('NL')
            else:
                final_format_dict['Type'].append('L')
        else:
            final_format_dict['Type'].append('~TD')
        final_format_dict['MAPE'].append(round(assessment_df['MAPE'].mean(), 2))
        final_format_dict['MAE'].append(round(assessment_df['MAE'].mean(), 2))
        final_format_dict['MSE'].append(round(assessment_df['MSE'].mean(), 2))
        final_format_dict['RMSE'].append(round(assessment_df['RMSE'].mean(), 2))

        print("> Model [{}] results merged - {}/{}".format(model, i+1, len(models_list)))

    if periodicity == 'complete':
        # Convert into a df
        return pd.DataFrame.from_dict(final_format_dict)
    else:
        return final_format_dict


def save_results(latex, df, file_name):
    """
    Stores the results into csv and in case in latex format
    """

    # 1. Store into csv format
    df.to_csv(os.path.join(DATA_PATH, 'final_results', file_name), index=False)
    # 2. Store into latex format
    if latex:
        transform_to_latex(os.path.join(DATA_PATH, 'final_results', file_name))


def merge_all(periodicity_level, missing_values):
    """
    Merges the biweekly and monthly results from all the implemented models in the concerning project
    """

    result_directories = ['sarimax_results_clean', 'arimax_results_clean', 'sarima_lm_results_clean',
                          'arima_lm_results_clean', 'results/results_ML_sarimax']
    final_results_dict = {col: [] for col in final_table_columns}

    for j in range(len(result_directories)):
        result_directory = result_directories[j]

        # If the results have been performed by non TD models.
        if result_directory is 'results/results_ML_sarimax':

            # For the given periodicity, check for the approach the obtained results pero project
            prov_df_dict = merge_ml(periodicity=periodicity_level, missing_values=missing_values)

            models = prov_df_dict['Approach']

            for i in range(len(models)):

                # Store the avg values in the final results dict.
                final_results_dict['Approach'].append(models[i])
                final_results_dict['Type'].append('~TD')
                final_results_dict['MAPE'].append(prov_df_dict['MAPE'][i])
                final_results_dict['MAE'].append(prov_df_dict['MAE'][i])
                final_results_dict['MSE'].append(prov_df_dict['MSE'][i])
                final_results_dict['RMSE'].append(prov_df_dict['RMSE'][i])

        else:

            # For the given periodicity, check for the approach the obtained results per project
            project_results_dir_path = os.path.join(DATA_PATH, result_directory, f"{periodicity_level}_results")
            assessment_df = pd.read_csv(os.path.join(project_results_dir_path, "assessment.csv"))

            # NOTE: We filter out the rows of the projects which couldn't get calculated with the time series approach
            assessment_df = assessment_df[~assessment_df['PROJECT'].isin(missing_values)]

            # Store the avg values in the final results dict.
            if result_directory is "sarimax_results_clean" or result_directory is "sarima_lm_results_clean":
                final_results_dict['Type'].append('STD')
            else:
                final_results_dict['Type'].append('TD')

            final_results_dict['Approach'].append(result_directory[:-6])
            final_results_dict['MAPE'].append(round(assessment_df['MAPE'].mean(), 2))
            final_results_dict['MAE'].append(round(assessment_df['MAE'].mean(), 2))
            final_results_dict['MSE'].append(round(assessment_df['MSE'].mean(), 2))
            final_results_dict['RMSE'].append(round(assessment_df['RMSE'].mean(), 2))

        print(f"> Final results file generated for approach {result_directory} - {j+1}/{len(result_directories)}")

    return pd.DataFrame.from_dict(final_results_dict)


def check_missing_results(periodicity_level):
    """
    Checks the existing missing values from errors in the calculations of the TS models
    """

    result_directories = ['sarimax_results_clean', 'arimax_results_clean', 'sarima_lm_results_clean',
                          'arima_lm_results_clean']

    missing_val_list = []

    for result_directory in result_directories:
        result_path = os.path.join(DATA_PATH, result_directory, f"{periodicity_level}_results")
        results_df = pd.read_csv(os.path.join(result_path, 'assessment.csv'))
        null_mask = results_df.isnull().any(axis=1)
        null_rows = results_df[null_mask]
        missing_val_list.extend(list(null_rows['PROJECT']))

    missing_val_list = list(set(missing_val_list))  # Get only unique values
    return missing_val_list


def merge_results(periodicity_level, missing_values):
    """
    Merges the results of the used models into the aggregated values for global comparison anchored to the according
    periodicity level
    """

    if periodicity_level == "complete":  # Meaning that the data has only been processed by ML models.

        # We deal with the complete data separately as we only need the averages of the ML models
        # ALso, we do not consider missing values in this case as there is no TS model to compare
        results_complete_df = merge_ml(periodicity=periodicity_level, missing_values=[])
        save_results(latex=True, df=results_complete_df, file_name="complete_data_results.csv")

    else:

        results_df = merge_all(periodicity_level=periodicity_level, missing_values=missing_values)
        save_results(latex=True, df=results_df, file_name=f"{periodicity_level}_data_results.csv")


def combine_results():
    """
    Combines the obtained results from each of the models into the same file for comparison purposes
    """

    periodicity_levels = ['biweekly', 'monthly', 'complete']
    if not os.path.exists(os.path.join(DATA_PATH, 'final_results')):
        os.mkdir(os.path.join(DATA_PATH, 'final_results'))

    overall_missing_result_pros = []
    # merge_outcomes into biweekly data, monthly and complete data
    for periodicity in periodicity_levels:
        print(f"> Processing periodicity level = {periodicity}")

        if not periodicity == 'complete':
            check_results(periodicity_level=periodicity)  # Check missing values
            # For the calculations we only consider the non-missing projects
            missing_projects = check_missing_results(periodicity_level=periodicity)
            overall_missing_result_pros.extend(missing_projects)

        # Merge all the results into combined result df
        merge_results(periodicity_level=periodicity, missing_values=missing_projects)

    # Final check to know which projects have been considered in the biweekly and monthly model comparison
    list_projects = os.listdir(os.path.join(DATA_PATH, 'biweekly_data'))
    pros_considered_in_results = [list_project[:-4] for list_project in list_projects if list_project[:-4]
                                  not in overall_missing_result_pros]

    final_path = os.path.join(DATA_PATH, 'final_results', 'resulting_projects.txt')
    with open(final_path, 'w') as file:
        file.write('CONSIDERED PROJECTS IN THE RESULTS:\n')
        # Join the list elements into a single string with a newline character
        data_to_write = '\n'.join(pros_considered_in_results)
        # Write the data to the file
        file.write(data_to_write)
        file.write('\nNON CONSIDERED PROJECTS:\n')
        more_to_write = '\n'.join(overall_missing_result_pros)
        file.write(more_to_write)
        file.close()

