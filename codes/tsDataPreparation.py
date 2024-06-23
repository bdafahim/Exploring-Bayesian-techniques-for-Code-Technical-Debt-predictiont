import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from commons import DATA_PATH

# CONSTANTS
#MULTIVARIATE_PATH = "C:/OULU/TECHDEBT2023/data/multivariate_data.csv"
#FINAL_FORMAL_PATH = "C:/OULU/TECHDEBT2023/data/final_overall_df.csv"
#COMPLETE_PATH = "C:/OULU/TECHDEBT2023/data/complete_data"
#BIWEEKLY_DIR = "C:/OULU/TECHDEBT2023/data/biweekly_data"
#MONTHLY_DIR = "C:/OULU/TECHDEBT2023/data/monthly_data"


def varianceThreshold(MULTIVARIATE_PATH):
    # 1. VARIANCE THRESOLDING (If a column has low variance, it means it doesn't change
    # much across different projects and could potentially be removed.)
    df = pd.read_csv(MULTIVARIATE_PATH)
    variances = df.iloc[:, 4:].var()

    variances_4qr = variances[variances >= variances.describe()['75%']]
    # We could set a thresold ourselves, and then if after removing the chosen columns,
    # some of the projects result having few significant columns, we can discard there
    # projects.

    return variances_4qr


def zeroPercentage(MULTIVARIATE_PATH):

    # 2. PERCENTAGE OF NON-ZERO VALUES (Remove columns with high percentage of zeros)
    df = pd.read_csv(MULTIVARIATE_PATH)
    percentage_nonzeros = (df.iloc[:, 4:][df.iloc[:, 4:] != 0].count(axis=0)/len(df) * 100)
    percentage_nonzeros_4qr = percentage_nonzeros[percentage_nonzeros >= percentage_nonzeros.describe()['75%']]

    return percentage_nonzeros_4qr


def featureImportance(MULTIVARIATE_PATH):

    # 3. IMPLEMENTING A RANDOM FOREST FOR FEATURE IMPORTANCE
    df = pd.read_csv(MULTIVARIATE_PATH)
    X = df.drop(columns=df.iloc[:, :4])
    Y = df['SQALE_INDEX']

    rf_model = RandomForestRegressor(n_estimators=500, criterion="squared_error", random_state=42)
    rf_model.fit(X, Y)

    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({"Predictor": X.columns, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fourth_quartile = importance_df['Importance'].quantile(0.75)
    importance_4qr = importance_df[importance_df['Importance'] > fourth_quartile]

    return importance_4qr


def correlationAnalysis(MULTIVARIATE_PATH):

    # 4. PERFORMING A CORRELATION ANALYSIS BETWEEN EACH PREDICTOR AND THE RESPONSE VARIABLE
    df = pd.read_csv(MULTIVARIATE_PATH)
    data = df.drop(columns=df.iloc[:, [0, 2, 3]])
    correlation_matrix = data.corr()
    corr_sqale = correlation_matrix['SQALE_INDEX'].sort_values(ascending=False)
    corr_sqale = corr_sqale[1:]  # Without the SQALE_INDEX value itself
    corr_sqale_4qr = corr_sqale[corr_sqale >= corr_sqale.describe()['75%']]

    return corr_sqale_4qr


def cropPredictors(df_path, survival_predictors):
    """
    Based on the chosen preprocessing techniques, the original commit dataframes get part of their
    observations cropped, and only the significant set of predictors moves forward with the
    last version of the project data frame.

    :param df_path: Path for the initial version of the dataframe
    :param survival_predictors: Predictors that showed importance after preprocessing step.
    :return: Newly formatted data frames and the names of the projects.
    """

    # Since some observations have only two years of activity, there are two possible approaches
    # (1: Performing weekly series and observing whether it gives reasonable answers.
    # 2: Performing monthly series and excluding projects offering low amount of data.)
    input_df = pd.read_csv(df_path)

    survival_cols = input_df[input_df.columns.intersection(survival_predictors)]
    subset_df = pd.concat([input_df[['PROJECT', 'SQALE_INDEX', 'COMMIT', 'COMMIT_DATE']], survival_cols], axis=1)
    subset_df.to_csv(os.path.join(DATA_PATH, "final_overall_df.csv"), index=False)

    grouped_df = subset_df.groupby('PROJECT')  # Group values by projects.
    project_dataframes = [group for _, group in grouped_df]
    project_names = []
    for project_df in project_dataframes:
        project_names.append(project_df['PROJECT'].unique()[0])
        del project_df['PROJECT'], project_df['COMMIT']

    for project_name, pro_df in zip(project_names, project_dataframes):
        clean_name = project_name.split(':')[-1]
        pro_df.to_csv(os.path.join(os.path.join(DATA_PATH, 'complete_data'), clean_name + ".csv"))

    return project_dataframes, project_names


def nearest_observation(df, timeframe):
    """
    Help function for the time series creation, it considers the creation of empty rows
    for cases in which there is no close existing observation to the defined time period.

    :param df: Current df consisting in Sqale index and code smell
    violated rules for a specific project
    :param timeframe: Considered timeframe for the time series data.
    :return: out_df: Contains observations time ordered.
    """

    # Set the variable type of the commit dates as datetime for the calculations
    df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
    if timeframe == "BW":  # Bi-weekly
        time_duration = timedelta(days=14)
    else:  # Monthly
        time_duration = timedelta(days=30)

    # Setting the first observation of the timeframe
    out_df = pd.DataFrame(columns=list(df.columns))
    out_df['COMMIT_DATE'] = pd.to_datetime(out_df['COMMIT_DATE'])
    out_df.loc[0, :] = df.iloc[0, :]

    for idx in df.index:

        last_date = pd.to_datetime(out_df.loc[len(out_df)-1, 'COMMIT_DATE'])
        time_lag = df['COMMIT_DATE'][idx] - last_date
        # The next commit surpasses the predefined timeframe
        if time_lag > time_duration:
            floor = time_lag // time_duration
            if floor == 1:  # If the lag is exactly 7 days or less than 14
                out_df.loc[len(out_df.index)] = df.loc[idx]
            elif floor >= 2:  # The lag is more than 1 week ahead from the last observation.
                for weeks in range(1, floor):  # The last additional point will be filled with the actual value of the original df.
                    empty_row = pd.Series({}, dtype=object)
                    out_df.loc[len(out_df.index), 'COMMIT_DATE'] = last_date + time_duration*weeks

                out_df.loc[len(out_df.index)] = df.loc[idx]
        else:
            continue

    return out_df


def tsCreation(input_dfs):
    """
    Time series creation from the original commit data of the selected projects. This
    function considers biweekly and monthly timeframes for the TS creation and orders the
    data based on the commit dates, obtaining the closest ones to the corresponding date.

    :param input_dfs: Initial project grouped dataframes with the unordered observations
    :return: dataframes time ordered bi-weekly and monthly
    """

    # Bi-weekly data
    biweekly_dataframes = []
    for project_df in input_dfs:
        # Gets the closest datapoint to the weekly timeframe.
        project_w_df = nearest_observation(df=project_df, timeframe='BW')
        biweekly_dataframes.append(project_w_df)

    # Monthly data
    monthly_dataframes = []
    for project_df in input_dfs:
        project_m_df = nearest_observation(df=project_df, timeframe='M')
        monthly_dataframes.append(project_m_df)

    return biweekly_dataframes, monthly_dataframes


def interpolator(biweekly_dfs, monthly_dfs, project_names):
    """
    Interpolates missing data values for missing observation periods through the
    'Linear interpolation techniques'.

    :param biweekly_dfs: project dataframes ordered in a biweekly basis
    :param monthly_dfs: project dataframes ordered in a monthly basis
    :param project_names: names of the TDD projects
    :return: -
    """

    biweekly_ts = []
    monthly_ts = []
    for project_name, biweekly_data, monthly_data in zip(project_names, biweekly_dfs, monthly_dfs):

        biweekly_data['COMMIT_DATE'] = pd.to_datetime(biweekly_data['COMMIT_DATE'])
        monthly_data['COMMIT_DATE'] = pd.to_datetime(monthly_data['COMMIT_DATE'])

        biweekly_data = biweekly_data.set_index('COMMIT_DATE')
        monthly_data = monthly_data.set_index('COMMIT_DATE')

        print(f"> Length of series for bi-weekly data for project {project_name}: {len(biweekly_data)}")
        print(f"> Length of series for monthly data for project {project_name}: {len(monthly_data)}")
        print(f"---------------------------------------------------------------------")

        biweekly_ts.append(biweekly_data)
        monthly_ts.append(monthly_data)

        with open(r'/Users/badruddduzaahmed/Documents/Oulu/Thesis/replication-package/data/project_names.txt', 'w') as fp:
            for item in project_names:
                # write each item on a new line
                fp.write("%s\n" % item)
        fp.close()

        biweekly_data = biweekly_data.astype(np.number)
        monthly_data = monthly_data.astype(np.number)
        biweekly_data = biweekly_data.interpolate(method='linear', order=2, limit_direction='both')
        monthly_data = monthly_data.interpolate(method='linear', order=2, limit_direction='both')

        proname_clean = project_name.split(':')[-1]
        biweekly_data.to_csv(os.path.join(DATA_PATH, 'biweekly_data', f"{proname_clean}.csv"))
        monthly_data.to_csv(os.path.join(DATA_PATH, 'monthly_data', f"{proname_clean}.csv"))


def data_prepare():

    var_df = varianceThreshold(os.path.join(DATA_PATH, 'raw-data', 'multivariate_data.csv'))
    zer_perc_df = zeroPercentage(os.path.join(DATA_PATH, 'raw-data', 'multivariate_data.csv'))
    feature_imp = featureImportance(os.path.join(DATA_PATH, 'raw-data', 'multivariate_data.csv'))
    correlation = correlationAnalysis(os.path.join(DATA_PATH, 'raw-data', 'multivariate_data.csv'))

    print(f"Length of Variance Thresolding predictors: {len(var_df)}")
    print(f"Length of Zero Percentage predictors: {len(zer_perc_df)}")
    print(f"Length of Random Forest predictors: {len(feature_imp)}")
    print(f"Length of Correlation Analysis predictors: {len(correlation)}")

    # Getting the set of common predictors
    common_predictors = list(set(var_df.keys()) &
                             set(zer_perc_df.keys()) &
                             set(feature_imp['Predictor']) &
                             set(correlation.keys()))

    print(f"Number of common predictors: {len(common_predictors)}")
    print(common_predictors)

    # Cropping the set of keys from the initial dataframes
    project_dataframes, project_names = cropPredictors(df_path=os.path.join(DATA_PATH, 'raw-data', 'multivariate_data.csv'),
                                                       survival_predictors=common_predictors)

    # Time series creation in the preconsidered timeframe types
    biweekly_dfs, monthly_dfs = tsCreation(input_dfs=project_dataframes)

    # Interpolation of missing data due to NA time gaps in the project time series.
    interpolator(biweekly_dfs, monthly_dfs, project_names)
    
