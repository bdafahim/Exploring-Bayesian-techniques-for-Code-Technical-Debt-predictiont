import pandas as pd
from datetime import datetime
import numpy as np
from commons import DATA_PATH
import os

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def preprocessing():
    # Get the imported raw datasets from SQL into a pandas dataframe
    commits_df = pd.read_csv(os.path.join(DATA_PATH, "raw-data", 'commits.csv'))
    issues_df = pd.read_csv(os.path.join(DATA_PATH, "raw-data", 'issues.csv'))

    rules_list = list(issues_df['RULE'].unique())
    header_cols = ["PROJECT", 'SQALE_INDEX', "COMMIT", "COMMIT_DATE"] + rules_list
    df = pd.DataFrame(columns=header_cols)
    df['PROJECT'] = commits_df['PROJECT_ID']
    df['COMMIT'] = commits_df['REVISION']
    df['COMMIT_DATE'] = commits_df['COMMIT_DATE']
    df['SQALE_INDEX'] = commits_df['SQALE_INDEX']
    df.iloc[:, 4:] = 0
    print("> final dataframe structured. Starting preprocessing...")

    for index_i in range(len(issues_df)):

        project_id = issues_df.iloc[index_i, 0]
        rule = issues_df.iloc[index_i, 2]
        creation_date = datetime.strptime(issues_df.iloc[index_i, -2], DATE_FORMAT)

        if issues_df.iloc[index_i, -1] is not np.nan:
            close_date = datetime.strptime(issues_df.iloc[index_i, -1], DATE_FORMAT)
        else:
            close_date = np.nan

        for index_c in range(len(commits_df)):  

            # Get the row index of the commit in the df:
            idx = df.index[df["COMMIT"] == commits_df.iloc[index_c, 1]]

            # The project must be the same within the comparison
            if commits_df.iloc[index_c, 0] != project_id:
                continue

            if close_date is np.nan:

                if creation_date <= datetime.strptime(commits_df.iloc[index_c, 2], DATE_FORMAT):
                    df.loc[idx, rule] += 1

            else:

                if creation_date <= datetime.strptime(commits_df.iloc[index_c, 2], DATE_FORMAT) <= close_date:
                    df.loc[idx, rule] += 1

        print("{}\{} rows processed from the issue data".format(index_i, len(issues_df)))

        if index_i == 5:
            break

    # Store the final dataframe
    df.to_csv(os.path.join(DATA_PATH, "raw-data", "multivariate_data.csv"), index=False)
    print("> final dataframe COMPLETED.")
    print("> {} rows".format(len(df)))