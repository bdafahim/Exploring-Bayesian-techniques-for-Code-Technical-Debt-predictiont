
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE
import json
import logging
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
from numpy.linalg import inv
from itertools import islice
import scipy.stats as ss
from abc import ABC, abstractmethod
import json
from scipy.stats import anderson





logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Define the BaseLikelihood class
class BaseLikelihood(ABC):
    """
    Abstract base class to define the structure for likelihood functions.
    Future classes inheriting this must implement the abstract methods.
    """
    @abstractmethod
    def pdf(self, data: np.array):
        raise NotImplementedError("PDF is not defined. Please define in derived class.")

    @abstractmethod
    def update_theta(self, data: np.array, **kwargs):
        raise NotImplementedError("Update theta is not defined. Please define in derived class.")

# Apply Anderson-Darling Test for Normality
def check_normality(data, column_name):
    result = anderson(data)
    print(f"AD test statistic for {column_name}: {result.statistic}")
    print("Critical values:", result.critical_values)
    print("Significance levels:", result.significance_level)
# Main method to trigger the change point detection process
def trigger_changepoint_detection(df_path, project_name, periodicity=None):
    try:
        # Read the data from CSV
        df = pd.read_csv(df_path)
        df['COMMIT_DATE'] = pd.to_datetime(df['COMMIT_DATE'])
        df.set_index('COMMIT_DATE', inplace=True)
        df = df.dropna()

        # Check normality on original data
        print(f"Normality Check on Original Data ({project_name}):")
        check_normality(df['SQALE_INDEX'].dropna(), 'SQALE_INDEX')
        for feature in df.columns.drop('SQALE_INDEX'):
            check_normality(df[feature].dropna(), feature)
        
        # Log Transformation
        df_transformed = df.apply(lambda x: np.log1p(x) if (x > 0).all() else x)

        # Re-check normality after log transformation
        print(f"\nNormality Check on Log-Transformed Data ({project_name}):")
        check_normality(df_transformed['SQALE_INDEX'].dropna(), 'SQALE_INDEX (Log Transformed)')
        for feature in df_transformed.columns.drop('SQALE_INDEX'):
            check_normality(df_transformed[feature].dropna(), feature)

        return

        # Get dependent variables (features) and the target variable (SQALE_INDEX)
        features = df_transformed.drop(columns=["SQALE_INDEX"])
        target = df_transformed["SQALE_INDEX"].values

        # Instantiate the Bayesian Change Point detection model using Multivariate T distribution
        detector = MultivariateT(dims=features.shape[1])
         # Store the results in a JSON file
        base_path = os.path.join(DATA_PATH, 'Changepoint_Result', periodicity)
        os.makedirs(base_path, exist_ok=True)
        plot_path = os.path.join(base_path, 'Changepoint_Plots')
        os.makedirs(plot_path, exist_ok=True)


        # Iterate through each time step
        change_points = []
        for t in range(len(target)):
            # Update the detector with the current observation
            detector.update_theta(data=features.iloc[t].values)

            # Get the posterior probability of change point at this time step
            prob = detector.pdf(features.iloc[t].values)
            
            # Assume a threshold to detect changepoint (This can be tuned)
            if prob.max() < 0.01:
                change_points.append(df.index[t])
        
        # Plot the changepoints on the original data
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, target, label="SQALE_INDEX")
        plt.scatter(change_points, [target[df.index.get_loc(cp)] for cp in change_points], color='red', label="Change Points")
        plt.title(f"Change Point Detection for {project_name} ({periodicity})")
        plt.xlabel("Date")
        plt.ylabel("SQALE_INDEX")
        plt.legend()
        plt.savefig(os.path.join(plot_path, f"{project_name}.png"))
        #plt.show()

        output_path = os.path.join(base_path, 'Bayesian_Changepoints_JSON', f"{project_name}_changepoints.json")
        os.makedirs(output_path, exist_ok=True)

        change_points_data = []

        for cp in change_points:
            # Find the corresponding SQALE_INDEX value for the change point date
            sqale_index_value = df.loc[cp, 'SQALE_INDEX']

            # Conditional conversion: if periodicity is 'complete', convert SQALE_INDEX to float
            if periodicity == 'complete':
                sqale_index_value = float(sqale_index_value)
            # Append a dictionary with date and SQALE_INDEX
            change_points_data.append({
                'date': cp.strftime('%Y-%m-%d %H:%M:%S'),  # Convert Timestamp to a string format
                'SQALE_INDEX': sqale_index_value
            })
        # Construct the path for the JSON file
        output_dir = os.path.join(base_path, 'Changepoints_JSON')
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory for JSON files exists
        # Now, specify the file path for saving the changepoints
        output_path = os.path.join(output_dir, f"{project_name}_changepoints.json")

        # Save the change_points_data as a JSON file
        with open(output_path, "w") as f:
            json.dump({"change_points": change_points_data}, f, indent=4)

        # Optionally, return change points for further analysis
        return change_points

    except Exception as e:
        logger.error(f"Error processing {project_name}: {str(e)}")
        return None




# Bayesian Change Point Detection Model using Multivariate T-Distribution
class MultivariateT(BaseLikelihood):
    def __init__(self, dims: int = 1, dof: int = 0, kappa: int = 1, mu: float = -1, scale: float = -1):
        if dof == 0:
            dof = dims + 1
        if mu == -1:
            mu = [0] * dims
        else:
            mu = [mu] * dims
        if scale == -1:
            scale = np.identity(dims)
        else:
            scale = np.identity(scale)

        self.t = 0
        self.dims = dims
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def pdf(self, data: np.array):
        self.t += 1
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims((self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        ret = np.empty(self.t)
        try:
            for i, (df, loc, shape) in islice(
                enumerate(zip(t_dof, self.mu, inv(expanded * self.scale))), self.t
            ):
                ret[i] = multivariate_t.pdf(x=data, df=df, loc=loc, shape=shape)
        except AttributeError:
            raise Exception(
                "You need scipy 1.6.0 or greater to use the multivariate t distribution"
            )
        return ret

    def update_theta(self, data: np.array, **kwargs):
        centered = data - self.mu
        self.scale = np.concatenate(
            [
                self.scale[:1],
                inv(
                    inv(self.scale)
                    + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2))
                    * (np.expand_dims(centered, 2) @ np.expand_dims(centered, 1))
                ),
            ]
        )
        self.mu = np.concatenate(
            [
                self.mu[:1],
                (np.expand_dims(self.kappa, 1) * self.mu + data)
                / np.expand_dims(self.kappa + 1, 1),
            ]
        )
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])


# function call
def bayesian_change_point_detection():
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
        method_biweekly = trigger_changepoint_detection(
            df_path=os.path.join(biweekly_data_path, biweekly_file),
            project_name=project,
            periodicity="biweekly",
        )

    # Process monthly data
    for monthly_file in monthly_files:
        if monthly_file == '.DS_Store':
            continue
        project = monthly_file[:-4]
        print(f"> Processing {project} for monthly data")
        method_monthly = trigger_changepoint_detection(
            df_path=os.path.join(monthly_data_path, monthly_file),
            project_name=project,
            periodicity="monthly",
        )

    # Process complete data
    for complete_file in complete_files:
        if complete_file == '.DS_Store':
            continue
        project = complete_file[:-4]
        print(f"> Processing {project} for complete data")
        method_complete = trigger_changepoint_detection(
            df_path=os.path.join(complete_data_path, complete_file),
            project_name=project,
            periodicity="complete",
        )