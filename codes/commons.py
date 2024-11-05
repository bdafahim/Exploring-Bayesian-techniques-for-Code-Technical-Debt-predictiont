# Main paths for the construction of the project

DATA_PATH = "/Users/badruddduzaahmed/Documents/Oulu/Thesis/replication-package/data"
assessment_statistics = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE"]
final_table_columns = ['Approach', 'Type', 'MAPE', 'MAE', 'MSE', 'RMSE']
FINAL_TS_TABLE_COLS = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE", "AIC", "BIC"]
INITIAL_VARS = ['S1213', 'RedundantThrowsDeclarationCheck', 'S00117', 'S00122', 'S1488', 'S1905', 'UselessImportCheck',
                'DuplicatedBlocks', 'S1226', 'S00112', 'S1155', 'S00108', 'S1151', 'S1132', 'S1481']

# Flag values for process control of the project
PREPROCESSING = False
SARIMAX = False
SIMULATION = False
SIMULATION_FUTURE_POINTS = False
RELATED_WORK = False
ML_MODELS = False
COMBINE_RESULTS = False
DGLM = False
ORBIT = False
PYBSTS = False
PYMC = False
PYDLM = False
CHANGEPOINT = True