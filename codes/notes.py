# Load the best model configuration and regressors from the appropriate file
    if(seasonality):
         model_config_path = os.path.join(DATA_PATH, "best_sarimax_models", periodicity, f"{project_name}.json")
    else:
        model_config_path = os.path.join(DATA_PATH, "best_arimax_models", periodicity, f"{project_name}.json")

    if not os.path.exists(model_config_path):
        print(f"Model configuration file not found for project {project_name} and periodicity {periodicity}.")
        return None

    with open(model_config_path, 'r') as file:
        model_data = json.load(file)
        best_model_cfg = model_data.get('model_params')
        best_aic = model_data.get('best_aic')
        best_regressors = model_data.get('best_regressors')
        output_flag = True

    if not best_model_cfg or not best_regressors:
        print(f"Model configuration or regressors missing for project {project_name}.")
        output_flag = False
        return None

  '''residuals = actual - predicted
        plt.figure(figsize=(12, 6))
        plt.plot(testing_df['COMMIT_DATE'], residuals, label='Residuals', marker='o')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Commit Date')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} Residuals Over Time')
        plt.legend()
        plt.show()'''
