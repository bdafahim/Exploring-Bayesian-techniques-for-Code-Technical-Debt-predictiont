def get_best_model_params(project_name, periodicity, training_df, seasonality, backward_modelling):
    if project_name == 'archiva':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[0, 1, 1], [0, 0, 0, 26]]
            best_aic = 1606.92
            best_regressors = ["S00117", "S00108"]
        else:
            best_model_cfg = [[0, 1, 0], [0, 0, 0, 12]]
            best_aic = 812.77
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S1488",
                "S1905",
                "UselessImportCheck",
                "S00108"
            ]
        
    elif project_name == 'httpcore':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[0, 1, 0], [0, 0, 0, 26]]
            best_aic = 2931.39
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S1488",
                "S1905",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S1151"
            ]
        else:
            best_model_cfg = [[0, 1, 0], [0, 0, 0, 12]]
            best_aic = 1385.29
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S1488",
                "DuplicatedBlocks",
                "S00112"
            ]
    elif project_name == 'digester':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[2, 1, 1], [0, 0, 0, 26]]
            best_aic = 2931.39
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S1905",
                "S1488",
            ]
        else:
            best_model_cfg = [[2, 1, 0], [0, 0, 0, 12]]
            best_aic = 2407.12
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S1226",
                "S1155",
                "S1132"
            ]
    elif project_name == 'collections':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[2, 0, 0], [0, 0, 0, 26]]
            best_aic = 2931.39
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S00122",
                "S1488",
                "S1905",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S1155",
                "S00108",
                "S1151"
            ]
        else:
            best_model_cfg = [[0, 1, 4], [0, 0, 0, 12]]
            best_aic = 2407.12
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S1488",
                "S1905",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S1151",
                "S1132",
                "S1481"
            ]
    elif project_name == 'batik':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[2, 0, 0], [0, 1, 0, 26]]
            best_aic = -169.57
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S00122",
                "S1488",
                "S1905",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S00112",
                "S1155",
                "S00108",
                "S1151",
                "S1132",
                "S1481"
            ]
        else:
            best_model_cfg = [[0, 1, 4], [0, 0, 0, 12]]
            best_aic = 2407.12
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S1488",
                "S1905",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S1151",
                "S1132",
                "S1481"
            ]
    elif project_name == 'bcel':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[1, 1, 0], [0, 0, 0, 26]]
            best_aic = 3559.22
            best_regressors = [
                "S1213",
                "S1488",
                "S1226",
                "S00112",
                "S1155",
                "S1151",
                "S1132",
                "S1481"
            ]
        else:
            best_model_cfg = [[2, 1, 2], [0, 0, 0, 12]]
            best_aic = 1797.43
            best_regressors = [
                "S00117",
                "S1226",
                "S00112",
                "S1155",
                "S1151",
                "S1132",
                "S1481"
            ]
    elif project_name == 'beanutils':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[1, 0, 0], [0, 0, 0, 26]]
            best_aic = 5363.08
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S00122",
                "S1488",
                "S1905",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S1155",
                "S00108",
                "S1151",
                "S1132",
                "S1481"
            ]
        else:
            best_model_cfg = [[0, 1, 0], [0, 0, 0, 12]]
            best_aic = 2521.03
            best_regressors = [
                "S1905",
                "S1226",
                "S00112",
                "S00108",
                "S1481"
            ]
    elif project_name == 'cocoon':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[0, 1, 0], [0, 0, 0, 26]]
            best_aic = 1434.43
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S00122",
                "S1905",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S1155"
            ]
        else:
            best_model_cfg = [[1, 1, 0], [0, 1, 1, 12]]
            best_aic = 2521.03
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S00112",
                "S00108",
                "S1151",
                "S1132"
            ]
    elif project_name == 'codec':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[0, 1, 1], [0, 0, 0, 26]]
            best_aic = 1434.43
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S00122",
                "S1905",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S00108",
                "S1151",
                "S1481"
            ]
        else:
            best_model_cfg = [[1, 0, 0], [1, 0, 0, 12]]
            best_aic = 113.07
            best_regressors = [
                "S1905",
                "DuplicatedBlocks",
                "S1481"
            ]
    elif project_name == 'commons-cli':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[1, 1, 1], [0, 0, 1, 26]]
            best_aic = 4963.69
            best_regressors = [
                "S1488",
                "DuplicatedBlocks",
                "S00112",
                "S1132"
            ]
        else:
            best_model_cfg = [[2, 1, 2], [0, 0, 0, 12]]
            best_aic = 2199.92
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S1488",
                "S1226",
                "S00112",
                "S1155",
                "S00108",
                "S1132",
                "S1481"
            ]
    elif project_name == 'commons-exec':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[1, 0, 0], [0, 0, 0, 26]]
            best_aic = 2518.98
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S1488",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S00112",
                "S00108",
                "S1151",
                "S1132",
                "S1481"
            ]
        else:
            best_model_cfg = [[2, 1, 2], [0, 0, 0, 12]]
            best_aic = 2196.91
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S1488",
                "S1226",
                "S00112",
                "S1155",
                "S1132",
                "S1481"
            ]
    elif project_name == 'commons-fileupload':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[2, 0, 1], [0, 0, 0, 26]]
            best_aic = 2044.56
            best_regressors = [
                "S1905",
                "UselessImportCheck"
            ]
        else:
            best_model_cfg = [[1, 1, 1], [0, 0, 0, 12]]
            best_aic = 887.55
            best_regressors = [
                "S1488",
                "S1905",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S00112",
                "S1155",
                "S1132",
                "S1481"
            ]
    elif project_name == 'commons-io':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[1, 0, 0], [1, 0, 1, 26]]
            best_aic = 20.0
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S00112",
                "S1132",
                "S1481"
            ]
        else:
            best_model_cfg = [[2, 0, 1], [0, 0, 0, 12]]
            best_aic = 26.0
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00122",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S00112",
                "S1155",
                "S00108",
                "S1151",
                "S1132",
                "S1481"
            ]
    elif project_name == 'commons-jelly':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[1, 0, 0], [0, 0, 0, 26]]
            best_aic = 4744.49
            best_regressors = [
                "RedundantThrowsDeclarationCheck",
                "S00122",
                "S1905",
                "UselessImportCheck",
                "DuplicatedBlocks",
                "S1226",
                "S1155",
                "S1151",
                "S1132",
                "S1481"
            ]
        else:
            best_model_cfg = [[1, 0, 1], [1, 0, 0, 12]]
            best_aic = 79.91
            best_regressors = [
                "S1213",
                "S00122",
                "S1488",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S1155",
                "S00108",
                "S1151",
                "S1132"
            ]
    elif project_name == 'commons-jexl':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[1, 1, 2], [0, 0, 0, 26]]
            best_aic = 3755.07
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "S00122",
                "DuplicatedBlocks",
                "S1226",
                "S00112",
                "S1155",
                "S00108",
                "S1132"
            ]
        else:
            best_model_cfg = [[2, 1, 0], [1, 0, 0, 12]]
            best_aic = 1503.06
            best_regressors = [
                "S00117",
                "S1151"
            ]
    elif project_name == 'configuration':
        output_flag = True
        if periodicity == 'biweekly':
            best_model_cfg = [[3, 1, 1], [0, 0, 0, 26]]
            best_aic = 3105.48
            best_regressors = [
                "S1213",
                "RedundantThrowsDeclarationCheck",
                "S00117",
                "MethodCyclomaticComplexity",
                "S1226",
                "S00108",
                "S1186"
            ]
        else:
            best_model_cfg = [[0, 1, 1], [0, 0, 0, 12]]
            best_aic = 1536.79
            best_regressors = [
                "S1192",
                "RedundantThrowsDeclarationCheck",
                "MethodCyclomaticComplexity",
                "S00108",
                "S1151"
            ]
    else:
        best_model_cfg, best_aic, best_regressors, output_flag = backward_modelling(
            df=training_df, periodicity=periodicity, seasonality=seasonality
        )
    
    return best_model_cfg, best_aic, best_regressors, output_flag