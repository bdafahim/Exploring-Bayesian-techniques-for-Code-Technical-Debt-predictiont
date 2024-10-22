import matplotlib.pyplot as plt
import numpy as np

def compare():

    # Biweekly data
    libraries = ["Orbit-ML DLT", "Orbit-ML ETS", "pyBATS", "pyBSTS", "pyDLM"]
    mape = [1.82, 2.91, 5.15, 3.13, 3.91]
    mae = [482.59, 797.09, 999.99, 793.71, 1122.11]
    rmse = [590.40, 1019.87, 982.33, 1025.78, 1275.28]

    # Plotting the histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # MAPE Histogram
    axes[0].barh(libraries, mape, color='skyblue')
    axes[0].set_xlabel('MAPE (%)')
    axes[0].set_title('MAPE Comparison')

    # MAE Histogram
    axes[1].barh(libraries, mae, color='lightgreen')
    axes[1].set_xlabel('MAE')
    axes[1].set_title('MAE Comparison')

    # RMSE Histogram
    axes[2].barh(libraries, rmse, color='lightcoral')
    axes[2].set_xlabel('RMSE')
    axes[2].set_title('RMSE Comparison')

    # Overall plot adjustments
    plt.suptitle('Comparison of Bayesian Models - MAPE, MAE, and RMSE', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()