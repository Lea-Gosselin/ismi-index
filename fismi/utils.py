import numpy as np

# RMSE between actual and fitted
def rmse(act,fit):
    meanSquaredError = ((fit - act) ** 2).mean()
    rmse = np.sqrt(meanSquaredError)
    return rmse