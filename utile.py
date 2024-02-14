import numpy as np


def inlocuireNAN(X):  # asumama ca primim X ca numpy.ndarray
    mediiCol = np.nanmean(a=X, axis=0)  # avem variabilele pe coloane
    locs = np.where(np.isnan(X))
    print(mediiCol)
    print(locs, type(locs))
    X[locs] = mediiCol[locs[1]]  # inlocuire cu indicii coloanelor pentru care lipsesc valori
    return X