import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix

#* General Functions


# mse loss
def mse(y_pred, y, give_rmse=False, repr=False):
    mse = np.mean((y_pred - y)**2)  
    rmse = np.sqrt(mse)
    
    if not give_rmse and not repr:
        return mse
    elif not give_rmse and repr:
        return f"The Mean Squared Error is: {mse}"
    elif give_rmse and not repr:
        return mse, rmse

    return f"The Mean Squared Error is: {mse}, The Root Mean Squared Error is: {rmse}"


### initialize 2D mixtures
def init_multivariate_mixtures(centres, samples=100, r=10, plot=False, return_aux=False):
    # initialize cluster means
    means = []
    for _ in range(centres):
        mean = np.random.choice(r*10, 2)/10
        means.append(list(mean))

    # initialize cluster covariances
    covs = []
    for _ in range(len(means)):
        covs.append(make_spd_matrix(2))

    # generate multivariate normal data
    X = []
    for mean, cov in zip(means, covs):
        x = np.random.multivariate_normal(mean, cov, samples)
        X += list(x)

    # making array
    X = np.array(X)
    np.random.shuffle(X)

    if plot:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7, edgecolor='black')

    if return_aux:
        return X, means, covs

    return X

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
