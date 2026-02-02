import numpy as np

from model import BTTKM
from kernels import quadratic_kernel


def generate_lin_dataset(dimensionality, number_data_points, noise_variance):
    parameters = np.random.normal(0,1, dimensionality)
    noise = np.random.normal(0, noise_variance, dimensionality)

    X_train = np.random.uniform(-1, 1, (number_data_points, dimensionality))
    X_test = np.random.uniform(-1, 1, (number_data_points, dimensionality))

    Y_train = np.zeros(number_data_points)
    Y_test = np.zeros(number_data_points)
    for i, x in enumerate(X_train):
        Y_train[i] = np.dot(x, parameters)
        X_train[i] = x + noise
    for i, x in enumerate(X_test):
        Y_test[i] = np.dot(x, parameters)
        X_test[i] = x + noise

    return X_train, Y_train, X_test, Y_test, parameters

def generate_quadratic_dataset(dimensionality, number_data_points, noise_variance):
    X_train = np.random.uniform(-1, 1, (number_data_points, dimensionality))
    X_test = np.random.uniform(-1, 1, (number_data_points, dimensionality))

    ranks = [1, 3, 3, 1]
    dims = [3, 3, 3]
    model = BTTKM(3, ranks, dims, quadratic_kernel)

    Y_train = model.predict(X_train) + np.random.normal(0, scale=noise_variance, size=(number_data_points,1))
    Y_test = model.predict(X_test) + np.random.normal(0, noise_variance, (number_data_points,1))
    return X_train, Y_train.T, X_test, Y_test, model