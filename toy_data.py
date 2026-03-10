import numpy as np

from model import BTTKM
from kernels import quadratic_kernel, pure_power_features_full, no_kernel
from utils import khatri_rao, unfold

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
    print("generating dataset")
    X_train = np.random.uniform(-1, 1, (number_data_points, dimensionality))
    X_test = np.random.uniform(-1, 1, (number_data_points, dimensionality))

    ranks = [1, 3, 3, 1]
    dims = [5, 5, 5]
    model = BTTKM(3, ranks, dims, quadratic_kernel)

    Y_train = model.predict(X_train) + np.random.normal(0, scale=noise_variance, size=(number_data_points,1))
    Y_test = model.predict(X_test) + np.random.normal(0, noise_variance, (number_data_points,1))
    return X_train, Y_train.T, X_test, Y_test, model

def generate_pure_power_dataset(D_model, D_features, M, number_data_points, noise_variance):
    print("generating dataset")
    X_train = np.random.uniform(-1, 1, (number_data_points, D_model))
    X_test = np.random.uniform(-1, 1, (number_data_points, D_model))

    if D_features > 1:
        X_train_gen = pure_power_features_full(X_train, D_features, 1).transpose(1,0,2)
        X_train_gen = X_train_gen.reshape(100,D_model*D_features)
        X_test_gen = pure_power_features_full(X_test, D_features, 1).transpose(1,0,2)
        X_test_gen = X_test_gen.reshape(100,D_model*D_features)

    else:
        X_train_gen = X_train
        X_test_gen = X_test

    ranks = [3 for _ in range(int(D_model*D_features - 1))]
    ranks = [1]+ ranks + [1]
    dims = [M for _ in range(int(D_model*D_features))]
    model = BTTKM(D_model*D_features, ranks, dims, pure_power_features_full)
    Y_train = model.predict(X_train_gen) + np.random.normal(0, scale=noise_variance, size=(number_data_points,1))
    Y_test = model.predict(X_test_gen) + np.random.normal(0, noise_variance, (number_data_points,1))
    return X_train, Y_train, X_test, Y_test, model