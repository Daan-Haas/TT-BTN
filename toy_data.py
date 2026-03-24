import numpy as np

from models.TT_model import BTTKM
from kernels import quadratic_kernel, pure_power_features_full
from utils import unfold, khatri_rao


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

def generate_pure_power_dataset(D, M_true, M_max, R_true, R_max, N, noise_variance=0, update='both'):
    print("generating dataset")
    X_train = np.random.standard_normal((N, D))
    X_test = np.random.standard_normal((N, D))
    feature_map_train = pure_power_features_full(X_train, M_max)
    feature_map_test = pure_power_features_full(X_test, M_max)

    if update in ['delta', 'both']:
        M = [np.random.choice(range(M_true, M_max)) for _ in range(D)]
        print(f"M = {M}")
    else:
        M = [M_max for _ in range(D)]
    if update in ['lambda', 'both']:
        R = [np.random.choice(range(R_true, R_max)) for _ in range(D-1)]
        print(f"R = {R}")
    else:
        R = [R_max for _ in range(D-1)]
    R = [1] + R + [1]
    feature_map_train = [feature_map_train[d,:,:M[d]] for d in range(D)]
    feature_map_test = [feature_map_test[d,:,:M[d]] for d in range(D)]
    W = [0.25*np.random.random([R[i],M[i],R[i+1]]) for i in range(D)]
    Y_train = np.ones((N, 1))  # N x 1
    Y_test = np.ones((N, 1))
    for d in range(D):
        Y_train = khatri_rao(feature_map_train[d], Y_train) @ unfold(W[d], 3).T  # (N x R_d M_d)(R_d M_d x R_{d+1})
        Y_test = khatri_rao(feature_map_test[d], Y_test) @ unfold(W[d], 3).T

    Y_train = Y_train + np.random.normal(0, noise_variance)
    Y_test = Y_test + np.random.normal(0,noise_variance)
    return X_train, Y_train/np.linalg.norm(Y_train), X_test, Y_test/np.linalg.norm(Y_test)

def generate_dense_dataset(N, D, M, scale=1, noise_variance=0):
    X = np.random.standard_normal((N, D))
    Phi = pure_power_features_full(X, M).transpose([1,0,2]).reshape(N,-1)

    ground_truth = scale * np.random.standard_normal(D*M)

    Y = Phi @ ground_truth
    return X, Y
