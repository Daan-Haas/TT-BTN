import numpy as np

from models.TT_model import BTTKM
from kernels import quadratic_kernel, pure_power_features_full
from utils import unfold, khatri_rao

def generate_pure_power_dataset(D, M, R_min, R_max, number_data_points, noise_variance):
    print("generating dataset")
    X_train = np.random.uniform(-1, 1, (number_data_points,D))
    X_test = np.random.uniform(-1, 1, (number_data_points,D))

    ranks = [np.random.randint(R_min, R_max) for _ in range(D - 1)]
    ranks = [1]+ ranks + [1]
    dims = [M for _ in range(D)]
    print(ranks, dims)
    model = BTTKM(D, ranks, dims, pure_power_features_full)
    model.W = [model.W[d] for d in range(D)]
    mean_Y_train, _ = model.predict(X_train)
    mean_Y_test, _ = model.predict(X_test)
    Y_train = mean_Y_train + np.random.normal(0, scale=noise_variance, size=(number_data_points,1))
    Y_test = mean_Y_test + np.random.normal(0, noise_variance, (number_data_points,1))
    return X_train, Y_train, X_test, Y_test, model

def generate_dense_dataset(N, D, M, scale=1, noise_variance=0):
    X = np.random.standard_normal((N, D))
    Phi = pure_power_features_full(X, M**2).transpose([1,0,2]).reshape(N,-1)

    ground_truth = scale * np.random.standard_normal(D*M**2)

    Y = Phi @ ground_truth
    return X, Y
