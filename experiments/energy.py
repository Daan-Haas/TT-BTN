import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import TT_model
from kernels import pure_power_features_full

with open("data/energy.csv") as energy_data:
    data = pd.read_csv(energy_data, header=None)
    data = data.values[1:,:]
    data = data.astype(float)

X = data[:,:-1]
D = X.shape[1]
N = X.shape[0]
Y = data[:,-1]
RMSE = []
for i in range(10):
    np.random.seed(i)
    indices = np.random.permutation(len(X))
    split_index = int(0.90 * len(X))  # 90% for training, 10% for testing
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    Y_train, Y_test = Y[indices[:split_index]], Y[indices[split_index:]]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std  # Use train stats

    Y_mean = Y_train.mean()
    Y_std = Y_train.std()
    Y_train = (Y_train - Y_mean) / Y_std

    R = [5 for i in range(X_train.shape[1] -1)]
    R = [1]+R+[1]
    M = [8 for _ in range(X_train.shape[1])]

    a, b = 1, 1e-3
    c, d = [1e-6 * np.ones(R[d]) for d in range(D+1)], [1e-6 * np.ones(R[d]) for d in range(D+1)]
    g, h = [1e-6 * np.ones(M[d]) for d in range(D)], [1e-6 * np.ones(M[d]) for d in range(D)]

    model = TT_model.BTTKM(X_train.shape[1], R, M, pure_power_features_full)

    model.train(X_train, Y_train, a_0=a, b_0=b,
                           c_0=c, d_0=d,
                           g_0=g, h_0=h,plotting=False)

    predictions_mean = model.predict(X_test)
    predictions_mean_unscaled = predictions_mean * Y_std + Y_mean
    error = predictions_mean_unscaled - Y_test

    RMSE.append(np.sqrt(np.sum(np.square(error))/N))

    plt.scatter(X_test[:,2], Y_test)
    plt.scatter(X_test[:,2],predictions_mean_unscaled)
    plt.show()

print(RMSE)

