import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from models import TT_model, CPD_model
from kernels import pure_power_features_full

with open("data/concrete.csv") as concrete_data:
    data = pd.read_csv(concrete_data, header=None)
    data = data.values[1:,:]
    data = data.astype(float)

X = data[:,:-1]
Y = data[:,-1]

feature_dimension = 20
max_rank_CPD = 25
max_rank_TT = 5

TT_RMSE = []
TT_nlls = []
BTTKM_time = []
TT_ranks = []
TT_times = []


CPD_RMSE = []
CPD_nlls = []
TN_BKM_time = []
CPD_ranks = []
CPD_times = []

for i in range(10):
    print(f"Concrete, repeat {i}")
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

    D = X_train.shape[1]
    N = X_train.shape[0]

    R = [max_rank_TT for _ in range(D -1)]
    R = [1]+R+[1]
    M = [feature_dimension for _ in range(D)]

    a, b = 1e-2,1e-3
    c, d = [1e-5 * np.ones(R[d]) for d in range(D+1)], [1e-6 * np.ones(R[d]) for d in range(D+1)]
    g, h = [1e-6 * np.ones(M[d]) for d in range(D)], [1e-6 * np.ones(M[d]) for d in range(D)]

    BTTKM = TT_model.BTTKM(D, R, M, pure_power_features_full)
    TT_start_time = time.time()
    BTTKM.train(X_train, Y_train,
                a_0=a, b_0=b,
                c_0=c, d_0=d,
                max_iter=50,
                plotting=False,
                convergence_bound=1e-4,
                lambda_update=True,
                delta_update=True,
                rank_pruning=True)
    TT_end_time = time.time()
    TT_times.append(TT_end_time - TT_start_time)

    TT_ranks.append(BTTKM.R)

    TT_predictions_mean, TT_predictions_std = BTTKM.predict(X_test)
    TT_predictions_mean_unscaled = (TT_predictions_mean * Y_std) + Y_mean
    TT_predictions_std_unscaled = TT_predictions_std * Y_std

    TT_error = TT_predictions_mean_unscaled - Y_test.reshape(-1, 1)
    TT_RMSE.append(np.sqrt(np.sum(TT_error ** 2) / N))

    TT_nll = (0.5 * np.log(2 * np.pi * TT_predictions_std_unscaled ** 2)
              + 0.5 * (TT_error ** 2) / (TT_predictions_mean_unscaled ** 2))
    TT_nlls.append(np.mean(TT_nll))

    a, b = 1e-3, 1e-3
    c, d = 1e-5 * np.ones(max_rank_CPD), 1e-6 * np.ones(max_rank_CPD)
    g, h = 1e-6 * np.ones(feature_dimension), 1e-6 * np.ones(feature_dimension)

    BTNKM = CPD_model.btnkm(D, 20, 25)
    CPD_start_time = time.time()
    R, _, _, _, _, _, _ = BTNKM.train(
        features=X_train,
        target=Y_train,
        input_dimension=feature_dimension,
        max_rank=max_rank_CPD,
        shape_parameter_tau=a,
        scale_parameter_tau=b,
        shape_parameter_lambda=c,
        scale_parameter_lambda=d,
        shape_parameter_delta=g,
        scale_parameter_delta=h,
        max_iter=50,
        precision_update=True,
        lambda_update=True,
        delta_update=True,
        plot_results=False,
        prune_rank=True,
    )
    CPD_end_time = time.time()
    CPD_times.append(CPD_end_time - CPD_start_time)
    # Predict (mse is returned by the predict function)
    CPD_prediction_mean, CPD_prediction_std, _ = BTNKM.predict(
        features=X_test, input_dimension=feature_dimension
    )

    CPD_prediction_mean_unscaled = CPD_prediction_mean * Y_std + Y_mean
    CPD_prediction_std_unscaled = CPD_prediction_std * Y_std

    # nll
    nll = 0.5 * np.log(2 * np.pi * CPD_prediction_std_unscaled ** 2) + 0.5 * (
            (Y_test - CPD_prediction_mean_unscaled) ** 2
    ) / (CPD_prediction_std_unscaled ** 2)
    CPD_nlls.append(np.mean(nll))

    # rmse
    rmse = np.sqrt(np.mean((CPD_prediction_mean_unscaled - Y_test) ** 2))
    CPD_RMSE.append(rmse)

    CPD_ranks.append(R)

    # plt.scatter(X_test[:,0], Y_test, alpha=0.7)
    # plt.scatter(X_test[:,0], TT_predictions_mean_unscaled, alpha=0.7)
    # plt.show()
    print("TT:\n")
    print(f"RMSE:{TT_RMSE[-1]}, nll:{TT_nlls[-1]}, time:{TT_times[-1]}, rank:{np.mean(TT_ranks[-1])}")
    print("\n\nCPD:\n")
    print(f"RMSE:{CPD_RMSE[-1]}, nll:{CPD_nlls[-1]}, time:{CPD_times[-1]}, rank:{CPD_ranks[-1]}")

print("TT:\n")
print(
    f"mean RMSE:{np.mean(TT_RMSE)} with standard deviation:{np.std(TT_RMSE)}, rank:{np.mean([np.mean(TT_rank) for TT_rank in TT_ranks])}, in {np.sum(TT_times)} seconds")
print(f"mean nll:{np.mean(TT_nlls)} with standard deviation:{np.std(TT_nlls)}")
print("\n\nCPD:\n")
print(
    f"mean RMSE:{np.mean(CPD_RMSE)} with standard deviation:{np.std(CPD_RMSE)}, rank:{np.mean(CPD_ranks)}, in {np.sum(CPD_times)} seconds")
print(f"mean nll:{np.mean(CPD_nlls)} with standard deviation:{np.std(CPD_nlls)}")

with open("concrete.txt", "w") as f:
    f.write(f"TT:\n"
            f"mean RMSE:{np.mean(TT_RMSE)} with standard deviation:{np.std(TT_RMSE)}\n"
            f"mean nll:{np.mean(TT_nlls)} with standard deviation:{np.std(TT_nlls)}\n"
            f"average rank:{np.mean(TT_ranks)}, trained in {np.sum(TT_times)} seconds"
            f"\n\nCPD:\n"
            f"mean RMSE:{np.mean(CPD_RMSE)} with standard deviation:{np.std(CPD_RMSE)}\n"
            f"mean nll:{np.mean(CPD_nlls)} with standard deviation:{np.std(CPD_nlls)}\n"
            f"average rank:{np.mean(CPD_ranks)}, trained in {np.sum(CPD_times)} seconds")
