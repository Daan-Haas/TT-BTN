import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score

from models import TT_model, CPD_model
from kernels import pure_power_features_full

with open("data/spambase.csv") as spambase_data:
    df = pd.read_csv(spambase_data, header=None)
    df.columns = df.iloc[0]
    df = df[1:]
    df.reset_index(drop=True, inplace=True)
    df = df.values
    df = df.astype(float)

X = df[:,:-1]
Y = df[:,-1]
Y = np.array([float(y[0]) if isinstance(y, (list, np.ndarray)) else float(y) for y in Y])
Y = np.where(Y > 0, 1, -1)

feature_dimension = 30
CPD_max_rank = 10
TT_max_rank = 3

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

    D = X_train.shape[1]
    N = X_train.shape[0]

    R = [TT_max_rank for _ in range(D -1)]
    R = [1]+R+[1]
    M = [feature_dimension for _ in range(D)]

    a, b = 1e-1,1e-3
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

    predictions_mean, predictions_std = BTTKM.predict(X_test)

    accuracy = 1 - accuracy_score(Y_test, np.sign(predictions_mean))
    TT_RMSE.append(accuracy)

    probs_gt_zero = norm.sf(0, loc=predictions_mean, scale=predictions_std)  # P(y > 0)
    y_test_binary = (Y_test + 1) // 2
    eps = 1e-15
    y_pred_prob = np.clip(probs_gt_zero, eps, 1 - eps)
    nll = -np.mean(
        y_test_binary * np.log(y_pred_prob)
        + (1 - y_test_binary) * np.log(1 - y_pred_prob)
    )
    TT_nlls.append(nll)


    a, b = 1e-2, 1e-3
    c, d = 1e-5 * np.ones(CPD_max_rank), 1e-6 * np.ones(CPD_max_rank)
    g, h = 1e-6 * np.ones(feature_dimension), 1e-6 * np.ones(feature_dimension)
    BTNKM = CPD_model.btnkm(D, feature_dimension, CPD_max_rank)
    CPD_start_time = time.time()
    R, _, _, _, _, _, _ = BTNKM.train(
        features=X_train,
        target=Y_train,
        input_dimension=feature_dimension,
        max_rank=CPD_max_rank,
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
    prediction_mean, prediction_std, mse = BTNKM.predict(
        features=X_test,
        input_dimension=feature_dimension,
        true_values=Y_test,
        classification=True,
    )

    CPD_RMSE.append(mse)

    # NLL
    probs_gt_zero = norm.sf(0, loc=prediction_mean, scale=prediction_std)  # P(y > 0)
    y_test_binary = (Y_test + 1) // 2
    eps = 1e-15
    y_pred_prob = np.clip(probs_gt_zero, eps, 1 - eps)
    nll = -np.mean(
        y_test_binary * np.log(y_pred_prob)
        + (1 - y_test_binary) * np.log(1 - y_pred_prob)
    )
    CPD_nlls.append(nll)

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

with open("spambase.txt", "w") as f:
    f.write(f"TT:\n"
            f"mean RMSE:{np.mean(TT_RMSE)} with standard deviation:{np.std(TT_RMSE)}\n"
            f"mean nll:{np.mean(TT_nlls)} with standard deviation:{np.std(TT_nlls)}\n"
            f"average rank:{np.mean(TT_ranks)}, trained in {np.sum(TT_times)} seconds"
            f"\n\nCPD:\n"
            f"mean RMSE:{np.mean(CPD_RMSE)} with standard deviation:{np.std(CPD_RMSE)}\n"
            f"mean nll:{np.mean(CPD_nlls)} with standard deviation:{np.std(CPD_nlls)}\n"
            f"average rank:{np.mean(CPD_ranks)}, trained in {np.sum(CPD_times)} seconds")
