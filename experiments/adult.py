import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score

from models import TT_model
from kernels import pure_power_features_full

with open("data/adult.csv") as adult_data:
    data = pd.read_csv(adult_data, header=None)
    data = data.values[1:,:]
    data = data.astype(float)

X = data[:,:-1]
Y = data[:,-1]
Y = np.where(Y > 0, 1, -1)
RMSE = []
nlls = []
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

    R = [5 for _ in range(D -1)]
    R = [1]+R+[1]
    M = [20 for _ in range(D)]

    a, b = 1e-2,1e-3
    c, d = [1e-6 * np.ones(R[d]) for d in range(D+1)], [1e-6 * np.ones(R[d]) for d in range(D+1)]
    g, h = [1e-6 * np.ones(M[d]) for d in range(D)], [1e-6 * np.ones(M[d]) for d in range(D)]

    model = TT_model.BTTKM(X_train.shape[1], R, M, pure_power_features_full)
    model.train(X_train, Y_train, a_0=a, b_0=b, plotting=False)

    predictions_mean, predictions_std = model.predict(X_test)

    accuracy = 1 - accuracy_score(Y_test, np.sign(predictions_mean))
    RMSE.append(accuracy)

    probs_gt_zero = norm.sf(0, loc=predictions_mean, scale=predictions_std)  # P(y > 0)
    y_test_binary = (Y_test + 1) // 2
    eps = 1e-15
    y_pred_prob = np.clip(probs_gt_zero, eps, 1 - eps)
    nll = -np.mean(
        y_test_binary * np.log(y_pred_prob)
        + (1 - y_test_binary) * np.log(1 - y_pred_prob)
    )
    nlls.append(nll)
    #
    # plt.scatter(X_test[:, 0], Y_test, alpha=0.7)
    # plt.scatter(X_test[:, 0], predictions_mean_unscaled, alpha=0.7)
    # plt.show()
    print(f"RMSE:{RMSE[-1]}, nll:{nlls[-1]}")

print(f"mean RMSE:{np.mean(RMSE)} with standard deviation:{np.std(RMSE)}")
print(f"mean nll:{np.mean(nlls)} with standard deviation:{np.std(nlls)}")

with open("adult.txt", "w") as f:
    f.write(f"mean RMSE:{np.mean(RMSE)} with standard deviation:{np.std(RMSE)}\nmean nll:{np.mean(nlls)} with standard deviation:{np.std(nlls)}")
