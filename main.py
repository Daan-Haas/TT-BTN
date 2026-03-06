import matplotlib.pyplot as plt

from utils import *
from model import *
from toy_data import *

seed = 101
np.random.seed(seed)
### Data settings ###
I = 5 #
N = 100 # Number of data points
noise_var = 0.01 # Noise variance

### Model settings ###
D = 3 # Number of cores
ranks = [5 for _ in range(D-1)] # Tensor-train ranks
ranks = [1] + ranks + [1] # first and last rank must be 1 to maintain output dimension
dims = [I for _ in range(D)] # dimensionality of kernels

X_train, Y_train, X_test, Y_test, ground_truth = generate_quadratic_dataset(I, N, 0)
uninformative_train_features = np.random.random([N, 2])
X_train = np.append(X_train, uninformative_train_features, axis=1)

uninformative_test_features = np.random.random([N, 2])
X_test = np.append(X_test, uninformative_test_features, axis=1)

model = BTTKM(D, ranks, [I + 2 for _ in range(D)], quadratic_kernel)
model.train(X_train, Y_train, delta_update=True, iteration_limit=200)
results = model.predict(X_test)

error_term = 0.5*model.expectation_tau* np.linalg.norm(Y_test.reshape(model.N,1) - results)**2

L2_norms = 0
ln_q_W = 0
lambda_term = 0
delta_term = 0
for d in range(model.D):
    lambda_mat_next = np.diag(model.lambda_R[d + 1])  # R_{d+1} x R_{d+1}
    lambda_mat_prev = np.diag(model.lambda_R[d])  # R_d x R_d
    delta_mat = np.diag(model.delta[d])  # M_d x M_d
    variance_term = np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)

    L2_norms += np.trace(variance_term @ (np.outer(model.W[d].reshape((-1,1), order='F'), model.W[d].reshape((-1,1), order='F')) + np.diag(np.diag(model.Sigma[d]))))
    ln_q_W += 0.5*np.log(np.linalg.norm(model.Sigma[d])) + (model.R[d]*model.M[d]/2)*(1+np.log(2*np.pi))
    for r in range(model.R[d]):
        lambda_term += np.log(gamma(model.c_N[d][r])) + (1-np.log(model.d_N[d][r])- (1e-6/model.d_N[d][r]))*model.c_N[d][r]
    for m in range(model.M[d]):
        delta_term += np.log(gamma(model.g_N[d][m])) + (1-np.log(model.h_N[d][m]) - (1e-6/model.h_N[d][m]))*model.g_N[d][m]

tau_term = np.log(gamma(model.a_N)) + (1 - np.log(model.b_N) - (1/model.b_N))*model.a_N
ELBO = -error_term - L2_norms - ln_q_W - lambda_term - delta_term - tau_term

plt.scatter(X_test[:,0], Y_test, label="Ground truth")
plt.scatter(X_test[:,0], results, label="Predicted", alpha=0.8)
plt.title(f"predictions on Validation set, ELBO: {ELBO:.2e}")
plt.legend()
plt.show()
fig, axs = plt.subplots(2, D, constrained_layout=True)
plt.set_cmap("binary")
for d in range(D):
    relevance_grid = [np.outer(model.delta[d], model.lambda_R[d]) for d in range(D)]
    axs[0,d].matshow(relevance_grid[d])
    ylabels = [f'{model.delta[d][m]:.1g}' for m in range(model.M[d])]
    xlabels = [f'{model.lambda_R[d][r]:.1g}' for r in range(model.R[d])]
    axs[0,d].set_yticks(range(len(model.delta[d])), labels=ylabels)
    axs[0,d].set_xticks(range(len(model.lambda_R[d])), labels=[], rotation=90)

    axs[0,d].set_title(f"Core {d}")
    relevance_grid = [np.outer(model.delta[d], model.lambda_R[d+1]) for d in range(D)]
    axs[1,d].matshow(relevance_grid[d])
    ylabels = [f'{model.delta[d][m]:.1g}' for m in range(model.M[d])]
    xlabels = [f'{model.lambda_R[d+1][r]:.1g}' for r in range(model.R[d+1])]
    axs[1,d].set_yticks(range(len(model.delta[d])), labels=ylabels)
    axs[1,d].set_xticks(range(len(model.lambda_R[d+1])), labels=[], rotation=90)
axs[0,0].set_ylabel(r"Relevance $\boldsymbol{\lambda}_d, \boldsymbol{\delta}_d$")
axs[1,0].set_ylabel(r"Relevance $\boldsymbol{\lambda}_{d+1}, \boldsymbol{\delta}_d$")
plt.show()