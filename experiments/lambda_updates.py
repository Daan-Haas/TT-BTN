from models.TT_model import *
from toy_data import *

seed = 101
# np.random.seed(seed)
### Data settings ###
I = 5 #
N = 100 # Number of data points
noise_var = 0.01 # Noise variance

### Model settings ###
D = 3 # Number of cores
# X_train, Y_train, X_test, Y_test, ground_truth = generate_pure_power_dataset(3, 5, 5, 3, 5, N, noise_var, update='lambda')

ranks = [3 for _ in range(D-1)] # Tensor-train ranks
ranks = [1] + ranks + [1] # first and last rank must be 1 to maintain output dimension
dims = [I for _ in range(D)] # dimensionality of kernels

X_train, Y_train, X_test, Y_test, ground_truth = generate_pure_power_dataset(3, 3, 5, N, noise_var)
ranks = [5 for _ in range(D-1)] # Tensor-train ranks
ranks = [1] + ranks + [1] # first and last rank must be 1 to maintain output dimension
model = BTTKM(D, ranks, [I for _ in range(D)], pure_power_features_full)
model.train(X_train, Y_train, a_0=1e-3, b_0=1e-5, lambda_update=True, iteration_limit=200)
results = model.predict(X_test)
train_results = model.predict(X_train)
plt.scatter(X_train[:, 0], Y_train, label='ground truth')
plt.scatter(X_train[:, 0], train_results, label="predicted", alpha=0.8)
plt.legend()
plt.show()
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
fig, axs = plt.subplots(1, D, constrained_layout=True)
fig.tight_layout()
plt.set_cmap("binary")
for d in range(D):
    relevance_grid = [np.outer(model.delta[d], model.lambda_R[d]) for d in range(D)]
    axs[d].matshow(relevance_grid[d])
    ylabels = [f'{model.delta[d][m]:.1g}' for m in range(model.M[d])]
    xlabels = [f'{model.lambda_R[d][r]:.1g}' for r in range(model.R[d])]
    axs[d].set_yticks(range(len(model.delta[d])), labels=[])
    axs[d].set_xticks(range(len(model.lambda_R[d])), labels=xlabels, rotation=90)

    axs[d].set_xlabel(r"Relevance $\boldsymbol{\lambda}_d$")
plt.show()