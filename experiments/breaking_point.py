import matplotlib.pyplot as plt

from models.CPD_model import *
from models.TT_model import *
from toy_data import generate_pure_power_dataset

max_rank = 3
max_tries = 5
input_dimensions = [50, 100, 150]

all_CPD_failures = []
all_TT_failures = []

for D in input_dimensions:
    a, b = 1e-2, 1e-3
    c, d = 1e-6 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
    g, h = 1e-6 * np.ones(max_rank), 1e-6 * np.ones(max_rank)

    tries = 0
    CPD_failures = 0
    TT_failures = 0
    while tries < max_tries:
        X_train, Y_train, X_test, Y_test = generate_pure_power_dataset(D,3,3, 3, 3, 100,
                                                                       update='neither')
        CPD_model = btnkm(D,3, 3)

        ranks = [max_rank for _ in range(D-1)]
        ranks = [1] + ranks + [1]
        dims = [3 for _ in range(D)]
        TT_model = BTTKM(D, ranks, dims, pure_power_features_full)

        for i in range(D-1):
            if i == 0:
                TT_model.W[i] = CPD_model.W_D[i].reshape([1,3,3])
            else:
                for j in range(max_rank):
                    for k in range(max_rank):
                        TT_model.W[i][j,:,k] = CPD_model.W_D[i][j,:] if j==k else np.zeros(3)
        TT_model.W[D-1] = CPD_model.W_D[D-1].reshape(3,3,1)


        CPD_model.train(X_train, Y_train, 3, max_rank,
                    shape_parameter_tau=a,
                    scale_parameter_tau=b,
                    shape_parameter_lambda=c,
                    scale_parameter_lambda=d,
                    shape_parameter_delta=g,
                    scale_parameter_delta=h,
                    prune_rank=False,
                    plot_results=False
                    )

        TT_model.train(X_train, Y_train,
                       a_0=a, b_0=b,
                       c_0=c, d_0=d,
                       g_0=g, h_0=h,
                       iteration_limit=10,
                       printing=False,
                       plotting=False)

        tries += 1
        if np.linalg.norm(CPD_model.W_D) == 0:
            CPD_failures += 1
        if max([np.linalg.norm(TT_model.W[d]) for d in range(D)]) < 1e-16:
            TT_failures += 1
        print(f'after {tries} attempts, CPD norm = {np.linalg.norm(CPD_model.W_D)}, TT norm = {max([np.linalg.norm(TT_model.W[d]) for d in range(D)])}')
    print(f'with {D} cores, {CPD_failures} under/overflow errors for CPD, {TT_failures} under/overflow errors for TT.')
    all_CPD_failures.append(CPD_failures)
    all_TT_failures.append(TT_failures)

plt.plot(input_dimensions, all_CPD_failures, label='CPD')
plt.plot(input_dimensions, all_TT_failures, label='TT')
plt.title("Occurrences of over or underflow")
plt.xlabel("Dimension")
plt.ylabel("number of over/underflow issues")
plt.legend()
plt.savefig('breaking_point.png')
plt.show()