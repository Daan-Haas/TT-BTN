import matplotlib.pyplot as plt
import numpy as np

from kernels import pure_power_features_full
from models.CPD_model import *
from models.TT_model import *
from toy_data import generate_dense_dataset

max_rank = 3
max_tries = 20
scale = 1000

all_CPD_failures = []
all_TT_failures = []
avg_upper_bounds = []
avg_lower_bounds = []
all_mags_CPD = []
all_mags_TT = []

N = 100
M = 3
Ds = [80, 100, 105, 110]

for D in Ds:
    a, b = 1e-2, 1e-3
    c, d = 1e-6 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
    g, h = 1e-6 * np.ones(max_rank), 1e-6 * np.ones(max_rank)

    tries = 0
    CPD_failures = 0
    TT_failures = 0

    upper_bounds = []
    lower_bounds = []
    mags_CPD = []
    mags_TT = []
    print(all_mags_TT)
    while tries < max_tries:
        np.random.seed(D*tries)
        X, Y = generate_dense_dataset(N,D,M, scale)

        Phi = pure_power_features_full(X, M)
        CPD_model = btnkm(D,3, 3)

        ranks = [max_rank for _ in range(D-1)]
        ranks = [1] + ranks + [1]
        dims = [3 for _ in range(D)]
        TT_model = BTTKM(D, ranks, dims, pure_power_features_full)

        upper_bound = 1
        lower_bound = 1
        for i in range(D-1):
            if i == 0:
                TT_model.W[i] = CPD_model.W_D[i].reshape([1,3,3])
            else:
                for j in range(max_rank):
                    for k in range(max_rank):
                        TT_model.W[i][j,:,k] =  CPD_model.W_D[i][j,:] if j==k else np.zeros(3)

            #calculate bounds
            mag_Phi_d = np.linalg.norm(Phi[i,0])
            _,S, _ = np.linalg.svd(CPD_model.W_D[i], full_matrices=False)
            upper_bound *= mag_Phi_d * np.max(S)
            lower_bound *= mag_Phi_d * np.min(S)
        upper_bounds.append(upper_bound)
        lower_bounds.append(lower_bound)
        TT_model.W[D-1] = CPD_model.W_D[D-1].reshape(3,3,1)

        CPD_model.train(X, Y, 3, max_rank,
                    shape_parameter_tau=a,
                    scale_parameter_tau=b,
                    shape_parameter_lambda=c,
                    scale_parameter_lambda=d,
                    shape_parameter_delta=g,
                    scale_parameter_delta=h,
                    prune_rank=False,
                    plot_results=False
                    )
        mags_CPD.append(np.linalg.norm(CPD_model.W_D[0]))

        TT_model.train(X, Y,
                       a_0=a, b_0=b,
                       fm_bias=0.2,
                       max_iter=10,
                       convergence_bound=1e-2,
                       plotting=False)
        mags_TT.append(np.linalg.norm(unfold(TT_model.W[0], 3)))

        tries += 1
        if np.linalg.norm(CPD_model.W_D) == 0:
            CPD_failures += 1
        if max([np.linalg.norm(TT_model.W[d]) for d in range(D)]) < 1e-16 or all(np.isnan([np.linalg.norm(TT_model.W[d]) for d in range(D)])):
            TT_failures += 1
        print(f'after {tries} attempts, CPD norm = {np.linalg.norm(CPD_model.W_D)}, TT norm = {max([np.linalg.norm(TT_model.W[d]) for d in range(D)])}')
    print(f'with {D} cores, {CPD_failures} under/overflow errors for CPD, {TT_failures} under/overflow errors for TT.')
    all_CPD_failures.append(CPD_failures)
    all_TT_failures.append(TT_failures)
    avg_upper_bounds.append(np.mean(upper_bounds))
    avg_lower_bounds.append(np.mean(lower_bounds))
    all_mags_CPD.append(mags_CPD)
    all_mags_TT.append(mags_TT)

plt.plot(Ds, all_CPD_failures, label='CPD')
plt.plot(Ds, all_TT_failures, label='TT')
plt.title("Occurrences of over or underflow")
plt.xlabel("ground truth scale")
plt.xticks(Ds)
plt.ylabel("number of over/underflow issues")
plt.yticks(range(0,21,2))
plt.legend()
plt.savefig('breaking_point.png')
plt.show()

plt.figure()
for i, D in enumerate(Ds):
    if i == 0:
        plt.scatter([D for _ in range(max_tries)], all_mags_CPD[i], c='orange', alpha=0.7, label='CPD')
        plt.scatter([D for _ in range(max_tries)], all_mags_TT[i], c='blue', alpha=0.7, label='TT')
    else:
        plt.scatter([D for _ in range(max_tries)], all_mags_CPD[i], c='orange', alpha=0.7)
        plt.scatter([D for _ in range(max_tries)], all_mags_TT[i], c='blue', alpha=0.7)
plt.plot(Ds, avg_upper_bounds, label='upper bound')
plt.plot(Ds, avg_lower_bounds, label='lower bound')
plt.yscale('log')
plt.legend()
plt.show()