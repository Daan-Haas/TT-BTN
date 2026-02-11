import numpy as np

from kernels import *
from utils import khatri_rao, Core, TensorTrain, block2outer, outer2block

class BTTKM:
    def __init__(self, nr_cores, ranks, dims, kernel):
        if (len(dims) != nr_cores) or (len(ranks) != nr_cores+1):
            raise ValueError(f"Incorrect number of dimensions or TT ranks, "
                             f"TT with {nr_cores} cores should have {nr_cores} dimensions and {nr_cores+1} ranks")
        if (ranks[0], ranks[-1]) != (1,1):
            raise ValueError("First and last TT rank should be 1")

        self.TT_ranks = ranks
        self.dims = dims
        self.D = nr_cores
        self.init_cores()
        size = [self.TT_ranks[d]*self.dims[d]*self.TT_ranks[d+1] for d in range(self.D)]
        self.Sigma = [np.eye(size[d]) for d in range(self.D)]
        self.kernel = kernel
        self.N = 100

    def init_cores(self):
        core_list = [Core(np.random.rand(self.TT_ranks[i],self.dims[i],self.TT_ranks[i+1])) for i in range(self.D)]
        self.W = TensorTrain(core_list)

    def train(self,
              X, Y,
              a_0=1e-6, b_0=1e-6,
              c_0=None, d_0=None,
              g_0=None, h_0=None,
              sigma_update=False,
              lambda_update=False,
              delta_update=False,
              tau_update=False,
              error_bound=1e-4, iteration_limit=100):
        print("Training")
        self.feature_map = self.kernel(X, self.D)
        self.N = X.shape[0]
        self.expectation_tau = a_0 / b_0

        c_0 = [1e-3 * np.ones(self.TT_ranks[d]) for d in range(self.D+1)]
        self.c_N = c_0.copy()
        d_0 = [1e-3 * np.ones(self.TT_ranks[d]) for d in range(self.D+1)]
        self.d_N = d_0.copy()
        self.lambda_R = [c_0[d]/d_0[d] for d in range(self.D+1)]

        g_0 = [1e-3 * np.ones(self.dims[d]) for d in range(self.D)]
        self.g_N = g_0.copy()
        h_0 = [1e-3 * np.ones(self.dims[d]) for d in range(self.D)]
        self.h_N = h_0.copy()
        self.delta = [g_0[d]/h_0[d] for d in range(self.D)]

        errors = [np.inf]
        it = 0

        while it < iteration_limit:
            # cores update
            vectorized_W = []
            for d in range(self.D):
                H_lt = khatri_rao(self.forward_accumulator_H(d), self.feature_map[d])
                H_gt = khatri_rao(self.feature_map[d], self.backward_accumulator_H(d))
                H_d = H_lt.T @ H_gt
                G_lt = khatri_rao(self.forward_accumulator_G(d), self.feature_map[d])
                G_d = khatri_rao(G_lt, self.backward_accumulator_G(d))

                mean_term = block2outer(self.expectation_tau*H_d,(self.TT_ranks[d], self.dims[d]**2))
                lambda_mat_next = np.diag(self.lambda_R[d+1])
                delta_mat = np.diag(self.delta[d])
                lambda_mat_prev = np.diag(self.lambda_R[d])
                variance_term = np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)
                self.Sigma[d] = np.linalg.inv(np.add(mean_term, variance_term))

                vectorized_W.append(self.expectation_tau*Y@G_d@self.Sigma[d])
            for d, core in enumerate(self.W.cores):
                core.core = np.permute_dims(vectorized_W[d].reshape((self.TT_ranks[d+1], self.dims[d], self.TT_ranks[d])), [2,1,0])

            # posterior update lambda
            # if lambda_update:
            #     for d in range(1, self.D):
            #         self.c_N[d] = c_0[d] + 0.5 * (self.TT_ranks[d-1]*self.dims[d-1]+self.TT_ranks[d+1]*self.dims[d])
            #
            #         W_unfold_3 = self.W.cores[d-1].unfold(3)
            #         W_unfold_1 = self.W.cores[d].unfold(1)
            #
            #         tensor_shape1 = (self.TT_ranks[d-1], self.dims[d-1], self.TT_ranks[d])
            #         variance_tensor_prev = np.reshape(np.diag(self.Sigma[d-1]), tensor_shape1)
            #         matrix_shape1 = tensor_shape1[2],tensor_shape1[1]*tensor_shape1[0]
            #         variance_matrix_prev = np.permute_dims(variance_tensor_prev, [2,1,0]).reshape(matrix_shape1)
            #
            #         sparsity_term1 = np.kron(np.diag(self.delta[d-1]), np.diag(self.lambda_R[d-1]))
            #         variance_term1 = variance_matrix_prev@np.kron(self.delta[d-1], self.lambda_R[d-1])
            #         expectation1 = np.add(np.diag(W_unfold_3 @ sparsity_term1 @ W_unfold_3.T), variance_term1)
            #
            #         tensor_shape2 = (self.TT_ranks[d], self.dims[d], self.TT_ranks[d+1])
            #         variance_tensor_next = np.reshape(np.diag( self.Sigma[d]), tensor_shape2)
            #         matrix_shape2 = (tensor_shape2[2], tensor_shape2[0]*tensor_shape2[1])
            #         variance_matrix_next = np.permute_dims(variance_tensor_next, [0,2,1]).reshape(matrix_shape2)
            #
            #         sparsity_term2 = np.kron(np.diag(self.lambda_R[d+1]), np.diag(self.delta[d]))
            #         variance_term2 = variance_matrix_next@np.kron(self.lambda_R[d], self.delta[d])
            #         expectation2 = np.add(np.diag(W_unfold_1 @ sparsity_term2 @ W_unfold_1.T), variance_term2)
            #
            #         self.d_N[d] = np.add(d_0[d], 0.5 * (expectation1 + expectation2))
            #
            #     for d in range(1, self.D):
            #         self.lambda_R[d] = self.c_N[d] / self.d_N[d]
            #
            # # # posterior update delta
            # if delta_update:
            #     for d in range(self.D):
            #         self.g_N[d] = g_0[d] + 0.5 * (self.TT_ranks[d] + self.TT_ranks[d+1])
            #         W_unfold_2 = self.W.cores[d].unfold(2)
            #
            #         tensor_shape = (self.TT_ranks[d], self.dims[d], self.TT_ranks[d+1])
            #         matrix_shape = (tensor_shape[1], tensor_shape[0]*tensor_shape[2])
            #         variance_tensor_d = np.reshape(np.diag(self.Sigma[d]), tensor_shape)
            #         variance_matrix_d = np.permute_dims(variance_tensor_d, [1,2,0]).reshape(matrix_shape)
            #
            #         sparsity_term = np.kron(np.diag(self.lambda_R[d+1]), np.diag(self.lambda_R[d]))
            #         variance_term = variance_matrix_d @ np.kron(self.lambda_R[d+1], self.lambda_R[d])
            #         expectation = np.add(np.diag(W_unfold_2 @ sparsity_term @ W_unfold_2.T), variance_term)
            #         self.h_N[d] = np.add(h_0[d], 0.5 * expectation)
            #
            #     for d in range(self.D):
            #         self.delta[d] = self.g_N[d] / self.h_N[d]
            #
            # # noise precision update
            # if tau_update:
            #     self.a_N = a_0 + self.N / 2
            #     Expectation_G = self.forward_accumulator_G(self.D)
            #     Expectation_H = np.ones(self.N).T @ self.forward_accumulator_H(self.D)
            #     frob_errors = np.linalg.norm(Y)**2 - 2*Y@Expectation_G + Expectation_H
            #     self.b_N = b_0 + 0.5 * frob_errors
            #     self.expectation_tau = self.a_N / self.b_N

            predictions = self.predict(X)
            error = Y.reshape(self.N,1) - predictions
            MSE = np.sum(np.square(error))
            errors.append(MSE)

            it += 1
            print(f"MSE: {errors[it]}")
            # if abs(errors[-1] - errors[-2]) < error_bound:
            #     plt.plot(errors)
            #     plt.show()
            #     print("convergence bound reached, exiting")
            #     break

        if it == iteration_limit:
            print("iteration limit reached, exiting")

    def predict(self, X):
        self.feature_map = self.kernel(X, self.D)
        return self.forward_accumulator_G(self.D)

    def forward_accumulator_G(self, d):
        G_k = np.ones((self.N, 1))
        for k in range(d):
            G_k = khatri_rao(G_k, self.feature_map[k]) @ self.W.cores[k].unfold(3).T
        return G_k

    def backward_accumulator_G(self, d):
        G_k = np.ones((self.N, 1))
        for k in range(self.D-1, d, -1):
            G_k = khatri_rao(self.feature_map[k], G_k) @ self.W.cores[k].unfold(1).T
        return G_k

    def forward_accumulator_H(self, d):
        H_k = np.ones((self.N, 1))
        for k in range(d):
            Wk = self.W.cores[k].unfold(3).T
            mean_WW = np.kron(Wk, Wk)
            block_shape = (self.TT_ranks[k]*self.dims[k], self.TT_ranks[k+1])
            covariance_WW = outer2block(self.Sigma[k], block_shape, mean_WW.shape)
            expectation_WW = np.asarray(mean_WW + covariance_WW)
            H_k = khatri_rao(H_k, khatri_rao(self.feature_map[k], self.feature_map[k])) @ expectation_WW
        return H_k

    def backward_accumulator_H(self, d):
        H_k = np.ones((self.N, 1))
        for k in range(self.D-1, d, -1):
            Wk = self.W.cores[k].unfold(1).T
            mean_WW = np.kron(Wk, Wk)
            block_shape = (self.TT_ranks[k+1]*self.dims[k], self.TT_ranks[k])
            covariance_WW = outer2block(self.Sigma[k], block_shape, mean_WW.shape)
            expectation_WW = np.asarray(mean_WW + covariance_WW)
            H_k = khatri_rao(H_k, khatri_rao(self.feature_map[k], self.feature_map[k])) @ expectation_WW
        return H_k
