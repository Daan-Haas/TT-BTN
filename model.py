import numpy as np
import torch
from torch import Tensor

from utils import *

class Core:
    def __init__(self, data):
        self.core = data

    def __repr__(self):
        return str(self.core)

    def unfold(self, mode):
        if mode == 1:
            return self.core.permute([2,0,1]).reshape(self.core.size(dim=2),self.core.size(dim=0)*self.core.size(dim=1))
        elif mode == 2:
            return self.core.reshape(self.core.size(dim=0),self.core.size(dim=1)*self.core.size(dim=2))
        elif mode == 3:
            return self.core.permute([2,1,0]).reshape(self.core.size(dim=2),self.core.size(dim=1)*self.core.size(dim=0))
        else:
            raise ValueError("Unsupported mode {}, please select mode from [1,2,3]".format(mode))

    def size(self, dims):
        return self.core.size(dims)

class BTTKM:
    def __init__(self, nr_cores, ranks, dims):

        if (len(dims) != nr_cores) or (len(ranks) != nr_cores+1):
            raise ValueError(f"Incorrect number of dimensions or TT ranks, TT with {nr_cores} cores should have {nr_cores} dimensions and {nr_cores+1} ranks")
        if (ranks[0], ranks[-1]) != (1,1):
            raise ValueError("First and last TT rank should be 1")

        self.TT_ranks = ranks
        self.dims = dims
        self.nr_cores = nr_cores
        self.init_cores()
        self.tau_shape_parameter = 1
        self.tau_scale_parameter = 1
        self.delta_shape_parameter = [torch.ones(dims[i]) for i in range(nr_cores)]
        self.delta_scale_parameter = [torch.ones(dims[i]) for i in range(nr_cores)]
        self.lambda_shape_parameter = [torch.ones(ranks[i+1]) for i in range(nr_cores)]
        self.lambda_scale_parameter = [torch.ones(ranks[i+1]) for i in range(nr_cores)]

    def init_cores(self):
        self.W = TensorTrain([Core(torch.rand(self.TT_ranks[i],self.dims[i],self.TT_ranks[i+1])) for i in range(self.nr_cores)])

    def train(self, X, Y, c_0=None, d_0=None, g_0=None, h_0=None, a_0=1, b_0=1, error_bound=1e-4, iteration_limit=100):
        self.feature_map = Kernel(X, self.nr_cores)
        self.N = X.shape[0]
        self.D = X.shape[1]
        expectation_tau = self.tau_shape_parameter / self.tau_scale_parameter

        c_0r = (c_0 if c_0 else np.ones(self.TT_ranks[0]))
        c_0 = [c_0r for _ in range(self.nr_cores)]
        self.c_N = c_0
        d_0r = (d_0 if d_0 else np.ones(self.TT_ranks[0]))
        d_0 = [d_0r for _ in range(self.nr_cores)]
        self.d_N = d_0

        g_0r = (g_0 if g_0 else np.ones(self.TT_ranks[0]))
        g_0 = [g_0r for _ in range(self.nr_cores)]
        self.g_N = g_0
        h_0r = (h_0 if h_0 else np.ones(self.TT_ranks[0]))
        h_0 = [h_0r for _ in range(self.nr_cores)]
        self.h_N = h_0

        self.lambda_R = [c_0[d]/d_0[d] for d in range(len(c_0))]
        self.delta_M = [g_0[d]/h_0[d] for d in range(len(g_0))]

        error = np.inf
        it = 0

        while (error > error_bound) and (it < iteration_limit):

            # cores update
            for d in range(self.nr_cores):
                if d == 0:
                    print(self.feature_map[d].shape, self.backward_accumulator_H(d).shape)
                    GTG_d = self.feature_map[d] @ khatri_rao(self.feature_map[d].T, self.backward_accumulator_H(d))
                elif d == self.nr_cores:
                    GTG_d = khatri_rao(self.forward_accumulator_H(d), self.feature_map[d]).T @ self.feature_map[d]
                else:
                    GTG_d = khatri_rao(self.forward_accumulator_H(d), self.feature_map[d]).T @ khatri_rao(self.feature_map[d], self.backward_accumulator_H(d))
                self.Sigma[d] = np.inv(expectation_tau@GTG_d + Kronecker(Kronecker(self.expectation_lambda(d+1), self.expectation_delta(d)), self.expectation_lambda(d)))
                G_d = khatri_rao(khatri_rao(self.forward_accumulator_G(d), self.feature_map[d]), self.backward_accumulator_G(d))
                self.W.cores[d].core = expectation_tau*self.Sigma[d]@G_d@Y

            # posterior update lambda
            for d in range(1, self.nr_cores):
                self.c_N[d] = c_0[d] + 0.5 * (self.TT_ranks[d-1]*self.dims[d-1]+self.TT_ranks[d+1]*self.dims[d])
                W_unfold_3 = self.W.cores[d-1].unfold(3)
                W_unfold_1 = self.W.cores[d].unfold(1)

                variance_tensor_prev = torch.reshape(torch.tensor(np.diag(self.Sigma[d-1])), (self.TT_ranks[d-1], self.dims[d-1], self.TT_ranks[d]))
                variance_matrix_prev = variance_tensor_prev.permute([2,1,0]).reshape(variance_tensor_prev.size(dim=2),variance_tensor_prev.size(dim=1)*variance_tensor_prev.size(dim=0))
                expectation1 = np.diag(W_unfold_3 @ (Kronecker(np.diag(self.delta_M[d-1]), np.diag(self.lambda_R[d-1]))) @ W_unfold_3.T) + variance_matrix_prev@Kronecker(self.delta_M[d-1], self.lambda_R[d-1])

                variance_tensor_next = torch.reshape(torch.tensor(np.diag( self.Sigma[d])), (self.TT_ranks[d], self.dims[d], self.TT_ranks[d+1]))
                variance_matrix_next = variance_tensor_next.permute([2,0,1]).reshape(variance_tensor_next.size(dim=2),variance_tensor_next.size(dim=0)*variance_tensor_next.size(dim=1))
                expectation2 = np.diag(W_unfold_1 @ Kronecker(np.diag(self.lambda_R[d+1]), np.diag(self.delta_M[d])) @ W_unfold_1.T) + variance_matrix_next@Kronecker(self.lambda_R[d+1], self.delta_M[d])

                self.d_N[d] = d_0[d] + 0.5 * (expectation1 + expectation2)

            # posterior update delta
            for d in range(1, self.nr_cores):
                self.g_N[d] = g_0[d] + 0.5 * (self.TT_ranks[d] + self.TT_ranks[d+1])
                W_unfold_2 = self.W[d].unfold(2)

                variance_tensor_d = torch.reshape((torch.tensor(np.diag(self.Sigma[d])), (self.TT_ranks[d], self.dims[d], self.TT_ranks[d+1])))
                variance_matrix_d = variance_tensor_d.reshape([variance_tensor_d.size(dim=0),variance_tensor_d.size(dim=1), variance_tensor_d.size(dim=2)])
                expectation = np.diag(W_unfold_2 @ Kronecker(np.diag(self.lambda_R[d+1]), np.diag(self.lambda_R[d])) @ W_unfold_2.T) + variance_matrix_d @ Kronecker(self.lambda_R[d+1], self.lambda_R[d])
                self.h_N = h_0[d] + 0.5 * expectation

            # noise precision update
            self.a_N = a_0 + X.size(dims=1) / 2
            Expectation_G = self.forward_accumulator_G(self.nr_cores+1)
            Expectation_H = np.ones(X.size(dims=1)).T @ self.forward_accumulator_H(self.nr_cores+1)
            error = np.linalg.norm(Y)^2 - 2*Y.T@Expectation_G + Expectation_H
            self.b_N = b_0 + 0.5 * error

        it += 1


    def forward_accumulator_G(self, d):
        G_k = np.array([[1]])
        for k in range(1, d):
            G_k = khatri_rao(G_k, self.feature_map[k-1]) @ self.W.cores[k-1].unfold(3)
        return G_k

    def backward_accumulator_G(self, d):
        G_k = np.array([[1]])
        for k in range(self.nr_cores, d, -1):
            G_k = khatri_rao(G_k, self.feature_map[k+1]) @ self.W.cores[k+1].unfold(3)
        return G_k

    def forward_accumulator_H(self, d):
        H_k = np.array([[1]])
        for k in range(d):
            covariance_WW = np.cov(np.outer(self.W.cores[k-1].unfold(3).reshape(-1,1),self.W.cores[k-1].unfold(3).reshape(-1,1)))
            expectation_WW = (Kronecker(self.W.cores[k-1].unfold(3), self.W.cores[k-1].unfold(3))
                              + covariance_WW.reshape((self.TT_ranks[k-1]*self.dims[k-1])**2, self.TT_ranks[k]**2))
            H_k = khatri_rao(H_k, khatri_rao(self.feature_map[k-1], self.feature_map[k-1])) @ expectation_WW
        return H_k

    def backward_accumulator_H(self, d):
        H_k = np.ones((1, self.D**2))
        for k in range(self.nr_cores-2, d, -1):
            covariance_WW = np.cov(np.outer(self.W.cores[k+1].unfold(3).reshape(-1,1),self.W.cores[k+1].unfold(3).reshape(-1,1)))
            covariance_WW = covariance_WW.reshape(((self.TT_ranks[k+2]*self.dims[k+1])**2, self.TT_ranks[k+1]**2))
            mean_WW = Kronecker(self.W.cores[k+1].unfold(3), self.W.cores[k+1].unfold(3).T)
            mean_WW = mean_WW.reshape(((self.TT_ranks[k+2]*self.dims[k+1])**2, self.TT_ranks[k+1]**2))
            Expectation_WW = mean_WW + covariance_WW
            H_k = khatri_rao(H_k, row_khatri_rao(self.feature_map[k+1], self.feature_map[k+1]).T).T @ np.asarray(Expectation_WW)
        return H_k

    def expectation_delta(self, d: int) -> Tensor:
        return torch.div(self.delta_shape_parameter[d], self.delta_scale_parameter[d])

    def expectation_lambda(self, d: int) -> Tensor:
        return torch.div(self.lambda_shape_parameter[d], self.lambda_scale_parameter[d])


class TensorTrain:
    def __init__(self, cores: list[Core]):
        self.cores = cores
        for d in range(len(self.cores)-1):
            if not isinstance(self.cores[d], Core):
                raise ValueError(f"Core {d} is not a Core")
            if self.cores[d].size(dims=2) != self.cores[d+1].size(dims=0):
                raise ValueError(f"Dimensions of cores do not match, between cores {d} and {d+1}")

        self.TT_ranks = [core.size(0) for core in cores]
        self.TT_ranks.append(1)
        self.dims = [core.size(1) for core in cores]

    def construct(self):
        pass

    def TT_inner(self, other):
        if not isinstance(other, TensorTrain):
            raise NotImplementedError

        pass