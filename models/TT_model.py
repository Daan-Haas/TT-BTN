from math import gamma
import matplotlib.pyplot as plt
import numpy as np

from kernels import *
from utils import khatri_rao, unfold
from tqdm import tqdm, trange

class BTTKM:
    def __init__(self, nr_cores, R, M, kernel):
        if (len(M) != nr_cores) or (len(R) != nr_cores+1):
            raise ValueError(f"Incorrect number of dimensions or TT ranks, "
                             f"TT with {nr_cores} cores should have {nr_cores} dimensions and {nr_cores+1} ranks")
        if (R[0], R[-1]) != (1,1):
            raise ValueError("First and last TT rank should be 1")

        self.R = R
        self.M = M
        self.D = nr_cores
        self.init_cores()
        size = [self.R[d]*self.M[d]*self.R[d+1] for d in range(self.D)]
        self.Sigma = [0.1*np.eye(size[d]) for d in range(self.D)]
        self.var = [np.ones(size[d]) for d in range(self.D)]
        self.kernel = kernel
        self.a_N = 1e-2
        self.b_N = 1e-2

    def init_cores(self):
        core_list = [np.random.standard_normal([self.R[i],self.M[i],self.R[i+1]]) for i in range(self.D)]
        self.W = core_list

    def train(self,
              X, Y,
              a_0=1, b_0=1,
              c_0=None, d_0=None,
              g_0=None, h_0=None,
              lambda_update=False,
              delta_update=False,
              convergence_bound=1e-4,
              max_iter=100,
              rank_pruning=False,
              plotting=False):
        print("Training")
        self.feature_map = self.kernel(X, max(self.M))
        self.N = X.shape[0]
        self.a_N = a_0
        self.b_N = b_0
        self.expectation_tau = a_0 / b_0
        rank_tol = 1e-4

        self.a_N = a_0
        self.b_N = b_0

        if c_0 is None:
            c_0 = [1e-6 * np.ones(self.R[d]) for d in range(self.D+1)]
        self.c_N = c_0.copy()

        if d_0 is None:
            d_0 = [1e-6 * np.ones(self.R[d]) for d in range(self.D+1)]
        self.d_N = d_0.copy()

        self.lambda_R = [c_0[d]/d_0[d] for d in range(self.D+1)]

        if g_0 is None:
            g_0 = [1e-6 * np.ones(self.M[d]) for d in range(self.D)]
        self.g_N = g_0.copy()

        if h_0 is None:
            h_0 = [1e-6 * np.ones(self.M[d]) for d in range(self.D)]
        self.h_N = h_0.copy()

        self.delta = [g_0[d]/h_0[d] for d in range(self.D)]

        collapsed = False

        ELBO = [-np.inf]
        it = 0
        pbar = trange(max_iter, desc="Running", leave=True)
        for it in pbar:
            # cores update
            # Calculate cores once
            H_d = np.ones((self.N, 1))
            G_d = np.ones((self.N, 1))
            H_gt = []
            G_gt = []
            for d in range(self.D,0,-1):
                H_gt.insert(0, H_d)
                G_gt.insert(0, G_d)
                H_d = self.backward_H_one_step(H_d, d-1)
                G_d = self.backward_G_one_step(G_d, d-1)

            self.H_lt = np.ones((self.N, 1))
            self.G_lt = np.ones((self.N, 1))
            for d in range(self.D):
                W_norm = 0

                temp1 = khatri_rao(self.H_lt, self.feature_map[d])
                temp2 = khatri_rao(self.feature_map[d], H_gt[d])
                H_d = temp1.T @ temp2

                H_d = H_d.reshape([self.M[d], self.R[d], self.R[d], self.R[d+1], self.R[d+1], self.M[d]], order='F')
                H_d = H_d.transpose([1,0,3,2,5,4])
                H_d = H_d.reshape([self.R[d]*self.M[d]*self.R[d+1], self.R[d]*self.M[d]*self.R[d+1]], order='F')

                if np.max(abs(H_d.T - H_d))/np.linalg.norm(H_d) > 1e-6:
                    print(f"H_d not symmetrical: {H_d}")
                    break

                temp1 = khatri_rao(self.feature_map[d], self.G_lt)
                G_d = khatri_rao(G_gt[d], temp1)

                lambda_mat_next = np.diag(self.lambda_R[d+1])
                lambda_mat_prev = np.diag(self.lambda_R[d])
                delta_mat = np.diag(self.delta[d])

                variance_term = np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)
                self.Sigma[d] = np.linalg.pinv(np.add(self.expectation_tau*H_d, variance_term))
                self.var[d] = np.diag(self.Sigma[d])

                vectorized_W = self.expectation_tau*self.Sigma[d]@G_d.T@Y
                self.W[d] = vectorized_W.reshape((self.R[d], self.M[d], self.R[d+1]), order='F')

                self.H_lt = self.forward_H_one_step(self.H_lt, d)
                self.G_lt = self.forward_G_one_step(self.G_lt, d)
                W_norm += np.max(np.linalg.norm(self.W[d]))

            # # posterior update delta
            if delta_update:
                for d in range(self.D):
                    self.g_N[d] = g_0[d] + 0.5 * (self.R[d] + self.R[d + 1])
                    W_2 = unfold(self.W[d], 2)

                    tensor_shape = (self.R[d], self.M[d], self.R[d + 1])

                    variance_tensor_d = self.var[d].reshape(tensor_shape, order='F', copy=False)
                    variance_matrix_d = np.vstack([variance_tensor_d[:, i, :].reshape(1, -1).flatten() for i in
                                                   range(variance_tensor_d.shape[1])])

                    sparsity_term = np.kron(np.diag(self.lambda_R[d + 1]), np.diag(self.lambda_R[d]))
                    variance_term = variance_matrix_d @ np.kron(self.lambda_R[d + 1], self.lambda_R[d])

                    expectation = np.add(np.diag(W_2 @ sparsity_term @ W_2.T), variance_term)
                    self.h_N[d] = np.add(h_0[d], 0.5 * expectation)

                    self.delta[d] = self.g_N[d] / self.h_N[d]
                    self.delta[d][self.delta[d] < 1e-5] = 1e-5

            # posterior update lambda
            if lambda_update:
                for d in range(1, self.D):
                    self.c_N[d] = c_0[d] + 0.5 * (self.R[d-1]*self.M[d-1]+self.R[d+1]*self.M[d])

                    W_3 = unfold(self.W[d-1], 3)
                    core = self.W[d]
                    W_1 = np.vstack([core[i,:,:].T.reshape(1,-1).flatten() for i in range(core.shape[0])])

                    Delta_d_1 = np.diag(self.delta[d-1])
                    Delta_d = np.diag(self.delta[d])
                    Lambda_d_min_1 = np.diag(self.lambda_R[d-1])
                    Lambda_d_plus_1 = np.diag(self.lambda_R[d+1])

                    expectation1_term1 = np.diag(W_3@np.kron(Delta_d_1, Lambda_d_min_1)@W_3.T)

                    V_d = self.var[d-1].reshape(self.R[d-1], self.M[d-1], self.R[d], order='F', copy=False)
                    V_d_3 = np.vstack([V_d[:,:,i].T.reshape(1,-1).flatten() for i in range(V_d.shape[2])])

                    expectation1_term2 = V_d_3@np.kron(self.delta[d-1], self.lambda_R[d-1])
                    expectation1 = expectation1_term1 + expectation1_term2
                    expectation2_term1 = np.diag(W_1@np.kron(Lambda_d_plus_1, Delta_d)@W_1.T)

                    V_d = self.var[d].reshape(self.R[d], self.M[d], self.R[d+1], order='F', copy=False)
                    V_d_1 = np.vstack([V_d[i,:,:].T.reshape(1,-1).flatten() for i in range(V_d.shape[0])])

                    expectation2_term2 = V_d_1@np.kron(self.lambda_R[d+1], self.delta[d])
                    expectation2 = expectation2_term1 + expectation2_term2

                    self.d_N[d] = np.add(d_0[d], 0.5 * (expectation1 + expectation2))

                    self.lambda_R[d] = self.c_N[d] / self.d_N[d]
                    self.lambda_R[d][self.lambda_R[d] < 1e-5] = 1e-5

                    if rank_pruning:
                        Wall = unfold(self.W[d], 1)
                        comPower = np.diag(Wall @ Wall.T)
                        if np.sum(comPower) == 0:
                            rankest = self.R[d]
                        else:
                            var_explained = comPower / np.sum(comPower) * 100
                            rankest = np.sum(var_explained > rank_tol)
                        if rankest == 0:
                            collapsed = True
                            break

                        if self.R[d] != rankest:
                            print(f"pruning rank of {d}-th core from {self.R[d]} to {rankest}")
                            pruning_mask = [var_explained > rank_tol]
                            pruning_mask = pruning_mask[0]
                            self.W[d] = self.W[d][pruning_mask]
                            self.W[d-1] = self.W[d-1][:,:,pruning_mask]
                            Sigma_d_tensor = self.Sigma[d].reshape([self.R[d], self.M[d], self.R[d+1], self.R[d], self.M[d], self.R[d+1]], order='F', copy=False)
                            Sigma_d_tensor = np.delete(Sigma_d_tensor, ~pruning_mask, axis=0)
                            Sigma_d_tensor = np.delete(Sigma_d_tensor, ~pruning_mask, axis=3)
                            Sigma_d_min_tensor = self.Sigma[d-1].reshape(self.R[d-1], self.M[d-1], self.R[d], self.R[d-1], self.M[d-1], self.R[d], order='F', copy=False)
                            Sigma_d_min_tensor = np.delete(Sigma_d_min_tensor, ~pruning_mask, axis=2)
                            Sigma_d_min_tensor = np.delete(Sigma_d_min_tensor, ~pruning_mask, axis=5)
                            covar_d_tensor = self.var[d].reshape([self.R[d], self.M[d], self.R[d+1]], order='F', copy=False)
                            covar_d_tensor = np.delete(covar_d_tensor, ~pruning_mask, axis=0)
                            covar_d_min_tensor = self.var[d-1].reshape([self.R[d-1], self.M[d], self.R[d]], order='F', copy=False)
                            covar_d_min_tensor = np.delete(covar_d_min_tensor, ~pruning_mask, axis=2)
                            self.R[d] = sum(pruning_mask)
                            self.lambda_R[d] = self.lambda_R[d][pruning_mask]
                            self.Sigma[d] = Sigma_d_tensor.reshape([self.R[d]*self.M[d]*self.R[d+1], self.R[d]*self.M[d]*self.R[d+1]], order='F')
                            self.Sigma[d-1] = Sigma_d_min_tensor.reshape([self.R[d-1]*self.M[d-1]*self.R[d], self.R[d-1]*self.M[d-1]*self.R[d]], order='F')
                            self.var[d] = covar_d_tensor.reshape(-1)
                            self.var[d-1] = covar_d_min_tensor.reshape(-1)
                            c_0[d] = c_0[d][pruning_mask]
                            d_0[d] = d_0[d][pruning_mask]
                            self.c_N[d] = self.c_N[d][pruning_mask]
                            self.d_N[d] = self.d_N[d][pruning_mask]

            if collapsed:
                print(f"rank of {d}th core went to 0")
                break

            predictions, _ = self.predict(X)
            error_term = 0.5*self.expectation_tau* np.linalg.norm(Y.reshape((self.N,1), copy=False) - predictions)**2

            L2_norms = 0
            ln_q_W = 0
            lambda_term = 0
            delta_term = 0

            #ELBO calculation
            for d in range(self.D):
                lambda_mat_next = np.diag(self.lambda_R[d + 1])
                lambda_mat_prev = np.diag(self.lambda_R[d])
                delta_mat = np.diag(self.delta[d])
                variance_term = np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)

                L2_norms += np.trace(variance_term @ (np.outer(self.W[d].reshape((-1,1)), self.W[d].reshape((-1,1)))
                                                      + np.diag(self.var[d])))
                ln_q_W += 0.5*np.log(np.linalg.norm(self.Sigma[d])) + (self.R[d]*self.M[d]/2)*(1+np.log(2*np.pi))
                for r in range(self.R[d]):
                    lambda_term += ((1-np.log(self.d_N[d][r])
                                    - (d_0[d][r]/self.d_N[d][r]))*self.c_N[d][r]
                                    + np.log(gamma(self.c_N[d][r])))
                for m in range(self.M[d]):
                    delta_term += (np.log(gamma(self.g_N[d][m]))
                                   + (1-np.log(self.h_N[d][m])
                                   - (h_0[d][m]/self.h_N[d][m]))*self.g_N[d][m])

            tau_term = (1 - np.log(self.b_N) - (b_0/self.b_N))*self.a_N + np.log(gamma(self.a_N))
            ELBO.append(-error_term - L2_norms - ln_q_W - lambda_term - delta_term - tau_term)

            if it > 2:
                LB_rel_chan = (ELBO[-1] - ELBO[-2])/ELBO[2]
            else:
                LB_rel_chan = np.nan
            del predictions, tau_term
            it += 1

            # Display progress for every iteration
            pbar.set_postfix(
                {
                    "rel chan": f"{LB_rel_chan:.4f}",
                    "Fit": f"{ELBO[-1]:.4f}",
                    "err": f"{error_term:.1e}",
                }
            )

            if abs(LB_rel_chan) < convergence_bound:
                # noise precision update
                self.a_N = a_0 + self.N / 2
                Expectation_G = self.forward_accumulator_G(self.D)
                Expectation_H = np.ones(self.N).T @ self.forward_accumulator_H(self.D)
                frob_errors = np.linalg.norm(Y) ** 2 - 2 * (Y.T @ Expectation_G).squeeze() + Expectation_H.squeeze()
                self.b_N = b_0 + 0.5 * frob_errors
                self.expectation_tau = self.a_N / self.b_N
                print("Convergence bound reached, exiting")
                break
            if W_norm < 1e-100 or np.isnan(W_norm):
                print("model collapsed")
                break

        if it == max_iter:
            # noise precision update
            self.a_N = a_0 + self.N / 2
            Expectation_G = self.forward_accumulator_G(self.D)
            Expectation_H = np.ones(self.N).T @ self.forward_accumulator_H(self.D)
            frob_errors = np.linalg.norm(Y) ** 2 - 2 * (Y.T @ Expectation_G).squeeze() + Expectation_H.squeeze()
            self.b_N = b_0 + 0.5 * frob_errors
            self.expectation_tau = self.a_N / self.b_N
            print("Iteration limit reached, exiting")

        if plotting:
            fig, ax1 = plt.subplots()
            ax1.plot(ELBO[1:], label='ELBO')
            ax1.set_title("Training ELBO")
            ax1.set_xlabel("iteration")
            ax1.set_ylabel("ELBO")
            fig.legend()
            plt.show()

    def predict(self, X, classification=False):
        self.feature_map = self.kernel(X, max(self.M))
        self.N = self.feature_map.shape[1]
        predicted_mean = self.forward_accumulator_G(self.D)

        sum_matrix = 0
        for d in range(self.D):
            G_lt = khatri_rao(self.feature_map[d], self.forward_accumulator_G(d))
            G_d = khatri_rao(self.backward_accumulator_G(d), G_lt)
            GsigmaG = G_d @ self.Sigma[d] @ G_d.T
            sum_matrix += GsigmaG
        S = ((2 * self.a_N / ( 2 * self.a_N))
             * ((self.b_N / self.a_N) + (sum_matrix/(self.D * self.R[d] * self.M[d] * self.R[d+1]))))
        std = np.sqrt(np.diag(S))
        return predicted_mean, std

    def forward_accumulator_G(self, d):
        G_k = np.ones((self.N, 1))
        for k in range(d):
            G_k = khatri_rao(self.feature_map[k], G_k) @ unfold(self.W[k], 3).T
        return G_k

    def backward_accumulator_G(self, d):
        G_k = np.ones((self.N, 1)) # N x 1
        for k in range(self.D-1, d, -1):
            G_k = khatri_rao(self.feature_map[k], G_k) @ unfold(self.W[k], 1).T
        return G_k # N x R_d

    def forward_accumulator_H(self, d):
        H_k = np.ones((self.N, 1))
        for k in range(d):
            Wk = unfold(self.W[k], 3).T
            mean_WW = np.kron(Wk, Wk)
            mean_WW = mean_WW.reshape([self.R[k],self.M[k],self.R[k],self.M[k],self.R[k+1]**2], order='F')
            mean_WW = np.transpose(mean_WW,(0,2,1,3,4))
            mean_WW = mean_WW.reshape([(self.R[k]*self.M[k])**2, self.R[k+1]**2], order='F')

            covariance_shape = (self.R[k], self.M[k], self.R[k+1], self.R[k], self.M[k], self.R[k+1])
            covariance_WW = self.Sigma[k].reshape(covariance_shape, order='F')
            covariance_WW = np.transpose(covariance_WW, [0,3, 1,4, 2,5])
            covariance_WW = covariance_WW.reshape([(self.R[k]*self.M[k])**2, self.R[k+1]**2], order='F')

            expectation_WW = np.add(mean_WW, covariance_WW)

            H_k = khatri_rao(khatri_rao(self.feature_map[k], self.feature_map[k]), H_k) @ expectation_WW
        return H_k

    def backward_accumulator_H(self, d):
        H_k = np.ones((self.N, 1))
        for k in range(self.D-1, d, -1):
            Wk = unfold(self.W[k], 1).T
            mean_WW = np.kron(Wk, Wk)
            mean_WW = mean_WW.reshape([self.R[k+1], self.M[k], self.R[k+1], self.M[k], self.R[k]**2], order='F')

            mean_WW = np.transpose(mean_WW, [0,2,1,3,4])
            mean_WW = mean_WW.reshape([(self.R[k+1]*self.M[k])**2, self.R[k]**2], order='F')

            covariance_shape = (self.R[k], self.M[k], self.R[k+1], self.R[k], self.M[k], self.R[k+1])
            covariance_WW = self.Sigma[k].reshape(covariance_shape, order='F')
            covariance_WW = np.transpose(covariance_WW, [2,5,1,4,0,3])
            covariance_WW = covariance_WW.reshape([(self.M[k]*self.R[k+1])**2, self.R[k]**2], order='F')

            expectation_WW = np.add(mean_WW, covariance_WW)

            H_k = khatri_rao(khatri_rao(self.feature_map[k], self.feature_map[k]), H_k) @ expectation_WW
        return H_k

    def forward_H_one_step(self, H_lt, d):
        Wk = unfold(self.W[d], 3).T
        mean_WW = np.kron(Wk, Wk)
        mean_WW = mean_WW.reshape([self.R[d],self.M[d],self.R[d],self.M[d],self.R[d+1]**2], order='F')
        mean_WW = np.transpose(mean_WW,(0,2,1,3,4))
        mean_WW = mean_WW.reshape([(self.R[d]*self.M[d])**2, self.R[d+1]**2], order='F')

        covariance_shape = (self.R[d], self.M[d], self.R[d+1], self.R[d], self.M[d], self.R[d+1])
        covariance_WW = self.Sigma[d].reshape(covariance_shape, order='F')
        covariance_WW = np.transpose(covariance_WW, [0,3, 1,4, 2,5])
        covariance_WW = covariance_WW.reshape([(self.R[d]*self.M[d])**2, self.R[d+1]**2], order='F')

        expectation_WW = np.add(mean_WW, covariance_WW)
        H_k = khatri_rao(khatri_rao(self.feature_map[d], self.feature_map[d]), H_lt) @ expectation_WW
        return H_k

    def backward_H_one_step(self, H_gt, d):
        Wk = unfold(self.W[d], 1).T
        mean_WW = np.kron(Wk, Wk)
        mean_WW = mean_WW.reshape([self.R[d + 1], self.M[d], self.R[d + 1], self.M[d], self.R[d] ** 2], order='F')

        mean_WW = np.transpose(mean_WW, [0, 2, 1, 3, 4])
        mean_WW = mean_WW.reshape([(self.R[d + 1] * self.M[d]) ** 2, self.R[d] ** 2], order='F')

        covariance_shape = (self.R[d], self.M[d], self.R[d + 1], self.R[d], self.M[d], self.R[d + 1])
        covariance_WW = self.Sigma[d].reshape(covariance_shape, order='F')
        covariance_WW = np.transpose(covariance_WW, [2, 5, 1, 4, 0, 3])
        covariance_WW = covariance_WW.reshape([(self.M[d] * self.R[d + 1]) ** 2, self.R[d] ** 2], order='F')

        expectation_WW = np.add(mean_WW, covariance_WW)

        H_k = khatri_rao(khatri_rao(self.feature_map[d], self.feature_map[d]), H_gt) @ expectation_WW
        return H_k

    def forward_G_one_step(self, G_lt, d):
        G_k = khatri_rao(self.feature_map[d], G_lt) @ unfold(self.W[d], 3).T
        return G_k

    def backward_G_one_step(self, G_gt, d):
        G_k = khatri_rao(self.feature_map[d], G_gt) @ unfold(self.W[d], 1).T
        return G_k