from math import gamma
import matplotlib.pyplot as plt
import numpy as np

from kernels import *
from utils import khatri_rao, unfold
from tqdm import tqdm, trange

class BTTKM:
    def __init__(self, nr_cores, ranks, M, kernel):
        if (len(M) != nr_cores) or (len(ranks) != nr_cores+1):
            raise ValueError(f"Incorrect number of dimensions or TT ranks, "
                             f"TT with {nr_cores} cores should have {nr_cores} dimensions and {nr_cores+1} ranks")
        if (ranks[0], ranks[-1]) != (1,1):
            raise ValueError("First and last TT rank should be 1")

        self.R = ranks
        self.M = M
        self.D = nr_cores
        self.init_cores()
        size = [self.R[d]*self.M[d]*self.R[d+1] for d in range(self.D)]
        self.Sigma = [np.eye(size[d]) for d in range(self.D)]
        self.covar = [np.ones(size[d]) for d in range(self.D)]
        self.kernel = kernel

    def init_cores(self):
        core_list = [np.random.rand(self.R[i],self.M[i],self.R[i+1]) for i in range(self.D)]
        self.W = core_list

    def train(self,
              X, Y,
              a_0=1, b_0=1,
              c_0=None, d_0=None,
              g_0=None, h_0=None,
              lambda_update=False,
              delta_update=False,
              tau_update=False,
              error_bound=1e-4, iteration_limit=1000,
              rank_pruning=False,
              feature_pruning=False,
              early_stopping=False,
              printing=True,
              plotting=True):
        print("Training")
        self.feature_map = self.kernel(X, max(self.M))
        self.N = X.shape[0]
        self.a_N = a_0
        self.b_N = b_0
        self.expectation_tau = a_0 / b_0

        rank_tol = 1e-5
        c_0 = [1e-6 * np.ones(self.R[d]) for d in range(self.D+1)]
        self.c_N = c_0.copy()
        d_0 = [1e-6 * np.ones(self.R[d]) for d in range(self.D+1)]
        self.d_N = d_0.copy()
        self.lambda_R = [c_0[d]/d_0[d] for d in range(self.D+1)]
        lam_norm = []

        g_0 = [1 * np.ones(self.M[d]) for d in range(self.D)]
        self.g_N = g_0.copy()
        h_0 = [1 * np.ones(self.M[d]) for d in range(self.D)]
        self.h_N = h_0.copy()
        self.delta = [g_0[d]/h_0[d] for d in range(self.D)]
        del_norm = []

        W_norm = []
        collapsed = False

        ELBO = [-np.inf]
        it = 0
        pbar = trange(iteration_limit, desc="Running", leave=True)
        for it in pbar:
            # cores update
            for d in range(self.D):
                H_lt = khatri_rao(self.forward_accumulator_H(d), self.feature_map[d]) # N x M_d R_d**2
                H_gt = khatri_rao(self.feature_map[d], self.backward_accumulator_H(d)) # N x R_{d+1}**2 M_d
                H_d_mat = H_lt.T @ H_gt # M_d R_d**2 x R_{d+1}**2 M_d
                H_d_tens = H_d_mat.reshape([self.M[d], self.R[d], self.R[d], self.R[d+1], self.R[d+1], self.M[d]], order='F')
                # M_d x R_d x R_d x R_{d+1} x R_{d+1} x M_d
                H_d_tens_trans = H_d_tens.transpose([1,0,3,2,5,4])  # R_d x M_d x R_{d+1} x R_d x M_d x R_d+1
                H_d = H_d_tens_trans.reshape([self.R[d]*self.M[d]*self.R[d+1], self.R[d]*self.M[d]*self.R[d+1]], order='F')
                # R_d M_d R_{d+1} x R_d M_d R_d+1
                assert np.max(abs(H_d.T - H_d))/np.linalg.norm(H_d) < 1e-6, f"H_d not symmetrical: {H_d}"

                G_lt = khatri_rao(self.feature_map[d], self.forward_accumulator_G(d)) # N x R_d M_d
                G_d = khatri_rao(self.backward_accumulator_G(d), G_lt) # N x R_d M_d R_{d+1}

                lambda_mat_next = np.diag(self.lambda_R[d+1]) # R_{d+1} x R_{d+1}
                lambda_mat_prev = np.diag(self.lambda_R[d]) # R_d x R_d
                delta_mat = np.diag(self.delta[d]) # M_d x M_d
                variance_term = np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev) # R_d M_d R_{d+1} x R_d M_d R_{d+1}
                self.Sigma[d] = np.linalg.inv(np.add(self.expectation_tau*H_d, variance_term)) # R_d M_d R_{d+1} x R_d M_d R_{d+1}
                self.covar[d] = np.diag(self.Sigma[d])
                vectorized_W = self.expectation_tau*self.Sigma[d]@G_d.T@Y.reshape(-1,1)
                self.W[d] = vectorized_W.reshape((self.R[d], self.M[d], self.R[d+1]), order='F')
            W_norm.append(np.linalg.norm(self.W[d]))
                # print(f"||W|| = {np.linalg.norm(vectorized_W[d])}")
                # print(f"||G|| = {np.linalg.norm(G_d)}")
                # print(f"||H|| = {np.linalg.norm(H_d)}")
                # print(f"||Sigma|| = {np.linalg.norm(self.Sigma[d])}")
            # for d, core in enumerate(self.W.cores):
            #     core.core = vectorized_W[d].reshape((self.R[d], self.M[d], self.R[d+1]), order='C')

            # # posterior update delta
            if delta_update:
                for d in range(self.D):
                    self.g_N[d] = g_0[d] + 0.5 * (self.R[d] + self.R[d + 1])
                    W_2 = unfold(self.W[d], 2)

                    tensor_shape = (self.R[d], self.M[d], self.R[d + 1])

                    # lambda_mat_next = np.diag(self.lambda_R[d + 1])  # R_{d+1} x R_{d+1}
                    # lambda_mat_prev = np.diag(self.lambda_R[d])  # R_d x R_d
                    # delta_mat = np.diag(self.delta[d])  # M_d x M_d
                    # temp = np.linalg.solve(self.Sigma[d], np.eye(self.R[d]*self.M[d]*self.R[d + 1])) - np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)
                    # temp += np.kron(np.kron(lambda_mat_next, np.eye(self.M[d])), lambda_mat_prev)
                    # temp = np.linalg.solve(temp, np.eye(temp.shape[0]))
                    # variance_tensor_d = np.diag(temp).reshape(tensor_shape, order='F')

                    variance_tensor_d = self.covar[d].reshape(tensor_shape, order='F', copy=False)

                    variance_matrix_d = np.vstack([variance_tensor_d[:, i, :].reshape(1, -1).flatten() for i in
                                                   range(variance_tensor_d.shape[1])])
                    sparsity_term = np.kron(np.diag(self.lambda_R[d + 1]), np.diag(self.lambda_R[d]))
                    variance_term = variance_matrix_d @ np.kron(self.lambda_R[d + 1], self.lambda_R[d])
                    expectation = np.add(np.diag(W_2 @ sparsity_term @ W_2.T), variance_term)
                    self.h_N[d] = np.add(h_0[d], 0.5 * expectation)

                    self.delta[d] = self.g_N[d] / self.h_N[d]
                    if feature_pruning:
                        pruning_mask = self.delta[d] > 100
                        if any(pruning_mask):
                            self.W[d] = self.W[d][:, ~pruning_mask, :]
                            self.M[d] = sum(~pruning_mask)
                            indeces = np.where(~pruning_mask)[0].tolist()
                            self.feature_map[d] = self.feature_map[d][:][indeces]

                del_norm.append(np.linalg.norm(self.h_N))
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

                    lambda_mat_next = np.diag(self.lambda_R[d])  # R_{d+1} x R_{d+1}
                    lambda_mat_prev = np.diag(self.lambda_R[d-1])  # R_d x R_d
                    delta_mat = np.diag(self.delta[d-1])  # M_d x M_d
                    # temp = np.linalg.inv(self.Sigma[d-1]) - np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)
                    # temp += np.kron(np.kron(np.eye(self.R[d]), delta_mat), lambda_mat_prev)
                    # temp = np.linalg.inv(temp)
                    # V_d = np.diag(temp).reshape(self.R[d-1], self.M[d-1], self.R[d], order='F')

                    V_d = self.covar[d-1].reshape(self.R[d-1], self.M[d-1], self.R[d], order='F', copy=False)
                    V_d_3 = np.vstack([V_d[:,:,i].T.reshape(1,-1).flatten() for i in range(V_d.shape[2])])
                    expectation1_term2 = V_d_3@np.kron(self.delta[d-1], self.lambda_R[d-1])
                    expectation1 = expectation1_term1 + expectation1_term2

                    expectation2_term1 = np.diag(W_1@np.kron(Lambda_d_plus_1, Delta_d)@W_1.T)

                    # lambda_mat_next = np.diag(self.lambda_R[d+1])  # R_{d+1} x R_{d+1}
                    # lambda_mat_prev = np.diag(self.lambda_R[d])  # R_d x R_d
                    # delta_mat = np.diag(self.delta[d])  # M_d x M_d
                    # temp = np.linalg.inv(self.Sigma[d]) - np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)
                    # temp += np.kron(np.kron(lambda_mat_next, delta_mat), np.eye(self.R[d]))
                    # temp = np.linalg.inv(temp)
                    # V_d = np.diag(temp).reshape(self.R[d], self.M[d], self.R[d+1], order='F')

                    V_d = self.covar[d].reshape(self.R[d], self.M[d], self.R[d+1], order='F', copy=False)
                    V_d_1 = np.vstack([V_d[i,:,:].T.reshape(1,-1).flatten() for i in range(V_d.shape[0])])

                    expectation2_term2 = V_d_1@np.kron(self.lambda_R[d+1], self.delta[d])
                    expectation2 = expectation2_term1 + expectation2_term2

                    self.d_N[d] = np.add(d_0[d], 0.5 * (expectation1 + expectation2))

                    self.lambda_R[d] = self.c_N[d] / self.d_N[d]

                    if rank_pruning:
                        Wall = unfold(self.W[d], 1)
                        comPower = np.diag(Wall @ Wall.T)
                        var_explained = comPower / np.sum(comPower) * 100
                        rankest = np.sum(var_explained > rank_tol)
                        print(rankest)
                        if rankest == 0:
                            collapsed = True
                            break

                        if self.R[d] != rankest:
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
                            covar_d_tensor = self.covar[d].reshape([self.R[d], self.M[d], self.R[d+1]], order='F', copy=False)
                            covar_d_tensor = np.delete(covar_d_tensor, ~pruning_mask, axis=0)
                            covar_d_min_tensor = self.covar[d-1].reshape([self.R[d-1], self.M[d], self.R[d]], order='F', copy=False)
                            covar_d_min_tensor = np.delete(covar_d_min_tensor, ~pruning_mask, axis=2)
                            self.R[d] = sum(pruning_mask)
                            self.lambda_R[d] = self.lambda_R[d][pruning_mask]
                            self.Sigma[d] = Sigma_d_tensor.reshape([self.R[d]*self.M[d]*self.R[d+1], self.R[d]*self.M[d]*self.R[d+1]], order='F')
                            self.Sigma[d-1] = Sigma_d_min_tensor.reshape([self.R[d-1]*self.M[d-1]*self.R[d], self.R[d-1]*self.M[d-1]*self.R[d]], order='F')
                            self.covar[d] = covar_d_tensor.reshape([-1,1])
                            self.covar[d-1] = covar_d_min_tensor.reshape([-1,1])
                            c_0[d] = c_0[d][~pruning_mask]
                            d_0[d] = d_0[d][~pruning_mask]
                            self.c_N[d] = self.c_N[d][~pruning_mask]
                            self.d_N[d] = self.d_N[d][~pruning_mask]

                lam_norm.append([np.linalg.norm(self.lambda_R[d]) for d in range(self.D)])

            if collapsed:
                print("model collapsed")
                break
            # noise precision update
            if tau_update:
                self.a_N = a_0 + self.N / 2
                Expectation_G = self.forward_accumulator_G(self.D)
                Expectation_H = np.ones(self.N).T @ self.forward_accumulator_H(self.D)
                frob_errors = np.linalg.norm(Y)**2 - 2*(Y@Expectation_G).squeeze() + Expectation_H.squeeze()
                self.b_N = b_0 + 0.5 * frob_errors
                self.expectation_tau = self.a_N / self.b_N
            predictions = self.predict(X)
            error_term = 0.5*self.expectation_tau* np.linalg.norm(Y.reshape((self.N,1), copy=False) - predictions)**2

            L2_norms = 0
            ln_q_W = 0
            lambda_term = 0
            delta_term = 0

            for d in range(self.D):
                lambda_mat_next = np.diag(self.lambda_R[d + 1])  # R_{d+1} x R_{d+1}
                lambda_mat_prev = np.diag(self.lambda_R[d])  # R_d x R_d
                delta_mat = np.diag(self.delta[d])  # M_d x M_d
                variance_term = np.kron(np.kron(lambda_mat_next, delta_mat), lambda_mat_prev)

                L2_norms += np.trace(variance_term @ (np.outer(self.W[d].reshape((-1,1)), self.W[d].reshape((-1,1))) + np.diag(self.covar[d])))
                ln_q_W += 0.5*np.log(np.linalg.norm(self.Sigma[d])) + (self.R[d]*self.M[d]/2)*(1+np.log(2*np.pi))
                for r in range(self.R[d]):
                    lambda_term += (1-np.log(self.d_N[d][r])- (d_0[d][r]/self.d_N[d][r]))*self.c_N[d][r] #+ np.log(gamma(self.c_N[d][r]))
                for m in range(self.M[d]):
                    delta_term += np.log(gamma(self.g_N[d][m])) + (1-np.log(self.h_N[d][m]) - (h_0[d][m]/self.h_N[d][m]))*self.g_N[d][m]

            tau_term = np.log(gamma(self.a_N)) + (1 - np.log(self.b_N) - (b_0/self.b_N))*self.a_N
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
            if LB_rel_chan < error_bound:
                print("Convergence bound reached, exiting")
                break
            if W_norm[-1] < 1e-100 or np.isnan(W_norm[-1]):
                print("model collapsed")
                break
        if it == iteration_limit:
            print("Iteration limit reached, exiting")

        if plotting:
            fig, ax1 = plt.subplots()
            ax1.plot(ELBO, label='ELBO')
            ax2 = ax1.twinx()
            if lambda_update:
                ax2.plot(lam_norm, color="red", label=r"$||\lambda||$")
            if delta_update:
                ax2.plot(del_norm, color="purple", label=r"$||\delta||$")
            ax2.plot(W_norm, color="orange", label=r"$||\boldsymbol{\mathcal{W}}||$")
            ax1.set_title("Training ELBO")
            ax1.set_xlabel("iteration")
            ax1.set_ylabel("ELBO")
            ax2.set_ylabel(r"$||\cdot||$")
            fig.legend()
            plt.show()

    def predict(self, X):
        self.feature_map = self.kernel(X, max(self.M))
        self.N = self.feature_map.shape[1]
        return self.forward_accumulator_G(self.D)

    def forward_accumulator_G(self, d):
        G_k = np.ones((self.N, 1)) # N x 1
        for k in range(d):
            G_k = khatri_rao(self.feature_map[k], G_k) @ unfold(self.W[k], 3).T # (N x R_d M_d)(R_d M_d x R_{d+1})
        return G_k # N x R_{d+1}

    def backward_accumulator_G(self, d):
        G_k = np.ones((self.N, 1)) # N x 1
        for k in range(self.D-1, d, -1):
            G_k = khatri_rao(self.feature_map[k], G_k) @ unfold(self.W[k], 1).T # (N x R_{d+1} M_d)(R_{d+1} M_d x R_d)
        return G_k # N x R_d

    def forward_accumulator_H(self, d):
        H_k = np.ones((self.N, 1)) # N x R_d R_d
        for k in range(d):
            Wk = unfold(self.W[k], 3).T # R_d M_d x R_{d+1}
            mean_WW = np.kron(Wk, Wk) # R_d M_d R_d M_d x R_{d+1} R_{d+1}
            mean_WW = mean_WW.reshape([self.R[k],self.M[k],self.R[k],self.M[k],self.R[k+1]**2], order='F')# R_d x M_d x R_d x M_d x R_{d+1} R_{d+1}
            mean_WW = np.transpose(mean_WW,(0,2,1,3,4)) # R_d x R_d x M_d x M_d x R_{d+1}**2
            mean_WW = mean_WW.reshape([(self.R[k]*self.M[k])**2, self.R[k+1]**2], order='F')
            # R_d R_d M_d M_d x R_{d+1}**2

            covariance_shape = (self.R[k], self.M[k], self.R[k+1], self.R[k], self.M[k], self.R[k+1])
            covariance_WW = np.diag(self.covar[k]).reshape(covariance_shape, order='F') # R_d x M_d x R_{d+1} x R_d x M_d x R_{d+1}
            covariance_WW = np.transpose(covariance_WW, [0,3, 1,4, 2,5]) # R_d x R_d x M_d x M_d x R_{d+1} x R_{d+1}
            covariance_WW = covariance_WW.reshape([(self.R[k]*self.M[k])**2, self.R[k+1]**2], order='F')
            # R_d R_d M_d M_d x R_{d+1} R_{d+1}
            expectation_WW = np.add(mean_WW, covariance_WW) # R_d R_d M_d M_d x R_{d+1} R_{d+1}

            H_k = khatri_rao(khatri_rao(self.feature_map[k], self.feature_map[k]), H_k) @ expectation_WW
        return H_k # N x R_{d+1}**2

    def backward_accumulator_H(self, d):
        H_k = np.ones((self.N, 1))
        for k in range(self.D-1, d, -1):
            Wk = unfold(self.W[k], 1).T # R_{d+1} M_d x R_d
            mean_WW = np.kron(Wk, Wk) # R_{d+1} M_d R_{d+1} M_d x R_d R_d
            mean_WW = mean_WW.reshape([self.R[k+1], self.M[k], self.R[k+1], self.M[k], self.R[k]**2], order='F')
            # R_{d+1} x M_d x R_{d+1} x M_d x R_d R_d

            mean_WW = np.transpose(mean_WW, [0,2,1,3,4]) # R_{d+1} x R_{d+1} x M_d x M_d x R_d**2
            mean_WW = mean_WW.reshape([(self.R[k+1]*self.M[k])**2, self.R[k]**2], order='F')
            # R_{d+1} R_{d+1} M_d M_d x R_d**2

            covariance_shape = (self.R[k], self.M[k], self.R[k+1], self.R[k], self.M[k], self.R[k+1])
            covariance_WW = np.diag(self.covar[k]).reshape(covariance_shape, order='F') # R_d x M_d x R_{d+1} x R_d x M_d x R_{d+1}
            covariance_WW = np.transpose(covariance_WW, [2,5,1,4,0,3]) # R_{d+1} x R_{d+1} x M_d x M_d x R_d x R_d
            covariance_WW = covariance_WW.reshape([(self.M[k]*self.R[k+1])**2, self.R[k]**2], order='F') # R_{d+1} R_{d+1} M_d M_d x R_d R_d

            expectation_WW = np.add(mean_WW, covariance_WW)

            H_k = khatri_rao(khatri_rao(self.feature_map[k], self.feature_map[k]), H_k) @ expectation_WW
        return H_k # N x R_d**2
