import math

from model import *
from toy_data import *
from utils import *

def test_unfoldings():
    A = Core(np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]]))
    A_unfold_1 = A.unfold(1)
    A_unfold_2 = A.unfold(2)
    A_unfold_3 = A.unfold(3)
    max_i = A.core.shape[0]
    max_j = A.core.shape[1]
    max_k = A.core.shape[2]

    check_unfold_1 = []
    check_unfold_2 = []
    check_unfold_3 = []

    for i in range(A.core.shape[0]):
        for j in range(A.core.shape[1]):
            for k in range(A.core.shape[2]):
                check_unfold_1.append(A.core[i, j, k] == A_unfold_1[i, j + (k * max_j)])
                check_unfold_2.append(A.core[i, j, k] == A_unfold_2[j, i + (k * max_i)])
                check_unfold_3.append(A.core[i, j, k] == A_unfold_3[k, i + (j * max_i)])
    assert all(check_unfold_1), 'mode-1 unfolding incorrect'
    assert all(check_unfold_2), 'mode-2 unfolding incorrect'
    assert all(check_unfold_3), 'mode-3 unfolding incorrect'

def test_G_accumulators():
    D = 4  # Number of cores
    I = 3
    ranks = [2 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [I for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(I, 10, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)
    assert np.allclose(model.forward_accumulator_G(model.D), model.backward_accumulator_G(-1), rtol=1e-6), 'forward and backward G accumulation errors'

def test_H_accumulators():
    D = 4  # Number of cores
    I = 2
    ranks = [3 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [I for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(I, 10, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)
    assert np.allclose(model.forward_accumulator_H(model.D), model.backward_accumulator_H(-1), rtol=1e-6), 'forward and backward H accumulation errors'

def test_H_forward_against_G():
    D = 3  # Number of cores
    I = 2
    ranks = [3 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [I for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(I, 10, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)

    d = model.D - 1
    G_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_G(d))
    G = khatri_rao(model.backward_accumulator_G(d), G_lt)
    GTG = G.T @ G

    H_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_H(d)) # N x R_{d+1}**2 M_d
    H_gt = khatri_rao(model.backward_accumulator_H(d), model.feature_map[d]) #  N x M_d R_d**2
    H_d = H_lt.T @ H_gt # R_{d+1} R_{d+1} M_d x M_d R_d R_d
    H_d = H_d.reshape([model.R[d], model.R[d], model.M[d], model.M[d], model.R[d+1], model.R[d+1]], order='F')

    H_d = H_d.transpose([4, 2, 0, 5, 3, 1])
    H_d = H_d.reshape([model.R[d]*model.M[d]*model.R[d+1], model.R[d]*model.M[d]*model.R[d+1]], order='C')

    assert np.allclose(GTG, H_d), "GTG does not match forward accumulator H"

def test_H_backward_against_G():
    D = 3  # Number of cores
    I = 2
    ranks = [2 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [I for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(I, 10, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)

    d = 0
    G_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_G(d))
    G = khatri_rao(model.backward_accumulator_G(d), G_lt)
    GTG = G.T @ G

    H_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_H(d)) # N x R_{d+1}**2 M_d
    H_gt = khatri_rao(model.backward_accumulator_H(d), model.feature_map[d]) # N x M_d R_d**2
    H_d = H_lt.T @ H_gt # R_{d+1}**2 M_d x M_d R_d**2
    H_d = H_d.reshape([model.R[d+1], model.R[d+1], model.M[d], model.M[d], model.R[d], model.R[d]], order='F')
    H_d = H_d.transpose([4, 2, 0, 5, 3, 1])
    H_d = H_d.reshape([model.R[d]*model.M[d]*model.R[d+1], model.R[d]*model.M[d]*model.R[d+1]], order='C')
    assert np.allclose(GTG, H_d), "GTG does not match backward accumulator H"

def test_H_against_G_all_cores():
    D = 3  # Number of cores
    I = 3
    ranks = [2 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [I for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(I, 10, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)

    for d in range(D):
        G_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_G(d)) # N x R_{d+1} M_d
        G = khatri_rao(model.backward_accumulator_G(d), G_lt) # N x R_{d+1} M_d R_d
        GTG = G.T @ G
        H_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_H(d)) # N x R_{d+1}**2 M_d
        H_gt = khatri_rao(model.backward_accumulator_H(d), model.feature_map[d]) # N x M_d R_d**2
        H_d = H_lt.T @ H_gt # R_{d+1}**2 M_d x M_d R_d**2
        H_d = H_d.reshape([model.R[d], model.R[d], model.M[d], model.M[d], model.R[d+1], model.R[d+1]], order='F')
        H_d = H_d.transpose([4, 2, 0, 5, 3, 1])
        H_d = H_d.reshape([model.R[d]*model.M[d]*model.R[d+1], model.R[d]*model.M[d]*model.R[d+1]], order='C')

        assert np.allclose(GTG, H_d), f"GTG does not match H for d={d}"
    print("all cores match")

def print_unfold(mode):
    A = Core(np.array([[[1,5],[3,7]],[[2,6],[4,8]]]))
    print(f'Tensor:{A}')
    print(f'mode-{mode} unfolding of A: {A.unfold(mode)}')

def print_khatri_rao():
    A = np.array([[1,3],[2,4]])
    B = np.array([[5,7],[6,8]])
    print(f'A = {A},\n B = {B}\nA katri rao B = {khatri_rao(A,B)}')
    print('structure: [B_11 * A_1,:     B_12 * A_1,:]\n           [B_21 * A_1,:     B_12 * A_2,:]')

def tensorize_vector():
    A = np.arange(24)
    A_no_permute = A.reshape((2,3,4))
    A_permute = np.permute_dims(A.reshape((4,3,2)), [2,1,0])
    print(f'A_211 element without permute: {A_no_permute[1,0,0]}')
    print(f'A_211 element with permute: {A_permute[1,0,0]}')

def forward_step_by_step():
    D = 3  # Number of cores
    I = 2
    ranks = [2 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [I for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(I, 2, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)
    model.W.cores[0] = Core(np.array([[[1,3],[2,4]]]))
    model.W.cores[1] = Core(np.array([[[5,9],[7,11]],[[6, 10], [8,12]]]))
    model.W.cores[2] = Core(np.array([[[13],[15]],[[14],[16]]]))

    model.feature_map[0] = np.array([[1,3],[2,4]])
    model.feature_map[1] = np.array([[10,30],[20,40]])
    model.feature_map[2] = np.array([[100,300],[200,400]])
    model.forward_accumulator_H(3)

def backward_step_by_step():
    D = 3  # Number of cores
    I = 2
    ranks = [3 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [I for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(I, 2, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)
    model.W.cores[0] = Core(0.1*np.array([[[1,3, 3],[2,4,4]]]))
    model.W.cores[1] = Core(0.1*np.array([[[1,7,13],
                                           [4,10,16]],

                                          [[2,8,14],
                                           [5,11,17]],

                                          [[3,9,15],
                                           [6,12,18]]]))
    model.W.cores[2] = Core(0.1*np.array([[[13],[15]],[[14],[16]],[[17],[18]]]))

    model.feature_map[0] = np.array([[1,3],[2,4]])
    model.feature_map[1] = np.array([[1,3],[2,4]])
    model.feature_map[2] = np.array([[1,3],[2,4]])
    for d in range(D):
        G_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_G(d))
        G = khatri_rao(model.backward_accumulator_G(d), G_lt)
        GTG = G.T @ G
        H_lt = khatri_rao(model.feature_map[d], model.forward_accumulator_H(d))  # N x R_{d+1}**2 M_d
        H_gt = khatri_rao(model.backward_accumulator_H(d), model.feature_map[d])  # N x M_d R_d**2
        H_d = H_lt.T @ H_gt  # R_{d+1}**2 M_d x M_d R_d**2
        H_d = H_d.reshape([model.R[d], model.R[d], model.M[d], model.M[d], model.R[d+1], model.R[d+1]], order='F')
        H_d = H_d.transpose([4,2,0, 5,3,1])
        H_d = H_d.reshape([model.R[d] * model.M[d] * model.R[d + 1], model.R[d] * model.M[d] * model.R[d + 1]],
                          order='C')

        assert np.allclose(GTG, H_d), "GTG and H do not match"

test_G_accumulators()
test_H_accumulators()
test_H_backward_against_G()
test_H_forward_against_G()
test_H_against_G_all_cores()