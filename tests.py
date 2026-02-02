import torch
import math

from model import *
from toy_data import *
from utils import *

def test_unfoldings():
    A = Core(torch.tensor([[[1, 5], [2, 6]], [[3, 7], [4, 8]]]))
    A_unfold_1 = A.unfold(1)
    A_unfold_2 = A.unfold(2)
    A_unfold_3 = A.unfold(3)

    check_unfold_1 = []
    check_unfold_2 = []
    check_unfold_3 = []

    for i in range(A.core.size()[0]):
        for j in range(A.core.size()[1]):
            for k in range(A.core.size()[2]):
                check_unfold_1.append(A.core[i, j, k] == A_unfold_1[i, ((j + 1) + k * 2) - 1])
                check_unfold_2.append(A.core[i, j, k] == A_unfold_2[j, ((i + 1) + k * 2) - 1])
                check_unfold_3.append(A.core[i, j, k] == A_unfold_3[k, ((i + 1) + j * 2) - 1])
    print(all(check_unfold_1))
    print(all(check_unfold_2))
    print(all(check_unfold_3))

def test_G_accumulators():
    D = 2  # Number of cores
    ranks = [2 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [1 for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(1, 10, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)
    assert(np.allclose(model.forward_accumulator_G(model.D), model.backward_accumulator_G(-1), rtol=1e-6))

def test_H_accumulators():
    D = 2  # Number of cores
    ranks = [2 for _ in range(D - 1)]  # Tensor-train ranks
    ranks = [1] + ranks + [1]  # first and last rank must be 1 to maintain output dimension
    dims = [1 for _ in range(D)]  # dimensionality of kernel

    X_train, Y_train, X_test, Y_test, parameters = generate_lin_dataset(1, 10, 0)
    model = BTTKM(D, ranks, dims, no_kernel)
    model.train(X_train, Y_train, iteration_limit=0)
    assert(np.allclose(model.forward_accumulator_H(model.D), model.backward_accumulator_H(-1), rtol=1e-6))


def test_kronecker():
    pass

test_unfoldings()
test_G_accumulators()
test_H_accumulators()