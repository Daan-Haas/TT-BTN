import numpy as np
import torch

from numpy import ndarray
from torch import Tensor

class Core:
    def __init__(self, data):
        self.core = data

    def __repr__(self):
        return str(self.core)

    def unfold(self, mode):
        if mode == 1:
            shape = (self.core.size(dim=0),self.core.size(dim=1)*self.core.size(dim=2))
            return self.core.permute([0,2,1]).reshape(shape)
        elif mode == 2:
            shape = (self.core.size(dim=1), self.core.size(dim=0)*self.core.size(dim=2))
            return self.core.permute([1,2,0]).reshape(shape)
        elif mode == 3:
            shape = self.core.size(dim=2),self.core.size(dim=0)*self.core.size(dim=1)
            return self.core.permute([2,1,0]).reshape(shape)
        else:
            raise ValueError("Unsupported mode {}, please select mode from [1,2,3]".format(mode))

    def size(self, dims):
        return self.core.size(dims)


class TensorTrain:
    def __init__(self, cores: list[Core]):
        self.cores = cores
        for d in range(len(self.cores) - 1):
            if not isinstance(self.cores[d], Core):
                raise ValueError(f"Core {d} is not a Core")
            if self.cores[d].size(dims=2) != self.cores[d + 1].size(dims=0):
                raise ValueError(f"Dimensions of cores do not match, between cores {d} and {d + 1}")

        self.TT_ranks = [core.size(0) for core in cores]
        self.TT_ranks.append(1)
        self.dims = [core.size(1) for core in cores]

    def TT_inner(self, other):
        if not isinstance(other, TensorTrain):
            raise NotImplementedError
        pass

def khatri_rao(A: ndarray, B: ndarray) -> ndarray:
    """
    Khatri-Rao product (Columnwise Kronecker product)

    Parameters:
    :param A: (n, k) array_like
        Input array A
    :param B: (m, k) array_like
        Input array B
    :return: c: (n*m, k) ndarray
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if not A.shape[0] == B.shape[0]:
        raise ValueError('A and B must have same number of rows')
    c = np.vstack([np.kron(A[k, :], B[k, :]) for k in range(A.shape[0])])
    return c

def row_khatri_rao(A: ndarray, B: ndarray) -> ndarray:
    A = np.asarray(A)
    B = np.asarray(B)

    c = np.vstack([np.kron(A[k,:], B[k,:]) for k in range(A.shape[0])])
    return c

def Kronecker(A: Tensor | ndarray, B: Tensor | ndarray) -> Tensor | ndarray:
    if type(A) not in [Tensor, ndarray] or type(B) not in [Tensor, ndarray]:
        raise TypeError('A and B must be Tensors or ndarrays')
    if type(A) == Tensor:
        if type(B) == ndarray:
            B = torch.from_numpy(B)
        return torch.kron(A, B)
    elif type(B) == np.ndarray:
            return np.kron(A, B)
    else:
        A = torch.from_numpy(A)
        return torch.kron(A, B)
