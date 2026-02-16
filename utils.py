import numpy as np

from numpy import ndarray

class Core:
    def __init__(self, data):
        self.core = data

    def __repr__(self):
        return str(self.core)

    def unfold(self, mode):
        if mode == 1:
            return np.vstack([self.core[i,:,:].reshape(1,-1).flatten() for i in range(self.core.shape[0])]) # R_d x R_{d+1} M_d
        elif mode == 2:
            return np.vstack([self.core[:,i,:].T.reshape(1,-1).flatten() for i in range(self.core.shape[1])]) # M_d x R_d R_{d+1}
        elif mode == 3:
            return np.vstack([self.core[:,:,i].T.reshape(1,-1).flatten() for i in range(self.core.shape[2])]) # R_{d+1} x R_d M_d
        else:
            raise ValueError("Unsupported mode {}, please select mode from [1,2,3]".format(mode))


class TensorTrain:
    def __init__(self, cores: list[Core]):
        self.cores = cores
        for d in range(len(self.cores) - 1):
            if not isinstance(self.cores[d], Core):
                raise ValueError(f"Core {d} is not a Core")
            if self.cores[d].core.shape[2] != self.cores[d + 1].core.shape[0]:
                raise ValueError(f"Dimensions of cores do not match, between cores {d} and {d + 1}")

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
    # B kron A to maintain dimension ordering
    c = np.vstack([np.kron(B[k, :], A[k, :]) for k in range(A.shape[0])])
    return c
