import numpy as np

from numpy import ndarray

class Core:
    def __init__(self, data):
        self.core = data

    def __repr__(self):
        return str(self.core)

    def unfold(self, mode):
        if mode == 1:
            return np.vstack([self.core[i,:,:].reshape(1,-1).flatten() for i in range(self.core.shape[0])])
        elif mode == 2:
            return np.vstack([self.core[:,i,:].T.reshape(1,-1).flatten() for i in range(self.core.shape[1])])
        elif mode == 3:
            return np.vstack([self.core[:,:,i].T.reshape(1,-1).flatten() for i in range(self.core.shape[2])])
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

def block2block(A: ndarray[float], I: int, J: int, K:int) -> ndarray[float]:
    """
    takes an array of size IJIJx_ and transforms it into IIJJx_
    :param A: input array of size IJIJx_
    :param I: first and third index to become first and second
    :param J: second and fourth index to become third and fourth
    :return: permuted array of size IIJJx_ with same data as A
    """
    tensor = A.reshape(I,J,I,J,K**2)
    tensor = np.transpose(tensor,(0,2,1,3,4))
    return tensor.reshape(I*I*J*J,K**2)
