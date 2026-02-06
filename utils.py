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
            return np.vstack([self.core[:,i,:].reshape(1,-1).flatten() for i in range(self.core.shape[1])])
        elif mode == 3:
            return np.vstack([self.core[:,:,i].reshape(1,-1).flatten() for i in range(self.core.shape[2])])
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

        self.TT_ranks = [core.core.shape[0] for core in cores]
        self.TT_ranks.append(1)
        self.dims = [core.core.shape[1] for core in cores]

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

def block2outer(A: ndarray, block_shape: tuple[int, int]) -> ndarray:
    rows = int(np.sqrt(A.size))
    A_shape = A.shape
    result = np.zeros((rows, rows))
    for i in range(int(A_shape[1]/block_shape[1])):
        for j in range(int(A_shape[0]/block_shape[0])):
            row_start = j*block_shape[0]
            row_end = (j+1)*block_shape[0]
            col_start = i*block_shape[1]
            col_end = (i+1)*block_shape[1]
            result[i,:] = A[row_start:row_end, col_start:col_end].reshape(1,-1).flatten()
    return result

def outer2block(A: ndarray, block_shape: tuple[int, int], output_shape: tuple[int, int]) -> ndarray:
    result = np.zeros(output_shape)
    col = 0
    row = 0
    for i in range(A.shape[0]):
        block = A[i,:].reshape(block_shape)
        row_start = row*block_shape[0]
        row_end = (row+1)*block_shape[0]
        col_start = col*block_shape[1]
        col_end = (col+1)*block_shape[1]
        result[row_start:row_end, col_start:col_end] = block
        row += 1
        if row*block_shape[0] == output_shape[0]:
            row = 0
            col += 1
    return result