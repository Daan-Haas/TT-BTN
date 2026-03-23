import numpy as np

from numpy import ndarray

# class Core:
#     def __init__(self, data):
#         self.core = data
#
#     def __repr__(self):
#         return str(self.core)

def unfold(core, mode):
    if mode == 1:
        return np.vstack([core[i,:,:].reshape(1,-1).flatten() for i in range(core.shape[0])]) # R_d x R_{d+1} M_d
    elif mode == 2:
        return np.vstack([core[:,i,:].reshape(1,-1).flatten() for i in range(core.shape[1])]) # M_d x R_d R_{d+1}
    elif mode == 3:
        return np.vstack([core[:,:,i].T.reshape(1,-1).flatten() for i in range(core.shape[2])]) # R_{d+1} x R_d M_d
    else:
        raise ValueError("Unsupported mode {}, please select mode from [1,2,3]".format(mode))


# class TensorTrain:
#     def __init__(self, cores: list[Core]):
#         self.cores = cores
#         for d in range(len(self.cores) - 1):
#             if not isinstance(self.cores[d], Core):
#                 raise ValueError(f"Core {d} is not a Core")
#             if self.cores[d].core.shape[2] != self.cores[d + 1].core.shape[0]:
#                 raise ValueError(f"Dimensions of cores do not match, between cores {d} and {d + 1}")

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
        raise ValueError(f'A and B must have same number of rows, {A.shape[0]} does not equal {B.shape[0]}')
    c = np.vstack([np.kron(A[k, :], B[k, :]) for k in range(A.shape[0])])
    return c

def columnwise_kronecker(A, B):
    """Compute the columnwise Kronecker product of two matrices A and B."""
    # Check dimensions
    if A.shape[1] != B.shape[1]:
        raise ValueError("Number of columns in A and B must be the same.")

    # Dimensions of input matrices
    m, n = A.shape
    p, _ = B.shape

    # Initialize the result matrix
    K = np.zeros((m * p, n))

    # Calculate columnwise Kronecker product
    for i in range(n):
        K[:, i] = np.kron(A[:, i], B[:, i])

    return K


def dotkron(*matrices):
    """
    Computes the row-wise right-Kronecker product of two or three matrices.

    Parameters:
        matrices: Two or three matrices with the same number of rows.

    Returns:
        y: The resulting row-wise right-Kronecker product.

    Raises:
        ValueError: If the matrices do not have the same number of rows
                    or if the number of matrices is not 2 or 3.
    """
    if len(matrices) == 2:
        L, R = matrices
        r1, c1 = L.shape
        r2, c2 = R.shape

        if r1 != r2:
            raise ValueError("Matrices should have equal rows!")

        # Row-wise right-Kronecker product for two matrices
        y = np.tile(L, (1, c2)) * np.kron(R, np.ones((1, c1)))

    elif len(matrices) == 3:
        L, M, R = matrices
        r1, _ = L.shape
        r2, _ = M.shape
        r3, _ = R.shape

        if r1 != r2 or r2 != r3:
            raise ValueError("Matrices should have equal rows!")

        # Recursive call for three matrices
        y = dotkron(L, dotkron(M, R))

    else:
        raise ValueError("Please input 2 or 3 matrices!")

    return y


def temp(
    Phi, V, R
):  # her bir RXR block'u vectorize edip rowlarina koyuyor yeni matrix'in
    I = Phi.shape[1]
    V = np.reshape(V, (I, R, I, R), order="F")
    V_permuted = np.transpose(V, axes=(0, 2, 1, 3))
    result = np.reshape(V_permuted, (I**2, R**2))
    return dotkron(Phi, Phi) @ result


def dotkronX(A, B, y):
    """
    Computes the Kronecker dot product for large matrices using batch processing.

    Parameters:
    A : np.ndarray (N, DA)
        First input matrix
    B : np.ndarray (N, DB)
        Second input matrix
    y : np.ndarray (N, 1)
        Target vector
    batch_size : int, optional
        Number of samples to process in each batch (default is 10000)

    Returns:
    CC : np.ndarray (DA*DB, DA*DB)
        Computed Kronecker product matrix
    Cy : np.ndarray (DA*DB, 1)
        Computed target vector
    """
    N, DA = A.shape
    _, DB = B.shape

    CC = np.zeros((DA * DB, DA * DB))
    Cy = np.zeros((DA * DB, 1))
    y = y.reshape(-1, 1)  # Ensure y has shape (N, 1)

    batch_size = 10000
    for n in range(0, N, batch_size):
        idx = min(n + batch_size - 1, N)  # Ensure we don't exceed N

        temp = (np.tile(A[n:idx, :], (1, DB))) * (
            np.kron(B[n:idx, :], np.ones((1, DA)))
        )

        CC += temp.T @ temp  # Accumulate Kronecker product
        Cy += temp.T @ y[n:idx, :]

    return CC, Cy


def safe_division(X, A, epsilon=1):
    A_safe = np.where(A == 0, epsilon, A)
    Y = X / A_safe
    return Y


def safelog(x):
    return np.log(np.clip(x, 1e-10, 1e10))  # x to range [1e-10, 1e10]
