import numpy as np
import torch

from torch import Tensor
from numpy import ndarray

def contract(A: Tensor, B: Tensor, A_mode: int, B_mode: int) -> Tensor:
    """
    Wrapper function: Tensor contraction across a mode of A and B

    Parameters:
    :param A: torch.tensor
            Input array A
    :param B: torch.tensor
            Input array B
    :param A_mode: int
            mode of A to contract over, size of this mode must be equal to mode of B
    :param B_mode: int
            mode of B to contract over, size of this mode must be equal to mode of A

    :return:
        torch.tensor
        A x B
    """
    return torch.tensordot(A,B,([A_mode],[B_mode]))

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

    if not A.shape[1] == B.shape[1]:
        raise ValueError('A and B must have same number of columns')
    c = np.vstack([np.kron(A[:, k], B[:, k]) for k in range(B.shape[1])])
    return c

def row_khatri_rao(A: ndarray, B: ndarray) -> ndarray:
    A = np.asarray(A)
    B = np.asarray(B)

    return khatri_rao(A.T, B.T).T

def Kronecker(A: Tensor | ndarray, B: Tensor | ndarray) -> Tensor | ndarray:
    return torch.kron(A, B)

def Kernel(X, M):
    return np.array([X for _ in range(M)])

def pure_power_features_full(X, input_dimension):
    """
    Pure-Power Polynomial Features

    Parameters:
    X : numpy.ndarray
        Input array of shape (n_samples, n_features).
    M : int
        Maximum power (degree) of the polynomial features.

    Returns:
    Mati : numpy.ndarray
        Array of shape (n_features, n_samples, M) containing unit-norm pure-power features.
    """
    # Compute the pure-power features
    Mati = np.power(X.T[:, :, np.newaxis], np.arange(input_dimension))

    # Normalize each sample's features along the last axis to have a unit norm
    norms = np.linalg.norm(
        Mati, axis=2, keepdims=True
    )  # Compute norms along the power axis
    Mati = Mati / norms  # Normalize features to unit norm
    return np.permute_dims(Mati, (2, 0, 1))


    return Mati

def generate_toy_dataset(dimensionality, number_data_points, noise_variance, linear=True):
    X = np.random.normal(0, 1, (number_data_points, dimensionality))
    parameters = np.random.normal(0,1, dimensionality)
    noise = np.random.normal(0, noise_variance, dimensionality)
    Y = np.zeros(number_data_points)
    if linear:
        for i, x in enumerate(X):
            Y[i] = np.dot(x, parameters)
            X[i] = x + noise

    return X, Y