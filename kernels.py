import numpy as np

def no_kernel(X, M):
    return np.array([X for _ in range(M)])

def quadratic_kernel(X, D):
    Phi = np.zeros((D, X.shape[0], X.shape[1]))
    for d in range(D):
        Phi[d,:,:] = np.array([x**(d+1) for x in X])
    return Phi

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
    Mati = np.power(X[:, :, np.newaxis], np.arange(input_dimension))

    # Normalize each sample's features along the last axis to have a unit norm
    norms = np.linalg.norm(
        Mati, axis=2, keepdims=True
    )  # Compute norms along the power axis
    Mati = Mati / norms  # Normalize features to unit norm
    return np.permute_dims(Mati, (2, 0, 1))