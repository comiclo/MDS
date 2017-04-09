import numpy as np

def centralize(d):
    """
    d is the euclidean distance matrix
    """
    J = centring_matrix(len(d))
    return -0.5 * J * d * J


def centring_matrix(n):
    e = np.ones(shape=(n, n))
    I = np.eye(n)
    return I - e / n


def random_symmetric_maxtirx(n, scale=1):
    a = np.random.randn(n, n) * scale
    return (a + a.T) / 2
