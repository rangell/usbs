import numba as nb
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from typing import Callable

from IPython import embed


def create_C_innerprod(C: csc_matrix) -> Callable[[np.ndarray], np.double]:
    indptr = C.indptr
    indices = C.indices
    data = C.data
    @nb.njit(parallel=True, fastmath=True)
    def C_innerprod(X: np.ndarray) -> np.double:
        sum = 0.0
        for j in nb.prange(indptr.size - 1):
            for offset in nb.prange(indptr[j], indptr[j+1]):
                sum += data[offset] * X[indices[offset], j]
        return sum
    return C_innerprod


def create_C_add(C: csc_matrix) -> Callable[[np.ndarray], np.ndarray]:
    indptr = C.indptr
    indices = C.indices
    data = C.data
    @nb.njit(parallel=True, fastmath=True)
    def C_add(X: np.ndarray) -> np.ndarray:
        Y = np.empty_like(X)
        Y = X
        for j in nb.prange(indptr.size - 1):
            for offset in nb.prange(indptr[j], indptr[j+1]):
                i = indices[offset]
                Y[i, j] = data[offset] + X[i, j]
        return Y
    return C_add


def create_C_matvec(C: csc_matrix) -> Callable[[np.ndarray], np.ndarray]:
    indptr = C.indptr
    indices = C.indices
    data = C.data
    @nb.njit(parallel=True, fastmath=True)
    def C_matvec(u: np.ndarray) -> np.ndarray:
        v = np.zeros_like(u)
        for j in nb.prange(indptr.size - 1):
            for offset in nb.prange(indptr[j], indptr[j+1]):
                v[indices[offset]] += u[j] * data[offset]
        return v
    return C_matvec


def create_A_operator(n: int) -> Callable[[np.ndarray], np.ndarray]:
    @nb.njit(parallel=True, fastmath=True)
    def A_operator(X: np.ndarray) -> np.ndarray:
        z = np.empty((n,))
        for i in nb.prange(n):
            z[i] = X[i, i]
        return z
    return A_operator


def create_A_operator_slim(n: int) -> Callable[[np.ndarray], np.ndarray]:
    @nb.njit(parallel=True, fastmath=True)
    def A_operator_slim(u: np.ndarray) -> np.ndarray:
        z = np.empty((n,))
        for i in nb.prange(n):
            z[i] = u[i]**2
        return z
    return A_operator_slim


def create_A_adjoint(n: int) -> Callable[[np.ndarray], np.ndarray]:
    @nb.njit(parallel=True, fastmath=True)
    def A_adjoint(z: np.ndarray) -> np.ndarray:
        Y = np.zeros((n,n))
        for i in nb.prange(n):
            Y[i, i] = z[i]
        return Y
    return A_adjoint


def create_A_adjoint_slim(n: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    @nb.njit(parallel=True, fastmath=True)
    def A_adjoint(z: np.ndarray, u: np.ndarray) -> np.ndarray:
        v = np.empty((n,))
        for i in nb.prange(n):
            v[i] = z[i] * u[i]
        return v
    return A_adjoint


def create_proj_K(n: int) -> Callable[[np.ndarray], np.ndarray]:
    @nb.njit(parallel=True, fastmath=True)
    def proj_K(z: np.ndarray) -> np.ndarray:
        return np.ones((n,))
    return proj_K


if __name__ == "__main__":
    np.random.seed(0)
    MAT_PATH = "./data/maxcut/Gset/G1.mat"
    problem = loadmat(MAT_PATH)
    C = problem["Problem"][0][0][1]

    # create a random test matrix
    X = np.random.randn(*C.shape)
    u = np.random.randn(C.shape[0])

    C_innerprod = create_C_innerprod(C)
    C_add = create_C_add(C)
    C_matvec = create_C_matvec(C)
    A_operator = create_A_operator(C.shape[0])
    A_operator_slim = create_A_operator_slim(C.shape[0])
    A_adjoint = create_A_adjoint(C.shape[0])
    A_adjoint_slim = create_A_adjoint_slim(C.shape[0])
    proj_K = create_proj_K(C.shape[0])

    embed()
    exit()