import numpy as np
from numpy.linalg import qr, svd
from collections import defaultdict
from time import time


def bksvd(A, k, num_iter, block_size):
    """
    Block Krylov Truncated SVD of any matrix A.
    This code is taken from MATLAB code at https://github.com/cpmusco/bksvd.

    Returns
    -------
    U, S, V -- SVD-factors
    """
    # we want to iterate one the smaller axis of A
    tpose = False
    if A.shape[0] < A.shape[1]:
        A = A.T
        tpose = True

    # krylov subspace columns
    K = np.zeros((A.shape[1], block_size * num_iter))

    # random block initialization
    block = np.random.randn(A.shape[1], block_size)
    block, _ = qr(block)

    for i in range(num_iter):
        block = np.linalg.multi_dot([A.T, A, block])
        block, _ = qr(block)
        K[:, i*block_size:(i+1)*block_size] = block.copy()
    Q, _ = qr(K)

    # svd of projected matrix
    U, S, V = None, None, None
    Ut, St, Vt = svd(A.dot(Q), full_matrices=False)
    Vt = Vt.T
    S = St[:k]
    if tpose:
        V = Ut[:, :k]
        U = Q.dot(Vt[:, :k])
    else:
        U = Ut[:, :k]
        V = Q.dot(Vt[:, :k])

    return U, S, V


def _make_svd(A, k, block_size, K, i):
    Q, _ = qr(K[:, :(i+1)*block_size])
    U, S, V = None, None, None
    Ut, St, Vt = svd(A.dot(Q), full_matrices=False)
    S = St[:k]
    U = Ut[:, :k]
    V = Q.dot(Vt[:k, :].T)

    return U, S, V


def bksvd_h_conv(A, k, num_iter, block_size, dist, denominator):
    """
    Block Krylov Truncated SVD of symmetric matrix A

    Parameters
    ---------
    dist -- precalculated distance matrix
    denominator -- precalculated best residual norm

    Returns
    -------
    U, S, V.T -- SVD-factors
    conv -- dict with 'error' and 'time' fields
    """
    # krylov subspace columns
    K = np.zeros((A.shape[1], block_size * num_iter))

    # random block initialization
    block = np.random.randn(A.shape[1], block_size)
    block, _ = qr(block)

    # convergence history
    conv = defaultdict(list)

    for i in range(num_iter):
        # algorithm steps
        start = time()

        block = A.dot(A.dot(block))
        block, _ = qr(block)
        K[:, i*block_size:(i+1)*block_size] = block.copy()

        timer = time() - start

        # collect indermidiate results
        u, s, v = _make_svd(A, k, block_size, K, i)
        error = np.linalg.norm(u @ np.diag(s) @ v.T - dist) / denominator - 1

        conv['error'].append(error)
        conv['time'].append(timer)

    return *_make_svd(A, k, block_size, K, num_iter), conv


def bksvd_h(A, k, num_iter, block_size):
    """
    Block Krylov Truncated SVD of symmetric matrix A

    Returns
    -------
    U, S, V -- SVD-factors
    """
    # krylov subspace columns
    K = np.zeros((A.shape[1], block_size * num_iter))

    # random block initialization
    block = np.random.randn(A.shape[1], block_size)
    block, _ = qr(block)

    for i in range(num_iter):
        block = A.dot(A.dot(block))
        block, _ = qr(block)
        K[:, i*block_size:(i+1)*block_size] = block.copy()

    Q, _ = qr(K)

    # svd of projected matrix
    U, S, V = None, None, None
    Ut, St, Vt = svd(A.dot(Q), full_matrices=False)
    Vt = Vt.T

    S = St[:k]
    U = Ut[:, :k]
    V = Q.dot(Vt[:, :k])

    return U, S, V
