# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lanczos methods for solving large symmetric eigenvalue problems.

The "basic" Lanczos method has severe numerical stability issues when
implemented with non-exact arithmetic. "Full reorthogonalization" fixes these
issues, but requires increases the runtime from O(k) to O(k**2), which quickly
becomes prohibitively expensive.

Restarted Lanczos methods use restarts to avoid the quadratic scaling. In
particular, the "thick restart" variant we implement here is reported to be the
approach of choice for large-scale eigenvalue problems [2].

References:
[1] Wu, K., Simon, H. D. & Wang, L.-W. Thick-Restart Lanczos Method for
        Electronic Structure Calculations. Journal of Computational Physics 154,
        156–173 (1999). https://escholarship.org/content/qt6gk8b245/qt6gk8b245.pdf
[2] V. Hernandez, J. E. Roman, A. Tomas, and V. Vidal. Krylov-schur methods
        in SLEPc. Technical report, Universidad Politecnica de Valencia, 2007.
        http://slepc.upv.es/documentation/reports/str7.pdf

Future improvements:
- Tests for Hermitian (non-symmetric) matrices (with complex eigenvectors)
- JVP and VJP rules (needed for automatic differentitation).
- Support for finding the largest eigenvalue as well.
- Port to jax.scipy.sparse.linalg.eigsh
- Preconditioning for finding non-perpheral eigenvalues and solving
    generalized eigenvalue problems.
- Consider also implementing LOBPCG (a non-Lanczos method).
"""

from collections import namedtuple
from functools import partial
import jax
from jax import lax
import jax.numpy as jnp

from solver.utils import apply_A_adjoint_slim, apply_A_adjoint_batched
from utils.loop import while_loop

from IPython import embed


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)


def _iterative_classical_gram_schmidt(Q, x):
    """Orthogonalize x against the columns of Q."""
    # "twice is enough"
    # http://slepc.upv.es/documentation/reports/str1.pdf

    q = x
    r = 0

    # iteration 1
    h = _dot(Q.T.conj(), q)
    q = q - _dot(Q, h)
    r = r + h

    #iteration 2
    h = _dot(Q.T.conj(), q)
    q = q - _dot(Q, h)
    r = r + h

    return r, q


def _lanczos_restart(n, C, A_data, A_indices, adjoint_left_vec, k, m, Q, alpha, beta, tolerance):
    """The inner loop of the restarted Lanzcos method."""

    def body_fun(i, carray):
        Q, alphas, betas = carray
        q_prev = Q[:, i]
        q = C @ q_prev + apply_A_adjoint_slim(
            n, A_data, A_indices, adjoint_left_vec, q_prev)

        # We make the conservative choice of orthongalizing and reorthgonalizing
        # always, against the full set of Lanczos vectors. This could potentially be
        # sped up by selectively using local orthogonalization (as in the main
        # reference above), but at most we would save a factor of ~2. But as noted
        # in Reference [2]:
        #     In the context of thick-restart Lanczos, the conclusion is that full
        #     orthogonalization is the best option, since it provides the maximum
        #     robustness (which may favor the convergence of the method) at a moderate
        #     additional cost (in this context, orthogonalization with respect to the
        #     first p vectors is required anyway).
        # With JAX, it makes even more sense to do full orthogonalization, since
        # making use of matrix-multiplication for orthogonalization requires a
        # statically sized set of Lanczos vectors.
        Q_valid = (i >= jnp.arange(m + 1)) * Q
        r, q = _iterative_classical_gram_schmidt(Q_valid, q)
        alpha = r[i]
        beta = jnp.linalg.norm(q)

        ## handle the case where a Krylov subspace has been found (rare)
        #def _gen_random_ortho_vec():
        #    _, _q = _iterative_classical_gram_schmidt(
        #        Q_valid, jax.random.normal(jax.random.PRNGKey(i+1), shape=(n,)))
        #    return _q / jnp.linalg.norm(_q)

        #q = lax.cond(
        #    beta < tolerance,
        #    lambda _: _gen_random_ortho_vec(),
        #    lambda _: q / beta,
        #    None)

        # the following works in most cases
        q = q / beta

        Q = Q.at[:, i+1].set(q)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)
        return (Q, alphas, betas)

    (Q, alpha, beta) = lax.fori_loop(k, m, body_fun, (Q, alpha, beta))
    return Q, alpha, beta


def _build_arrowhead_matrix(alpha, beta, k):
    """Build the "arrowhead" shaped matrix needed for thick-restart Lanczos."""
    m = alpha.size
    upper_triangle = (
            jnp.einsum('i,j', beta * (k > jnp.arange(m)), k == jnp.arange(m))
            + jnp.diag(beta[:-1] * (k <= jnp.arange(m - 1)), k=1)
    )
    return upper_triangle + upper_triangle.T + jnp.diag(alpha)


def _pick_saved_ritz_values(lambda_, num_converged, num_desired):
    """Maximize the effective gap ratio for the smallest eigenvalue.

    Implements the first heuristic choice described under "Restarting strategies"
    in the referenced paper above.

    Args:
        lambda_: approximate eigenvalue values (Ritz values).
        num_converged: number of converged eigenvalues.
        num_desired: number of desired eigenvalues.

    Returns:
        saved: a boolean index marking Ritz values to save.
        gamma: array of gamma values, for debugging.
    """
    # TODO(shoyer): verify that these heuristics make sense for other
    # applications beyond eletronic structure theory
    m = lambda_.size
    n_eig = num_desired
    n_c = num_converged
    delta = jnp.minimum(m - n_eig, 2 * (m - n_c) // 5)

    k_range = jnp.arange(m)
    kr = k_range + delta
    lambda_target = lambda_[num_converged]

    gamma = jnp.where(
            kr <= m, (lambda_ - lambda_target) / (lambda_[kr] - lambda_target), 0)
    k_left = jnp.argmax(gamma)
    k_right = k_left + delta
    return (k_left >= k_range) | (k_right < k_range), gamma


_ThickRestartState = namedtuple(
        '_ThickRestartState',
        'Q alpha beta restart num_saved num_converged'
)


def _thick_restart_lanczos(
        n,
        C,
        A_data,
        A_indices,
        adjoint_left_vec,
        q0,
        num_desired,
        inner_iterations,
        max_restarts,
        tolerance,
):
    """Core implementation of the Lanczos method with thick-restart."""

    m = inner_iterations

    def cond_fun(state):
        return state.num_converged < num_desired

    def body_fun(state):
        Q, alpha, beta, restart, k, _ = state

        Q, alpha, beta = _lanczos_restart(
            n, C, A_data, A_indices, adjoint_left_vec, k, m, Q, alpha, beta, tolerance)

        T = _build_arrowhead_matrix(alpha, beta, k)
        ritz_values, Y = jnp.linalg.eigh(T)

        residual_norm = beta[-1] * abs(Y[-1, :])
        converged = residual_norm < tolerance
        num_converged = jnp.where(
                converged.all(), inner_iterations, jnp.argmin(converged))

        saved, _ = _pick_saved_ritz_values(
                ritz_values, num_converged, num_desired)
        k_next = jnp.sum(saved)
        saved_to_front = jnp.argsort(~saved)
        ritz_values = ritz_values[saved_to_front]
        Y = Y[:, saved_to_front]

        updated = k_next > jnp.arange(m)
        alpha_hat = jnp.where(updated, ritz_values, jnp.zeros_like(alpha))
        beta_hat = jnp.where(updated, beta[-1] * Y[-1, :], jnp.zeros_like(beta))
        Q_hat = Q.at[:, :-1].set(jnp.where(updated, _dot(Q[:, :-1], Y), Q[:, :-1]))
        Q_hat = Q_hat.at[:, k_next].set(Q[:, -1])

        return _ThickRestartState(
                Q_hat, alpha_hat, beta_hat, restart + 1, k_next, num_converged)

    Q = jnp.zeros((n, m+1)).at[:, 0].set(q0)
    alpha = jnp.zeros(m)
    beta = jnp.zeros(m)

    return while_loop(
        cond_fun,
        body_fun,
        _ThickRestartState(Q, alpha, beta, 0, 0, 0),
        max_restarts,
        unroll=True,
        jit=True)


def eigsh_smallest(
    n,
    C,
    A_data,
    A_indices,
    adjoint_left_vec,
    q0,
    num_desired,
    inner_iterations,
    max_restarts,
    tolerance
):
    """Find the `num_desired` smallest eigenvalues of the linear map A.

    Args:
        n: dimension of operator and eigenvectors.
        C: matrix parameterizing the SDP objective.
        A_data: values in constraint tensor located at `A_indices`.
        A_indices: corresponding locations of `A_data` in constraint tensor.
        adjoint_left_vec: vector to apply adjoint of constraint tensor.
        num_desired: number of desired smallest eigenvalues.
        inner_iterations: number of inner Lanczos iterations to use.
        max_restarts: maximum number of Lanczos restarts to allow. By default, there
            is no limit.
        tolerance: absolute residual norm of the eigenvalue problem
            ||A v_i - lambda_i v_i|| used for determining convergence.

    Returns:
        eigenvalues: array of shape (num_desired,)
        eigenvectors: array of shape (q0.size, num_desired).
    """
    state = _thick_restart_lanczos(
        n,
        C,
        A_data,
        A_indices,
        adjoint_left_vec,
        q0,
        num_desired,
        inner_iterations,
        max_restarts,
        tolerance)
    eigenvalues = state.alpha[:num_desired]
    eigenvectors = state.Q[:, :num_desired]

    return eigenvalues, eigenvectors
