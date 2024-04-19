import argparse
import git
import jax
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import json
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import LassoLars
import sys

from solver.specbm import specbm
from utils.common import SDPState, scale_sdp_state

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--max_iters", type=int, required=True,
                        help="number of iterations to run solver")
    parser.add_argument("--max_time", type=float, default=jnp.inf,
                        help="max running time in seconds for one solve")
    parser.add_argument("--obj_gap_eps", type=float, default=-jnp.inf,
                        help="early stop if obj_gap is less than this number")
    parser.add_argument("--infeas_gap_eps", type=float, default=-jnp.inf,
                        help="early stop if infeas_gap is less than this number")
    parser.add_argument("--max_infeas_eps", type=float, default=-jnp.inf,
                        help="early stop if max_infeas is less than this number")
    parser.add_argument("--lanczos_max_restarts", type=int, default=100,
                        help="number of restarts to use for thick restart lanczos")
    parser.add_argument("--lanczos_max_inner_iterations", type=int, default=32,
                        help="number of inner iterations to use for thick restart lanczos")
    parser.add_argument("--subprob_eps", type=float, default=1e-7,
                        help="error tolerance for IPM, alternating minimization, and lanczos")
    parser.add_argument("--subprob_max_iters", type=int, default=15,
                        help="max iters for IPM and alternating minimization")
    parser.add_argument("--k_curr", type=int, default=1,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--k_past", type=int, default=0,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--trace_factor", type=float, default=1.0,
                        help="how much space to give trace")
    parser.add_argument("--rho", type=float, default=0.1,
                        help="proximal parameter")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="sufficient decrease parameter")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":

    seed = 0
    hparams = get_hparams()
    hparams.solver = "specbm"
    print("cmd: ", " ".join(["python"] + sys.argv))
    print("git sha: ", git.Repo(search_parent_directories=True).head.object.hexsha)
    print("hparams: ", json.dumps(vars(hparams), indent=4))

    # specbm requires double floating point precision for stable functionality
    jax.config.update("jax_enable_x64", True)

    with open("activations.pkl", "rb") as f:
        activation_batches = pickle.load(f)

    activations = np.concatenate(activation_batches, axis=0)

    activation_dim = activations.shape[1]
    dict_size = 64 * activation_dim

    #dictionary = jax.random.normal(jax.random.PRNGKey(seed), shape=(activation_dim, dict_size))

    dictionary = activations[:dict_size,:].T
    dictionary /= jnp.linalg.norm(dictionary, axis=0, keepdims=True)

    # TEST
    dictionary = dictionary.at[:,1].set(dictionary[:,0])
    dictionary = dictionary.at[256:, 0].set(0)
    dictionary = dictionary.at[:256, 1].set(0)

    #### START: constant throughout optimization ####

    # since specbm is designed for maximization problems we implement negative identity matrix
    C = BCOO(
        (jnp.full((dict_size,), -1), jnp.tile(jnp.arange(dict_size), (2, 1)).T),
        shape=(dict_size, dict_size))

    _dim0 = jnp.tile(jnp.arange(activation_dim), (dict_size, 1)).T.reshape(-1,)[:, None]
    _dim1 = jnp.tile(jnp.arange(dict_size), (activation_dim, 1)).reshape(-1,)[:, None]
    _dim2 = jnp.tile(jnp.arange(dict_size), (activation_dim, 1)).reshape(-1,)[:, None]
    A_indices = jnp.hstack([_dim0, _dim1, _dim2])

    #### END: constant throughout optimization ####

    #### START: dynamic throughout optimization ####

    actv = activations[0]

    # LARS-Lasso
    #  model = LassoLars(alpha=0.001)
    #  model.fit(dictionary, actv)
    #  model.coef_
    

    # IDEA: should we learn this with CVXPY as well? Will this be slower?

    # the only variables that change in every iteration are `A_data` and `b`
    A_data=jnp.array(dictionary.reshape(-1,))

    n = C.shape[0]
    m = activation_dim
    SCALE_X = 1.0
    SCALE_C = 1.0
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    maxiter = np.iinfo(np.int32).max + 1
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False, maxiter=np.iinfo(np.int32).max)[0])
    SCALE_A /= norm_A

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=jnp.array(actv),
        b_ineq_mask=jnp.zeros((activation_dim,)),
        X=None,
        P=jnp.zeros((dict_size,)),
        Omega=None,
        y=jnp.zeros((activation_dim,)),
        z=jnp.zeros((activation_dim,)),
        tr_X=0.0,
        primal_obj=0.0,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    # scale the state
    sdp_state = scale_sdp_state(sdp_state)

    trace_ub = hparams.trace_factor * float(sdp_state.C.shape[0]) * sdp_state.SCALE_X

    sdp_state = specbm(
        sdp_state=sdp_state,
        n=sdp_state.C.shape[0],
        m=sdp_state.b.shape[0],
        trace_ub=trace_ub,
        trace_factor=hparams.trace_factor,
        rho=hparams.rho,
        beta=hparams.beta,
        k_curr=hparams.k_curr,
        k_past=hparams.k_past,
        max_iters=hparams.max_iters,
        max_time=hparams.max_time,
        obj_gap_eps=hparams.obj_gap_eps,
        infeas_gap_eps=hparams.infeas_gap_eps,
        max_infeas_eps=hparams.max_infeas_eps,
        lanczos_inner_iterations=min(sdp_state.C.shape[0], hparams.lanczos_max_inner_iterations),
        lanczos_max_restarts=hparams.lanczos_max_restarts,
        subprob_eps=hparams.subprob_eps,
        subprob_max_iters=hparams.subprob_max_iters,
        callback_fn=None,
        callback_static_args=None,
        callback_nonstatic_args=None)

    #### END: dynamic throughout optimization ####
    embed()
    exit()