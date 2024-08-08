import argparse
from collections import deque
import copy
from dataclasses import dataclass
import h5py
import sys
import time

import cvxpy as cp # type: ignore
import git
import jax
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import json
import numpy as np
from IPython import embed

from solver.usbs import usbs
from utils.common import SDPState, str2bool


class BnBSparsePCA(object):

    def __init__(self, cov_mx: np.array, k: int, hparams) -> None:
        self.cov_mx = cov_mx
        self.p = cov_mx.shape[0]
        self.k = k
        self.hparams = hparams

        # construct the problem built for cvxpy
        self.build_cvxpy_problem()

        # construct the problem using the sdp_state created in this repo
        self.build_sdp_state()

    def build_cvxpy_problem(self) -> None:
        # Create a symmetric matrix variable.
        self.X = cp.Variable((self.p, self.p), symmetric=True)

        # Create parameter for row/column selection for sparse PCA
        self.z = cp.Variable((self.p,), nonneg=True)

        # Create variables for 
        self.z_LB = cp.Parameter((self.p,), nonneg=True)
        self.z_UB = cp.Parameter((self.p,), nonneg=True)

        # Define the CVXPY problem.
        # The operator >> denotes matrix inequality.
        constraints = [self.X >> 0]
        constraints += [cp.trace(self.X) == 1]
        constraints += [cp.sum(self.z) <= k]
        constraints += [self.z >= self.z_LB]
        constraints += [self.z <= self.z_UB]

        M = (np.eye(self.p) + np.ones((self.p, self.p))) / 2
        constraints += [cp.abs(self.X) <= cp.multiply(M, self.z[:, None])]
        constraints += [cp.sum(cp.abs(self.X), axis=0) <= cp.sqrt(k) * self.z]
        self.prob = cp.Problem(cp.Maximize(cp.trace(self.cov_mx @ self.X)), constraints)

    def build_sdp_state(self):

        # create C adding room for z to be encoded in the diagonal of X
        # and extra p^2 slack variables for sum of absolute values
        C = BCOO.fromdense(self.cov_mx)
        C = BCOO((C.data, C.indices), shape=(2*self.p + self.p**2, 2*self.p + self.p**2))

        # tr(X) == 1
        A_indices = jnp.hstack((jnp.zeros((self.p, 1)), jnp.tile(jnp.arange(self.p), (2, 1)).T))
        A_data = jnp.ones((self.p,))
        b = jnp.asarray([1])
        b_ineq_mask = jnp.asarray([0])

        # sum(z) <= k
        A_indices = jnp.concatenate(
            [A_indices,
             jnp.hstack((jnp.ones((self.p, 1)), self.p + jnp.tile(jnp.arange(self.p), (2, 1)).T))])
        A_data = jnp.concatenate([A_data, jnp.ones((self.p,))])
        b = jnp.concatenate([b, jnp.asarray([self.k])])
        b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.asarray([1])])

        # TODO: rewrite this without using loops
        # `b.shape[0]` gives you the number of constraints so far
        # |X_ij| <= M_ij * z_i
        _A_indices = []
        _A_data = []
        _b = []
        _b_ineq_mask = []

        constraint_idx = b.shape[0]
        for i in range(self.p):
            for j in range(self.p):
                _A_indices.append([constraint_idx, i, j])
                _A_data.append(1.0)
                _A_indices.append([constraint_idx, i+self.p, i+self.p])
                _A_data.append(-1.0 if i == j else -0.5)
                _b.append(0)
                _b_ineq_mask.append(1)
                constraint_idx += 1

                _A_indices.append([constraint_idx, i, j])
                _A_data.append(-1.0)
                _A_indices.append([constraint_idx, i+self.p, i+self.p])
                _A_data.append(-1.0 if i == j else -0.5)
                _b.append(0)
                _b_ineq_mask.append(1)
                constraint_idx += 1

        # next constraint: cp.sum(cp.abs(self.X), axis=0) <= cp.sqrt(k) * self.z
        # split this constraint into two parts:
        # (1) -Y_ij <= X_ij <= Y_ij
        slack_idx = 2 * self.p
        for i in range(self.p):
            for j in range(self.p):
                # (1 - LHS)
                _A_indices.append([constraint_idx, i, j])
                _A_data.append(-1.0)
                _A_indices.append([constraint_idx, slack_idx, slack_idx])
                _A_data.append(-1.0)
                _b.append(0)
                _b_ineq_mask.append(1)
                constraint_idx += 1

                # (1 - RHS)
                _A_indices.append([constraint_idx, i, j])
                _A_data.append(1.0)
                _A_indices.append([constraint_idx, slack_idx, slack_idx])
                _A_data.append(-1.0)
                _b.append(0)
                _b_ineq_mask.append(1)
                constraint_idx += 1

                slack_idx += 1

        # (2) sum_j Y_ij <= sqrt(k) z_i
        slack_idx = 2 * self.p
        for i in range(self.p):
            for j in range(self.p):
                _A_indices.append([constraint_idx, slack_idx, slack_idx])
                _A_data.append(1.0)
                slack_idx += 1

            _A_indices.append([constraint_idx, self.p + i, self.p + i])
            _A_data.append(-jnp.sqrt(self.k))
            _b.append(0)
            _b_ineq_mask.append(0)
            constraint_idx += 1

        A_indices = jnp.concatenate([A_indices, jnp.asarray(_A_indices)])
        A_data = jnp.concatenate([A_data, jnp.asarray(_A_data)])
        b = jnp.concatenate([b, jnp.asarray(_b)])
        b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.asarray(_b_ineq_mask)])

        # save the LB and UB constraints for last ... we need to readily have access to these
        _A_indices = jnp.stack(
            [b.shape[0] + jnp.arange(self.p),
             self.p + jnp.arange(self.p),
             self.p + jnp.arange(self.p)]).T
        _A_data = jnp.full((self.p,), -1.0)
        _b = jnp.zeros((self.p,))
        _b_ineq_mask = jnp.ones((self.p,))

        A_indices = jnp.concatenate([A_indices, jnp.asarray(_A_indices)])
        A_data = jnp.concatenate([A_data, jnp.asarray(_A_data)])
        b = jnp.concatenate([b, jnp.asarray(_b)])
        b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.asarray(_b_ineq_mask)])

        _A_indices = jnp.stack(
            [b.shape[0] + jnp.arange(self.p),
             self.p + jnp.arange(self.p),
             self.p + jnp.arange(self.p)]).T
        _A_data = jnp.full((self.p,), 1.0)
        _b = jnp.ones((self.p,))
        _b_ineq_mask = jnp.ones((self.p,))

        A_indices = jnp.concatenate([A_indices, jnp.asarray(_A_indices)])
        A_data = jnp.concatenate([A_data, jnp.asarray(_A_data)])
        b = jnp.concatenate([b, jnp.asarray(_b)])
        b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.asarray(_b_ineq_mask)])

        # fix the indexing types
        A_indices = A_indices.astype(int)

        # TODO: figure out how to make this smaller
        X = jnp.zeros(C.shape)
        P = None
        Omega = None

        y = jnp.zeros_like(b)
        z = jnp.zeros_like(b)
        tr_X = 0.0
        primal_obj = 0.0

        SCALE_C = 1.0
        SCALE_X = 1.0
        SCALE_A = jnp.ones_like(b)

        self.init_sdp_state = SDPState(
            C=C,
            A_indices=A_indices,
            A_data=A_data,
            b=b,
            b_ineq_mask=b_ineq_mask,
            X=X,
            P=P,
            Omega=Omega,
            y=y,
            z=z,
            tr_X=tr_X,
            primal_obj=primal_obj,
            SCALE_C=SCALE_C,
            SCALE_X=SCALE_X,
            SCALE_A=SCALE_A)


    def branch_and_bound(self, max_iters) -> None:

        sdp_state = self.init_sdp_state

        trace_ub = float(self.k + 1 + self.k**2)
        sdp_state = usbs(
            sdp_state=sdp_state,
            n=sdp_state.C.shape[0],
            m=sdp_state.b.shape[0],
            trace_ub=trace_ub,
            trace_factor=self.hparams.trace_factor,
            rho=self.hparams.rho,
            beta=self.hparams.beta,
            k_curr=self.hparams.k_curr,
            k_past=self.hparams.k_past,
            max_iters=self.hparams.max_iters,
            max_time=self.hparams.max_time,
            obj_gap_eps=self.hparams.obj_gap_eps,
            infeas_gap_eps=self.hparams.infeas_gap_eps,
            max_infeas_eps=self.hparams.max_infeas_eps,
            lanczos_inner_iterations=min(sdp_state.C.shape[0], self.hparams.lanczos_max_inner_iterations),
            lanczos_max_restarts=self.hparams.lanczos_max_restarts,
            subprob_eps=self.hparams.subprob_eps,
            subprob_max_iters=self.hparams.subprob_max_iters,
            cond_exp_base=self.hparams.cond_exp_base,
            callback_fn=None,
            callback_static_args=None,
            callback_nonstatic_args=None)
        
        embed()
        exit()


        @dataclass
        class BnBNode:
            force_zero: set
            force_one: set

        int_solns = {}  # hash map for integer solutions: set -> obj
        prospective_nodes = set()  # keep track of all BnB nodes added to queue
        all_indices = set(range(self.p))
        active_list = deque([BnBNode(set(), set())])
        max_int_obj = 0.0
        max_int_assign = None

        num_nodes_visited = 0
        while len(active_list) > 0 and num_nodes_visited < max_iters:
            num_nodes_visited += 1
            #node = active_list.pop()
            node = active_list.popleft()
            self.z_LB.value = np.zeros((self.p,))
            self.z_UB.value = np.ones((self.p,))
            self.z_LB.value[list(node.force_one)] = 1.0
            self.z_UB.value[list(node.force_zero)] = 0.0
            
            start_time = time.time()
            self.prob.solve(solver=cp.SCS, warm_start=True)
            print("warm start time: ", time.time() - start_time)

            start_time = time.time()
            self.prob.solve(solver=cp.SCS, warm_start=False)
            print("cold start time: ", time.time() - start_time)

            # if num ones in z_LB == k, don't branch
            if len(node.force_one) == self.k or len(node.force_zero) == self.p - self.k:
                if self.prob.value > max_int_obj:
                    max_int_obj = self.prob.value
                continue

            # if relaxed max objective is lower than max_valid_obj don't branch
            if self.prob.value <= max_int_obj:
                continue

            # push branching nodes onto deque
            for i in ((all_indices - node.force_zero) - node.force_one):
                # force i to zero node
                force_zero_node = copy.deepcopy(node)
                force_zero_node.force_zero.add(i)
                hashable_force_zero = (frozenset(force_zero_node.force_zero),
                                       frozenset(force_zero_node.force_one))
                if hashable_force_zero not in prospective_nodes:
                    active_list.append(force_zero_node)
                    prospective_nodes.add(hashable_force_zero)

            # get rounded solution
            force_one_indices = np.argpartition(self.z.value, self.p - self.k)[-self.k:]
            fset_force_one_indices = frozenset(force_one_indices)
            if fset_force_one_indices not in int_solns.keys():
                self.z_LB.value[np.argpartition(self.z.value, self.p - self.k)[-self.k:]] = 1.0
                self.prob.solve(solver=cp.SCS, warm_start=False)
                int_solns[fset_force_one_indices] = self.prob.value
                if self.prob.value > max_int_obj:
                    max_int_obj = self.prob.value

        print(num_nodes_visited)

        return max_int_obj, max_int_assign


def get_hparams():
    parser = argparse.ArgumentParser() 
    #parser.add_argument("--data_path", type=str, required=True, help="path to mat file")
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
    parser.add_argument("--cond_exp_base", type=float, default=1.0,
                        help="incremental multiplicative factor on number of iterations "
                             "before next condition check and callback")
    parser.add_argument("--k_curr", type=int, default=1,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--k_past", type=int, default=0,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--trace_factor", type=float, default=1.0,
                        help="how much space to give trace")
    parser.add_argument("--rho", type=float, default=0.5,
                        help="proximal parameter")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="sufficient decrease parameter")
    parser.add_argument("--warm_start_strategy", type=str,
                        choices=["implicit", "explicit", "dual_only", "none"],
                        help="warm-start strategy to use")
    parser.add_argument("--no_rounding", type=str2bool, nargs='?', const=True, default=False,
                        help="turn rounding off during optimization")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":

    # essential for running USBS or CGAL
    jax.config.update("jax_enable_x64", True)

    hparams = get_hparams()
    hparams.solver = "usbs"
    print("cmd: ", " ".join(["python"] + sys.argv))
    print("git sha: ", git.Repo(search_parent_directories=True).head.object.hexsha)
    print("hparams: ", json.dumps(vars(hparams), indent=4))

    # some small test data
    pitprops = [[1,0.954,0.364,0.342,-0.129,0.313,0.496,0.424,0.592,0.545,0.084,-0.019,0.134],
                [0.954,1,0.297,0.284,-0.118,0.291,0.503,0.419,0.648,0.569,0.076,-0.036,0.144],
                [0.364,0.297,1,0.882,-0.148,0.153,-0.029,-0.054,0.125,-0.081,0.162,0.22,0.126],
                [0.342,0.284,0.882,1,0.22,0.381,0.174,-0.059,0.137,-0.014,0.097,0.169,0.015],
                [-0.129,-0.118,-0.148,0.22,1,0.364,0.296,0.004,-0.039,0.037,-0.091,-0.145,-0.208],
                [0.313,0.291,0.153,0.381,0.364,1,0.813,0.09,0.211,0.274,-0.036,0.024,-0.329],
                [0.496,0.503,-0.029,0.174,0.296,0.813,1,0.372,0.465,0.679,-0.113,-0.232,-0.424],
                [0.424,0.419,-0.054,-0.059,0.004,0.09,0.372,1,0.482,0.557,0.061,-0.357,-0.202],
                [0.592,0.648,0.125,0.137,-0.039,0.211,0.465,0.482,1,0.526,0.085,-0.127,-0.076],
                [0.545,0.569,-0.081,-0.014,0.037,0.274,0.679,0.557,0.526,1,-0.319,-0.368,-0.291],
                [0.084,0.076,0.162,0.097,-0.091,-0.036,-0.113,0.061,0.085,-0.319,1,0.029,0.007],
                [-0.019,-0.036,0.22,0.169,-0.145,0.024,-0.232,-0.357,-0.127,-0.368,0.029,1,0.184],
                [0.134,0.144,0.126,0.015,-0.208,-0.329,-0.424,-0.202,-0.076,-0.291,0.007,0.184,1]]


    normwine=[[1.0,0.0943969,0.211545,-0.310235,0.270798,0.289101,0.236815,-0.155929,0.136698,0.546364,-0.0717472,0.0723432,0.64372],
              [0.0943969,1.0,0.164045,0.2885,-0.0545751,-0.335167,-0.411007,0.292977,-0.220746,0.248985,-0.561296,-0.36871,-0.192011],
              [0.211545,0.164045,1.0,0.443367,0.286587,0.12898,0.115077,0.18623,0.00965194,0.258887,-0.0746669,0.00391123,0.223626],
              [-0.310235,0.2885,0.443367,1.0,-0.0833331,-0.321113,-0.35137,0.361922,-0.197327,0.018732,-0.273955,-0.276769,-0.440597],
              [0.270798,-0.0545751,0.286587,-0.0833331,1.0,0.214401,0.195784,-0.256294,0.236441,0.19995,0.0553982,0.0660039,0.393351],
              [0.289101,-0.335167,0.12898,-0.321113,0.214401,1.0,0.864564,-0.449935,0.612413,-0.0551364,0.433681,0.699949,0.498115],
              [0.236815,-0.411007,0.115077,-0.35137,0.195784,0.864564,1.0,-0.5379,0.652692,-0.172379,0.543479,0.787194,0.494193],
              [-0.155929,0.292977,0.18623,0.361922,-0.256294,-0.449935,-0.5379,1.0,-0.365845,0.139057,-0.26264,-0.50327,-0.311385],
              [0.136698,-0.220746,0.00965194,-0.197327,0.236441,0.612413,0.652692,-0.365845,1.0,-0.0252499,0.295544,0.519067,0.330417],
              [0.546364,0.248985,0.258887,0.018732,0.19995,-0.0551364,-0.172379,0.139057,-0.0252499,1.0,-0.521813,-0.428815,0.3161],
              [-0.0717472,-0.561296,-0.0746669,-0.273955,0.0553982,0.433681,0.543479,-0.26264,0.295544,-0.521813,1.0,0.565468,0.236183],
              [0.0723432,-0.36871,0.00391123,-0.276769,0.0660039,0.699949,0.787194,-0.50327,0.519067,-0.428815,0.565468,1.0,0.312761],
              [0.64372,-0.192011,0.223626,-0.440597,0.393351,0.498115,0.494193,-0.311385,0.330417,0.3161,0.236183,0.312761,1.0]]

    #f = h5py.File("../ScalableSPCA.jl/data/miniBoone.jld", "r")
    #normMiniBooNE = np.asarray(f["normMiniBooNE"])

    cov_mx = np.array(pitprops)
    k = 5

    misdo_solver = BnBSparsePCA(cov_mx=cov_mx, k=k, hparams=hparams)
    misdo_solver.branch_and_bound(max_iters=100)  # how do we store the solution? in the object? return it?
