import argparse
import copy
import git
from itertools import product
import json
import logging
import os
import pickle
import sys
import time

import jax
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import numba as nb
from numba.typed import List
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from scipy.sparse import vstack as sp_vstack
from scipy.sparse.csgraph import laplacian
from sklearn import random_projection
from sklearn.metrics import adjusted_rand_score as rand_idx
from sklearn.metrics import homogeneity_completeness_v_measure as cluster_f1

from solver.cgal import cgal
from solver.specbm import specbm
from utils.common import unscale_sdp_state, SDPState
from utils.ecc_helpers import (initialize_state,
                               cold_start_add_constraint,
                               warm_start_add_constraint,
                               create_sparse_laplacian)
from utils.trellis import Trellis

from IPython import embed


class EccClusterer(object):

    def __init__(self,
                 edge_weights: coo_matrix,
                 features: np.ndarray,
                 hparams: argparse.Namespace):

        self.hparams = hparams
        self.edge_weights = edge_weights
        self.edge_weights.data
        self.sparse_laplacian = create_sparse_laplacian(edge_weights=edge_weights, eps=0.8)

        self.features = features
        self.n = self.features.shape[0]
        self.ecc_constraints = []
        self.ecc_mx = None
        self.incompat_mx = None

        C = BCOO.from_scipy_sparse(self.sparse_laplacian).astype(float)
        self.cold_start_sdp_state = initialize_state(C=C, sketch_dim=hparams.sketch_dim)
        self.warm_start_sdp_state = copy.deepcopy(self.cold_start_sdp_state)


    def add_constraint(self, ecc_constraint: csr_matrix):
        self.ecc_constraints.append(ecc_constraint)
        self.ecc_mx = sp_vstack(self.ecc_constraints)
        self.n += 1

        num_points = self.features.shape[0]
        num_ecc = len(self.ecc_constraints)

        # "negative" SDP constraints
        uni_feats = sp_vstack([self.features, self.ecc_mx])
        self.incompat_mx = np.zeros(
                (num_points+num_ecc, num_ecc), dtype=bool)
        self._set_incompat_mx(num_points+num_ecc,
                              num_ecc,
                              uni_feats.indptr,
                              uni_feats.indices,
                              uni_feats.data,
                              self.ecc_mx.indptr,
                              self.ecc_mx.indices,
                              self.ecc_mx.data,
                              self.incompat_mx)
        ortho_indices = [(a, b+num_points)
                for a, b in zip(*np.where(self.incompat_mx)) if b == num_ecc - 1]

        # "positive" SDP constraints
        bin_features = self.features.astype(bool).tocsc()
        pos_ecc_mx = (self.ecc_mx > 0)
        (ecc_indices,
         points_indptr,
         points_indices) = self._get_feat_satisfied_hyperplanes(
                 bin_features.indptr,
                 bin_features.indices,
                 pos_ecc_mx.indptr,
                 pos_ecc_mx.indices,
                 self.incompat_mx)
        ecc_indices = [x + num_points for x in ecc_indices]
        points_indptr = list(points_indptr)
        points_indices = list(points_indices)

        sum_gt_one_constraints = []
        for idx, i in enumerate(ecc_indices):
            j_s = points_indices[points_indptr[idx]: points_indptr[idx+1]]
            if i == self.n - 1:
                sum_gt_one_constraints.append([(i,j) for j in j_s])

        self.warm_start_sdp_state = warm_start_add_constraint(
            old_sdp_state=self.cold_start_sdp_state,
            ortho_indices=ortho_indices,
            sum_gt_one_constraints=sum_gt_one_constraints,
            prev_pred_clusters=jnp.array(self.prev_pred_clusters),
            rho=self.hparams.rho,
            sketch_dim=-1)
        self.cold_start_sdp_state = cold_start_add_constraint(
            old_sdp_state=self.cold_start_sdp_state,
            ortho_indices=ortho_indices,
            sum_gt_one_constraints=sum_gt_one_constraints,
            sketch_dim=-1)


    @staticmethod
    @nb.njit(parallel=True)
    def _set_incompat_mx(n: int,
                         m: int,
                         indptr_a: np.ndarray,
                         indices_a: np.ndarray,
                         data_a: np.ndarray,
                         indptr_b: np.ndarray,
                         indices_b: np.ndarray,
                         data_b: np.ndarray,
                         incompat_mx: np.ndarray):
        for i in nb.prange(n):
            for j in nb.prange(m):
                ptr_a = indptr_a[i]
                ptr_b = indptr_b[j]
                next_ptr_a = indptr_a[i+1]
                next_ptr_b = indptr_b[j+1]
                while ptr_a < next_ptr_a and ptr_b < next_ptr_b:
                    if indices_a[ptr_a] < indices_b[ptr_b]:
                        ptr_a += 1
                    elif indices_a[ptr_a] > indices_b[ptr_b]:
                        ptr_b += 1
                    else:
                        if data_a[ptr_a] * data_b[ptr_b] == -1:
                            incompat_mx[i, j] = True
                            break
                        ptr_a += 1
                        ptr_b += 1

    @staticmethod
    @nb.njit
    def _get_feat_satisfied_hyperplanes(feats_indptr: np.ndarray,
                                        feats_indices: np.ndarray,
                                        pos_ecc_indptr: np.ndarray,
                                        pos_ecc_indices: np.ndarray,
                                        incompat_mx: np.ndarray):
        ecc_indices = List.empty_list(nb.int64)
        points_indptr = List.empty_list(nb.int64)
        points_indices = List.empty_list(nb.int64)

        points_indptr.append(0)
        for ecc_idx in range(pos_ecc_indptr.size-1):
            pos_feat_ptr = pos_ecc_indptr[ecc_idx]
            next_pos_feat_ptr = pos_ecc_indptr[ecc_idx+1]
            while pos_feat_ptr < next_pos_feat_ptr:
                ecc_indices.append(ecc_idx)
                points_indptr.append(points_indptr[-1])
                feat_id = pos_ecc_indices[pos_feat_ptr]
                point_ptr = feats_indptr[feat_id]
                next_point_ptr = feats_indptr[feat_id+1]

                while point_ptr < next_point_ptr:
                    point_idx = feats_indices[point_ptr]
                    if not incompat_mx[point_idx, ecc_idx]:
                        points_indptr[-1] += 1
                        points_indices.append(point_idx)
                    point_ptr += 1
                pos_feat_ptr += 1

        return (ecc_indices, points_indptr, points_indices)


    def _call_sdp_solver(self, sdp_state: SDPState, solver_name: str) -> None:
        trace_ub = (self.hparams.trace_factor
                    * float(sdp_state.C.shape[0])
                    * sdp_state.SCALE_X)

        print(">>>>> START: ", solver_name)
        if "specbm" in solver_name:
            out_sdp_state = specbm(
                sdp_state=sdp_state,
                n=sdp_state.C.shape[0],
                m=sdp_state.b.shape[0],
                trace_ub=trace_ub,
                trace_factor=self.hparams.trace_factor,
                rho=self.hparams.rho,
                beta=self.hparams.beta,
                k_curr=min(self.hparams.k_curr, sdp_state.C.shape[0]),
                k_past=self.hparams.k_past,
                max_iters=self.hparams.max_iters,
                max_time=self.hparams.max_time,
                obj_gap_eps=self.hparams.obj_gap_eps,
                infeas_gap_eps=self.hparams.infeas_gap_eps,
                max_infeas_eps=self.hparams.max_infeas_eps,
                lanczos_inner_iterations=min(sdp_state.C.shape[0], 32),
                lanczos_max_restarts=self.hparams.lanczos_max_restarts,
                subprob_eps=float(self.hparams.subprob_eps * sdp_state.SCALE_C * sdp_state.SCALE_X),
                subprob_max_iters=hparams.subprob_max_iters,
                callback_fn=None,
                callback_static_args=None,
                callback_nonstatic_args=None)
        elif "cgal" in solver_name:
            out_sdp_state = cgal(
                sdp_state=sdp_state,
                n=sdp_state.C.shape[0],
                m=sdp_state.b.shape[0],
                trace_ub=trace_ub,
                beta0=1.0,
                max_iters=self.hparams.max_iters,
                max_time=self.hparams.max_time,
                obj_gap_eps=self.hparams.obj_gap_eps,
                infeas_gap_eps=self.hparams.infeas_gap_eps,
                max_infeas_eps=self.hparams.max_infeas_eps,
                lanczos_inner_iterations=min(sdp_state.C.shape[0], 32),
                lanczos_max_restarts=self.hparams.lanczos_max_restarts,
                subprob_eps=float(self.hparams.subprob_eps * sdp_state.SCALE_C * sdp_state.SCALE_X),
                callback_fn=None,
                callback_static_args=None,
                callback_nonstatic_args=None)
        else:
            raise ValueError("Unknown solver name.")
        print("<<<<< END: ", solver_name)

        return out_sdp_state


    def build_and_solve_sdp(self):
        _ = self._call_sdp_solver(self.cold_start_sdp_state, "cgal/cold")
        _ = self._call_sdp_solver(self.warm_start_sdp_state, "cgal/warm")
        self.cold_start_sdp_state = self._call_sdp_solver(self.cold_start_sdp_state, "specbm/cold")
        _warm_end_state = self._call_sdp_solver(self.warm_start_sdp_state, "specbm/warm")

        #if len(self.ecc_constraints) == 1:
        #    embed()
        #    exit()

        unscaled_state = unscale_sdp_state(self.cold_start_sdp_state)
        sdp_obj_value = float(jnp.trace(-unscaled_state.C @ unscaled_state.X))
        pw_probs = np.array(jnp.clip(unscaled_state.X, a_min=0.0, a_max=1.0))
        
        return sdp_obj_value, pw_probs

    def build_trellis(self, pw_probs: np.ndarray):
        t = Trellis(adj_mx=pw_probs)
        t.fit()
        return t

    def get_intra_cluster_energy(self, leaves: np.ndarray):
        row_mask = np.isin(self.edge_weights.row, leaves)
        col_mask = np.isin(self.edge_weights.col, leaves)
        data_mask = row_mask & col_mask
        return np.sum(self.edge_weights.data[data_mask])

    def get_num_ecc_sat(self, leaves: np.ndarray, num_points: int):
        point_leaves = leaves[leaves < num_points]
        ecc_indices = leaves[leaves >= num_points] - num_points
        if ecc_indices.squeeze().size == 0:
            return 0
        feats = get_cluster_feats(self.features[point_leaves])
        ecc_avail = self.ecc_mx[ecc_indices]
        to_satisfy = (ecc_avail > 0).sum(axis=1)
        num_ecc_sat = ((feats @ ecc_avail.T).T == to_satisfy).sum()
        return num_ecc_sat

    @staticmethod
    @nb.njit(parallel=True)
    def get_membership_data(indptr: np.ndarray,
                            indices: np.ndarray):
        data = np.empty(indices.shape, dtype=np.int64)
        for i in nb.prange(indptr.size-1):
            for j in nb.prange(indptr[i], indptr[i+1]):
                data[j] = i
        return data

    @staticmethod
    @nb.njit
    def merge_memberships(lchild_indices: np.ndarray,
                          lchild_data: np.ndarray,
                          rchild_indices: np.ndarray,
                          rchild_data: np.ndarray,
                          parent_indices: np.ndarray,
                          parent_data: np.ndarray):
        assert lchild_indices.size == lchild_data.size
        assert rchild_indices.size == rchild_data.size
        assert parent_indices.size == parent_data.size
        assert (lchild_data.size + rchild_data.size) == parent_data.size
        lchild_ptr = 0
        rchild_ptr = 0
        for i in range(parent_data.size):
            if (rchild_ptr == rchild_indices.size or
                    (lchild_ptr < lchild_indices.size and
                     lchild_indices[lchild_ptr] < rchild_indices[rchild_ptr])):
                assert parent_indices[i] == lchild_indices[lchild_ptr]
                parent_data[i] = lchild_data[lchild_ptr]
                lchild_ptr += 1
            else:
                assert parent_indices[i] == rchild_indices[rchild_ptr]
                assert (lchild_ptr == lchild_indices.size
                        or lchild_indices[lchild_ptr] != rchild_indices[rchild_ptr])
                parent_data[i] = rchild_data[rchild_ptr]
                rchild_ptr += 1

    def cut_trellis(self, t: Trellis):
        num_ecc = len(self.ecc_constraints)
        num_points = self.n - num_ecc

        membership_indptr = t.leaves_indptr
        membership_indices = t.leaves_indices
        membership_data = self.get_membership_data(membership_indptr,
                                                   membership_indices)
        obj_vals = np.zeros((t.num_nodes,))
        num_ecc_sat = np.zeros((t.num_nodes,))

        for node in t.internal_nodes_topo_ordered():
            node_start = membership_indptr[node]
            node_end = membership_indptr[node+1]
            leaves = membership_indices[node_start:node_end]
            if num_ecc > 0:
                num_ecc_sat[node] = self.get_num_ecc_sat(leaves, num_points)
            obj_vals[node] = self.get_intra_cluster_energy(leaves)
            for lchild, rchild in t.get_child_pairs_iter(node):
                cpair_num_ecc_sat = num_ecc_sat[lchild] + num_ecc_sat[rchild]
                cpair_obj_val = obj_vals[lchild] + obj_vals[rchild]
                if (num_ecc_sat[node] < cpair_num_ecc_sat 
                    or (num_ecc_sat[node] == cpair_num_ecc_sat
                        and obj_vals[node] <= cpair_obj_val)):
                    num_ecc_sat[node] = cpair_num_ecc_sat
                    obj_vals[node] = cpair_obj_val
                    lchild_start = membership_indptr[lchild]
                    lchild_end = membership_indptr[lchild+1]
                    rchild_start = membership_indptr[rchild]
                    rchild_end = membership_indptr[rchild+1]
                    self.merge_memberships(
                            membership_indices[lchild_start:lchild_end],
                            membership_data[lchild_start:lchild_end],
                            membership_indices[rchild_start:rchild_end],
                            membership_data[rchild_start:rchild_end],
                            membership_indices[node_start:node_end],
                            membership_data[node_start:node_end],
                    )

        # The value of `node` is the root since we iterate over the trellis
        # nodes in topological order bottom up. Moreover, the values of
        # `node_start` and `node_end` also correspond to the root of the
        # trellis.
        best_clustering = membership_data[node_start:node_end]
        if num_ecc > 0:
            best_clustering = best_clustering[:-num_ecc]

        return best_clustering, obj_vals[node], num_ecc_sat[node]

    def pred(self):
        num_ecc = len(self.ecc_constraints)

        # Construct and solve SDP
        start_solve_time = time.time()
        sdp_obj_value, pw_probs = self.build_and_solve_sdp()
        end_solve_time = time.time()

        # Build trellis
        t = self.build_trellis(pw_probs)

        # Cut trellis
        pred_clustering, cut_obj_value, num_ecc_satisfied = self.cut_trellis(t)

        self.prev_pred_clusters = pred_clustering

        metrics = {
                'sdp_solve_time': end_solve_time - start_solve_time,
                'sdp_obj_value': sdp_obj_value,
                'cut_obj_value': cut_obj_value,
                'num_ecc_satisfied': int(num_ecc_satisfied),
                'num_ecc': num_ecc,
                'frac_ecc_satisfied': num_ecc_satisfied / num_ecc
                        if num_ecc > 0 else 0.0,
                'num_ecc_feats': self.ecc_mx.nnz 
                        if self.ecc_mx is not None else 0,
                'num_pos_ecc_feats': (self.ecc_mx > 0).nnz
                        if self.ecc_mx is not None else 0,
                'num_neg_ecc_feats': (self.ecc_mx < 0).nnz
                        if self.ecc_mx is not None else 0,
        }

        return pred_clustering, metrics


def get_cluster_feats(point_feats: csr_matrix):
    csc_indptr = point_feats.tocsc().indptr
    return csr_matrix((np.diff(csc_indptr) > 0).astype(np.int64))


@nb.njit(parallel=True)
def set_matching_matrix(gold_indptr: np.ndarray,
                        gold_indices: np.ndarray,
                        pred_indptr: np.ndarray,
                        pred_indices: np.ndarray,
                        matching_mx: np.ndarray) -> None:
    for i in nb.prange(gold_indptr.size - 1):
        for j in nb.prange(pred_indptr.size - 1):
            gold_ptr = gold_indptr[i]
            next_gold_ptr = gold_indptr[i+1]
            pred_ptr = pred_indptr[j]
            next_pred_ptr = pred_indptr[j+1]
            num_intersect = 0
            num_union = 0
            while gold_ptr < next_gold_ptr and pred_ptr < next_pred_ptr:
                if gold_indices[gold_ptr] < pred_indices[pred_ptr]:
                    gold_ptr += 1
                    num_union += 1
                elif gold_indices[gold_ptr] > pred_indices[pred_ptr]:
                    pred_ptr += 1
                    num_union += 1
                else:
                    gold_ptr += 1
                    pred_ptr += 1
                    num_intersect += 1
                    num_union += 1
            if gold_ptr < next_gold_ptr:
                num_union += (next_gold_ptr - gold_ptr)
            elif pred_ptr < next_pred_ptr:
                num_union += (next_pred_ptr - pred_ptr)
            matching_mx[i, j] = num_intersect / num_union
    return matching_mx


@nb.njit
def argmaximin(row_max: np.ndarray,
               col_max: np.ndarray,
               row_argmax: np.ndarray,
               col_argmax: np.ndarray,
               row_indices: np.ndarray,
               col_indices: np.ndarray):
    # This function picks the pair of gold and pred clusters which has 
    # the highest potential gain in cluster F1.
    best_maximin = 1.0
    best_argmaximin = (0, 0)
    for i in range(row_indices.size):
        curr_maximin = max(row_max[row_indices[i]], col_max[col_indices[i]])
        if row_max[row_indices[i]] < col_max[col_indices[i]]:
            curr_argmaximin = (col_argmax[col_indices[i]], col_indices[i])
        else:
            curr_argmaximin = (row_indices[i], row_argmax[row_indices[i]])
        if curr_maximin < best_maximin:
            best_maximin = curr_maximin
            best_argmaximin = curr_argmaximin
    return best_argmaximin


@nb.njit
def nb_isin_sorted(values: np.ndarray, query: int):
    dom_min = 0             # inclusive
    dom_max = values.size   # exclusive
    while dom_max - dom_min > 0:
        i = ((dom_max - dom_min) // 2) + dom_min
        if values[i] > query:
            dom_max = i
        elif values[i] < query:
            dom_min = i + 1
        else:
            return True
    return False


@nb.njit
def get_salient_feats(point_feats_indptr: np.ndarray,
                      point_feats_indices: np.ndarray,
                      point_idxs: np.ndarray,
                      salient_feat_counts: np.ndarray):
    for i in range(point_feats_indptr.size-1):
        in_focus_set = nb_isin_sorted(point_idxs, i)
        for j in range(point_feats_indptr[i], point_feats_indptr[i+1]):
            if salient_feat_counts[point_feats_indices[j]] == -1:
                continue
            if not in_focus_set:
                salient_feat_counts[point_feats_indices[j]] = -1
            else:
                salient_feat_counts[point_feats_indices[j]] += 1


@nb.njit
def get_point_matching_mx(gold_cluster_lbls: np.ndarray,
                          pred_cluster_lbls: np.ndarray):
    num_gold_clusters = np.unique(gold_cluster_lbls).size
    num_pred_clusters = np.unique(pred_cluster_lbls).size
    intersect_mx = np.zeros((num_gold_clusters, num_pred_clusters))
    union_mx = np.zeros((num_gold_clusters, num_pred_clusters))
    for i in range(gold_cluster_lbls.size):
        gold_idx = gold_cluster_lbls[i]
        pred_idx = pred_cluster_lbls[i]
        intersect_mx[gold_idx, pred_idx] += 1
        union_mx[gold_idx, :] += 1
        union_mx[:, pred_idx] += 1
        union_mx[gold_idx, pred_idx] -= 1
    return intersect_mx / union_mx


def gen_forced_ecc_constraint(point_feats: csr_matrix,
                              gold_cluster_lbls: np.ndarray,
                              pred_cluster_lbls: np.ndarray,
                              gold_cluster_feats: csr_matrix,
                              pred_cluster_feats: csr_matrix,
                              matching_mx: np.ndarray,
                              max_overlap_feats: int):
    # construct the point matching matrix
    point_matching_mx = get_point_matching_mx(
            gold_cluster_lbls, pred_cluster_lbls)

    # set perfect match rows and columns to zero so they will not be picked
    perfect_match = (point_matching_mx == 1.0)
    row_mask = np.any(perfect_match, axis=1)
    column_mask = np.any(perfect_match, axis=0)
    to_zero_mask = row_mask[:, None] | column_mask[None, :]
    point_matching_mx[to_zero_mask] = 0.0

    # greedily pick minimax match
    row_max = np.max(point_matching_mx, axis=1)
    col_max = np.max(point_matching_mx, axis=0)
    row_argmax = np.argmax(point_matching_mx, axis=1)
    col_argmax = np.argmax(point_matching_mx, axis=0)
    row_indices, col_indices = np.where(point_matching_mx > 0.0)
    gold_cluster_idx, pred_cluster_idx = argmaximin(
            row_max, col_max, row_argmax, col_argmax, row_indices, col_indices)

    logging.info(f'Gold Cluster: {gold_cluster_idx},'
                 f' Pred Cluster: {pred_cluster_idx}')
    
    # get points in gold and pred clusters, resp.
    gold_cluster_points = set(np.where(gold_cluster_lbls==gold_cluster_idx)[0])
    pred_cluster_points = set(np.where(pred_cluster_lbls==pred_cluster_idx)[0])

    gold_and_pred = np.asarray(list(gold_cluster_points & pred_cluster_points))
    gold_not_pred = np.asarray(list(gold_cluster_points - pred_cluster_points))
    pred_not_gold = np.asarray(list(pred_cluster_points - gold_cluster_points))

    # start the sampling process with overlap feats
    gold_and_pred_sfc = np.zeros((point_feats.shape[1],))
    get_salient_feats(
            point_feats.indptr,
            point_feats.indices,
            np.sort(gold_and_pred),
            gold_and_pred_sfc
    )
    sampled_overlap_feats = np.where(gold_and_pred_sfc == 1.0)[0]
    np.random.shuffle(sampled_overlap_feats)
    sampled_overlap_feats = sampled_overlap_feats[:max_overlap_feats]
    # NOTE: why doesn't this line below work well with the SDP?
    # i.e. why don't the most common features work best
    #sampled_overlap_feats = np.argsort(gold_and_pred_sfc)[-max_overlap_feats:]

    # now onto postive feats
    sampled_pos_feats = []
    gold_not_pred_lbls = np.asarray(
            [pred_cluster_lbls[i] for i in gold_not_pred])
    for pred_lbl in np.unique(np.asarray(gold_not_pred_lbls)):
        pred_cluster_mask = (gold_not_pred_lbls == pred_lbl)
        gold_not_pred_sfc = np.zeros((point_feats.shape[1],))
        get_salient_feats(
                point_feats.indptr,
                point_feats.indices,
                np.sort(gold_not_pred[pred_cluster_mask]),
                gold_not_pred_sfc
        )
        _sampled_pos_feats = np.where(gold_not_pred_sfc == np.max(gold_not_pred_sfc))[0]
        np.random.shuffle(_sampled_pos_feats)
        sampled_pos_feats.append(_sampled_pos_feats[0])
    sampled_pos_feats = np.asarray(sampled_pos_feats)

    # lastly, negative feats
    sampled_neg_feats = []
    pred_not_gold_lbls = np.asarray(
            [gold_cluster_lbls[i] for i in pred_not_gold])
    for gold_lbl in np.unique(np.asarray(pred_not_gold_lbls)):
        pred_cluster_mask = (pred_not_gold_lbls == gold_lbl)
        pred_not_gold_sfc = np.zeros((point_feats.shape[1],))
        get_salient_feats(
                point_feats.indptr,
                point_feats.indices,
                np.sort(pred_not_gold[pred_cluster_mask]),
                pred_not_gold_sfc
        )
        _sampled_neg_feats = np.where(pred_not_gold_sfc == np.max(pred_not_gold_sfc))[0]
        np.random.shuffle(_sampled_neg_feats)
        sampled_neg_feats.append(_sampled_neg_feats[0])
    sampled_neg_feats = np.asarray(sampled_neg_feats)

    # create the ecc constraint
    new_ecc_col = np.hstack(
            (sampled_overlap_feats,
             sampled_pos_feats,
             sampled_neg_feats)
    )
    new_ecc_row = np.zeros_like(new_ecc_col)
    new_ecc_data = np.hstack(
            (np.ones_like(sampled_overlap_feats),
             np.ones_like(sampled_pos_feats),
             -1*np.ones_like(sampled_neg_feats))
    )

    new_ecc = coo_matrix(
            (new_ecc_data, (new_ecc_row, new_ecc_col)),
            shape=(1,point_feats.shape[1]),
            dtype=np.int64
    ).tocsr()

    # for debugging
    constraint_str = ', '.join(
            [('+f' if d > 0 else '-f') + str(int(c))
                for c, d in zip(new_ecc_col, new_ecc_data)]
    )
    logging.info(f'Constraint generated: [{constraint_str}]')

    logging.info('Nodes with features: {')
    for feat_id in new_ecc_col:
        nodes_with_feat = point_feats.T[int(feat_id)].tocoo().col
        nodes_with_feat = [f'n{i}' for i in nodes_with_feat]
        logging.info(f'\tf{int(feat_id)}: {", ".join(nodes_with_feat)}')
    logging.info('}')

    # generate "equivalent" pairwise point constraints
    overlap_feats = set(sampled_overlap_feats)
    pos_feats = set(sampled_pos_feats)
    neg_feats = set(sampled_neg_feats)

    gold_not_pred = gold_cluster_points - pred_cluster_points
    pred_not_gold = pred_cluster_points - gold_cluster_points

    num_points = point_feats.shape[0]
    pairwise_constraints = dok_matrix((num_points, num_points))
    for s, t in product(pred_cluster_points, gold_not_pred):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(pos_feats)):
            if gold_cluster_lbls[s] == gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = 1
                else:
                    pairwise_constraints[t, s] = 1

    for s, t in product(gold_cluster_points, pred_not_gold):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(neg_feats)):
            if gold_cluster_lbls[s] != gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = -1
                else:
                    pairwise_constraints[t, s] = -1

    return new_ecc, pairwise_constraints.tocoo()


def gen_ecc_constraint(point_feats: csr_matrix,
                       gold_cluster_lbls: np.ndarray,
                       pred_cluster_lbls: np.ndarray,
                       gold_cluster_feats: csr_matrix,
                       pred_cluster_feats: csr_matrix,
                       matching_mx: np.ndarray,
                       max_overlap_feats: int,
                       max_pos_feats: int,
                       max_neg_feats: int,
                       overlap_col_wt: np.ndarray,
                       pos_col_wt: np.ndarray,
                       neg_col_wt: np.ndarray):

    # set perfect match rows and columns to zero so they will not be picked
    perfect_match = (matching_mx == 1.0)
    row_mask = np.any(perfect_match, axis=1)
    column_mask = np.any(perfect_match, axis=0)
    to_zero_mask = row_mask[:, None] | column_mask[None, :]
    matching_mx[to_zero_mask] = 0.0

    norm_factor = np.sum(matching_mx)
    if norm_factor == 0.0:
        logging.info('Features of gold clusters already fully satisfied.'
                     ' Cannot generate constraints to affect clustering.')
        logging.info('Exiting...')
        exit()

    # pick a gold cluster, pred cluster pair
    norm_matching_mx = matching_mx / norm_factor
    pair_ravel_idx = np.where(
            np.random.multinomial(1, norm_matching_mx.ravel()))
    gold_cluster_idx, pred_cluster_idx = np.unravel_index(
            pair_ravel_idx[0][0], matching_mx.shape)

    # select features to for the ecc constraint
    src_feats = pred_cluster_feats[pred_cluster_idx]
    tgt_feats = gold_cluster_feats[gold_cluster_idx]
    all_overlap_feats = ((src_feats + tgt_feats) == 2).astype(np.int64)
    all_pos_feats = ((tgt_feats - src_feats) == 1).astype(np.int64)
    all_neg_feats = ((src_feats - tgt_feats) == 1).astype(np.int64)

    def sample_csr_cols(mx, num_sample, col_wt=None):
        choices = mx.tocoo().col
        k = min(num_sample, choices.size)
        if k > num_sample:
            p = col_wt[choices]
            p /= np.sum(p)
        else:
            p = None
        samples = np.random.choice(choices, (k,), replace=False, p=p)
        return samples

    sampled_overlap_feats = sample_csr_cols(
            all_overlap_feats, max_overlap_feats, col_wt=overlap_col_wt)
    sampled_pos_feats = sample_csr_cols(
            all_pos_feats, max_pos_feats, col_wt=pos_col_wt)
    sampled_neg_feats = sample_csr_cols(
            all_neg_feats, max_neg_feats, col_wt=neg_col_wt)

    new_ecc_col = np.hstack(
            (sampled_overlap_feats,
             sampled_pos_feats,
             sampled_neg_feats)
    )
    new_ecc_row = np.zeros_like(new_ecc_col)
    new_ecc_data = np.hstack(
            (np.ones_like(sampled_overlap_feats),
             np.ones_like(sampled_pos_feats),
             -1*np.ones_like(sampled_neg_feats))
    )

    new_ecc = coo_matrix((new_ecc_data, (new_ecc_row, new_ecc_col)),
                         shape=src_feats.shape, dtype=np.int64).tocsr()

    # for debugging
    constraint_str = ', '.join(
            [('+f' if d > 0 else '-f') + str(int(c))
                for c, d in zip(new_ecc_col, new_ecc_data)]
    )
    logging.info(f'Constraint generated: [{constraint_str}]')

    logging.info('Nodes with features: {')
    for feat_id in new_ecc_col:
        nodes_with_feat = point_feats.T[int(feat_id)].tocoo().col
        nodes_with_feat = [f'n{i}' for i in nodes_with_feat]
        logging.info(f'\tf{int(feat_id)}: {", ".join(nodes_with_feat)}')
    logging.info('}')

    # generate "equivalent" pairwise point constraints
    overlap_feats = set(sampled_overlap_feats)
    pos_feats = set(sampled_pos_feats)
    neg_feats = set(sampled_neg_feats)

    gold_cluster_points = set(np.where(gold_cluster_lbls==gold_cluster_idx)[0])
    pred_cluster_points = set(np.where(pred_cluster_lbls==pred_cluster_idx)[0])

    gold_not_pred = gold_cluster_points - pred_cluster_points
    pred_not_gold = pred_cluster_points - gold_cluster_points

    num_points = point_feats.shape[0]
    pairwise_constraints = dok_matrix((num_points, num_points))
    for s, t in product(pred_cluster_points, gold_not_pred):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(pos_feats)):
            if gold_cluster_lbls[s] == gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = 1
                else:
                    pairwise_constraints[t, s] = 1

    for s, t in product(gold_cluster_points, pred_not_gold):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(neg_feats)):
            if gold_cluster_lbls[s] != gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = -1
                else:
                    pairwise_constraints[t, s] = -1

    return new_ecc, pairwise_constraints.tocoo()


def simulate(edge_weights: csr_matrix,
             point_features: csr_matrix,
             gold_clustering: np.ndarray,
             hparams: argparse.Namespace):

    gold_cluster_feats = sp_vstack([
        get_cluster_feats(point_features[gold_clustering == i])
            for i in np.unique(gold_clustering)
    ])

    clusterer = EccClusterer(edge_weights=edge_weights,
                             features=point_features,
                             hparams=hparams)

    pairwise_constraints_for_replay = []
    round_pred_clusterings = []

    for r in range(hparams.max_rounds):
        # compute predicted clustering
        pred_clustering, metrics = clusterer.pred()

        # get predicted cluster feats and do some label remapping for later
        uniq_pred_cluster_lbls = np.unique(pred_clustering)
        pred_cluster_feats = sp_vstack([
            get_cluster_feats(point_features[pred_clustering == i])
                for i in uniq_pred_cluster_lbls
        ])
        remap_lbl_dict = {j: i for i, j in enumerate(uniq_pred_cluster_lbls)}
        for i in range(pred_clustering.size):
            pred_clustering[i] = remap_lbl_dict[pred_clustering[i]]

        # for debugging
        logging.info('Gold Clustering: {')
        for cluster_id in np.unique(gold_clustering):
            nodes_in_cluster = list(np.where(gold_clustering == cluster_id)[0])
            nodes_in_cluster = [f'n{i}' for i in nodes_in_cluster]
            logging.info(f'\tc{cluster_id}: {", ".join(nodes_in_cluster)}')
        logging.info('}')

        logging.info('Predicted Clustering: {')
        for cluster_id in np.unique(pred_clustering):
            nodes_in_cluster = list(np.where(pred_clustering == cluster_id)[0])
            nodes_in_cluster = [f'n{i}' for i in nodes_in_cluster]
            logging.info(f'\tc{cluster_id}: {", ".join(nodes_in_cluster)}')
        logging.info('}')

        # record `pred_clustering` for later analysis
        round_pred_clusterings.append(copy.deepcopy(pred_clustering))

        # construct jaccard similarity matching matrix
        matching_mx = np.empty((gold_cluster_feats.shape[0],
                                pred_cluster_feats.shape[0]))
        set_matching_matrix(
                gold_cluster_feats.indptr, gold_cluster_feats.indices,
                pred_cluster_feats.indptr, pred_cluster_feats.indices,
                matching_mx
        )

        # handle some metric stuffs
        metrics['match_feat_coeff'] = np.mean(np.max(matching_mx, axis=1))
        metrics['rand_idx'] = rand_idx(gold_clustering, pred_clustering)
        metrics['f1'] = cluster_f1(gold_clustering, pred_clustering)[2]
        metric_str = '; '.join([
            k + ' = ' 
            + ('{:.4f}'.format(v) if isinstance(v, float) else str(v))
                for k, v in metrics.items()
        ])
        logging.info('Round %d metrics - ' + metric_str, r)

        # exit if we predict the ground truth clustering
        if metrics['rand_idx'] == 1.0:
            assert metrics['match_feat_coeff'] == 1.0
            logging.info('Achieved perfect clustering at round %d.', r)
            break

        # generate a new constraint
        while True:
            ecc_constraint, pairwise_constraints = gen_forced_ecc_constraint(
                    point_features,
                    gold_clustering,
                    pred_clustering,
                    gold_cluster_feats,
                    pred_cluster_feats,
                    matching_mx,
                    hparams.max_overlap_feats
            )

            #feature_counts = point_features.sum(axis=0)
            #overlap_col_wt = 1.0 / feature_counts
            #pos_col_wt = 1.0 / feature_counts
            #neg_col_wt = feature_counts
            ##ecc_constraint, pairwise_constraints = gen_ecc_constraint(
            ##        point_feats=point_features,
            ##        gold_clustering=gold_clustering,
            ##        pred_clustering=pred_clustering,
            ##        gold_cluster_feats=gold_cluster_feats,
            ##        pred_cluster_feats=pred_cluster_feats,
            ##        matching_mx=matching_mx,
            ##        max_over_lap_feats=hparams.max_overlap_feats,
            ##        max_pos_feats=2,
            ##        max_neg_feats=2,
            ##        overlap_col_wt=overlap_col_wt,
            ##        pos_col_wt=pos_col_wt,
            ##        neg_col_wt=neg_col_wt
            ##)

            #ecc_constraint, pairwise_constraints = gen_ecc_constraint(
            #        point_features,
            #        gold_clustering,
            #        pred_clustering,
            #        gold_cluster_feats,
            #        pred_cluster_feats,
            #        matching_mx,
            #        hparams.max_overlap_feats,
            #        3,  # max_pos_feats
            #        1,  # max_neg_feats
            #        overlap_col_wt,
            #        pos_col_wt,
            #        neg_col_wt
            #)
            already_exists = any([
                (ecc_constraint != x).nnz == 0
                    for x in clusterer.ecc_constraints
            ])
            if already_exists:
                logging.error('Produced duplicate ecc constraint')
                continue

            already_satisfied = (
                (pred_cluster_feats @ ecc_constraint.T) == ecc_constraint.nnz
            ).todense().any()
            if already_satisfied:
                logging.warning('Produced already satisfied ecc constraint')
                continue

            pairwise_constraints_for_replay.append(pairwise_constraints)
            break

        logging.info('Adding new constraint')
        clusterer.add_constraint(ecc_constraint)

    return (clusterer.ecc_constraints,
            pairwise_constraints_for_replay,
            round_pred_clusterings)


def get_hparams() -> argparse.Namespace:
    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--debug', action='store_true',
                        help="Enables and disables certain opts for debugging")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory for this run.")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="Path to preprocessed data.")

    # for specbm
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
    parser.add_argument("--sketch_dim", type=int, default=0,
                        help="dimension of Nystrom sketch")

    # for constraint generation
    parser.add_argument('--max_rounds', type=int, default=100,
                        help="number of rounds to generate feedback for")
    parser.add_argument('--max_overlap_feats', type=int, default=2,
                        help="max num overlap features to sample.")
    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    # TODO: make this a hparam
    np.random.seed(0)

    hparams = get_hparams()

    if not hparams.debug:
        # create output directory
        assert not os.path.exists(hparams.output_dir)
        os.makedirs(hparams.output_dir)

        # dump hparams
        pickle.dump(
                hparams, 
                open(os.path.join(hparams.output_dir, 'hparams.pkl'), "wb")
        )
        logging.basicConfig(
                filename=os.path.join(hparams.output_dir, 'out.log'),
                format='(ECC) :: %(asctime)s >> %(message)s',
                datefmt='%m-%d-%y %H:%M:%S',
                level=logging.INFO
        )
    else:
        logging.basicConfig(
                format='(ECC) :: %(asctime)s >> %(message)s',
                datefmt='%m-%d-%y %H:%M:%S',
                level=logging.INFO
        )

    logging.info("cmd: {}".format(" ".join(["python"] + sys.argv)))
    logging.info("git sha: {}".format(git.Repo(search_parent_directories=True).head.object.hexsha))
    logging.info('Experiment args:\n{}'.format(
        json.dumps(vars(hparams), sort_keys=True, indent=4)))

    with open(hparams.data_path, 'rb') as f:
        logging.info('loading preprocessed data.')
        blocks_preprocessed = pickle.load(f)

    ecc_for_replay = {}
    mlcl_for_replay = {}
    pred_clusterings = {}
    num_blocks = len(blocks_preprocessed)

    # problematic canopies
    # x "d schmidt"
    # x "h ishikawa"
    # x "k chen"
    # - "p wu"
    # - "s mueller"

    sub_blocks_preprocessed = {}
    #sub_blocks_preprocessed['d schmidt'] = blocks_preprocessed['d schmidt']
    #sub_blocks_preprocessed['h ishikawa'] = blocks_preprocessed['h ishikawa']
    #sub_blocks_preprocessed['k chen'] = blocks_preprocessed['k chen']
    #sub_blocks_preprocessed['p wu'] = blocks_preprocessed['p wu']

    #sub_blocks_preprocessed['v kolmanovskii'] = blocks_preprocessed['v kolmanovskii']
    #sub_blocks_preprocessed['r weiss'] = blocks_preprocessed['r weiss']
    #sub_blocks_preprocessed['m schuetzenberger'] = blocks_preprocessed['m schuetzenberger']
    #sub_blocks_preprocessed['m nagata'] = blocks_preprocessed['m nagata']
    #sub_blocks_preprocessed['m nagata'] = blocks_preprocessed['m nagata']
    #sub_blocks_preprocessed['l wang'] = blocks_preprocessed['l wang']
    #sub_blocks_preprocessed['a ostrowski'] = blocks_preprocessed['a ostrowski']
    sub_blocks_preprocessed = blocks_preprocessed

    for i, (block_name, block_data) in enumerate(sub_blocks_preprocessed.items()):
        edge_weights = block_data['edge_weights']
        point_features = block_data['point_features'].tocsr()
        gold_clustering = block_data['labels']

        ## skip small blocks
        #if edge_weights.shape[0] < 20:
        #    continue

        assert edge_weights.shape[0] == point_features.shape[0]
        num_clusters = np.unique(gold_clustering).size

        logging.info(f'loaded block \"{block_name}\" ({i+1}/{num_blocks})')
        logging.info(f'\t number of points: {edge_weights.shape[0]}')
        logging.info(f'\t number of clusters: {num_clusters}')
        logging.info(f'\t number of features: {point_features.shape[1]}')

        (block_ecc_for_replay,
         block_mlcl_for_replay,
         round_pred_clusterings) = simulate(
                edge_weights,
                point_features,
                gold_clustering,
                hparams
        )

        ecc_for_replay[block_name] = block_ecc_for_replay
        mlcl_for_replay[block_name] = block_mlcl_for_replay
        pred_clusterings[block_name] = round_pred_clusterings

    if not hparams.debug:
        logging.info('dumping ecc and mlcl constraints for replay')
        ecc_fname = os.path.join(hparams.output_dir, 'ecc_for_replay.pkl')
        mlcl_fname = os.path.join(hparams.output_dir, 'mlcl_for_replay.pkl')
        pred_fname = os.path.join(hparams.output_dir, 'pred_clusterings.pkl')
        pickle.dump(ecc_for_replay, open(ecc_fname, 'wb'))
        pickle.dump(mlcl_for_replay, open(mlcl_fname, 'wb'))
        pickle.dump(pred_clusterings, open(pred_fname, 'wb'))