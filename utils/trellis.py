from typing import List, Tuple

import numba as nb
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


@nb.njit
def get_trellis_node_id(
        leaves_indptr: np.ndarray,
        leaves_indices: np.ndarray,
        new_leaves: np.ndarray
    ) -> Tuple[np.int64, np.ndarray, np.ndarray]:
    num_trellis_nodes = leaves_indptr.size - 1
    num_new_leaves = new_leaves.size
    node_id = -1
    for i in range(num_trellis_nodes):
        ptr = leaves_indptr[i]
        next_ptr = leaves_indptr[i+1]
        if (next_ptr - ptr) == num_new_leaves:
            match = True
            for idx, j in enumerate(range(ptr, next_ptr)):
                if leaves_indices[j] != new_leaves[idx]:
                    match = False
                    break
            if match:
                node_id = i
                break
    if node_id == -1:
        # node does not exist yet; create new trellis node in `leaves_*`
        node_id = num_trellis_nodes
        leaves_indptr = np.append(
                leaves_indptr, leaves_indptr[-1] + num_new_leaves)
        leaves_indices = np.append(leaves_indices, new_leaves)
    return node_id, leaves_indptr, leaves_indices


@nb.njit
def build_trellis_from_trees(trees: np.ndarray):
    num_trees = trees.shape[0]
    num_tree_nodes = trees.shape[1]
    num_leaves = (num_tree_nodes + 1) // 2
    leaves_indptr = np.arange(num_leaves+1, dtype=np.int64)
    leaves_indices = np.arange(num_leaves, dtype=np.int64)
    child_pairs_indptr = np.zeros((num_leaves+1,), dtype=np.int64)
    child_pairs_indices = np.empty((0,), dtype=np.int64)
    curr_leaves_mask = np.empty((num_leaves,), dtype=np.bool_)
    lchild_leaves_mask = np.empty((num_leaves,), dtype=np.bool_)
    rchild_leaves_mask = np.empty((num_leaves,), dtype=np.bool_)

    for b in range(num_trees):
        membership = np.arange(num_leaves)
        for curr in range(num_leaves, num_tree_nodes):
            child_mask = (trees[b] == curr)
            children = np.where(child_mask)[0]
            for i in range(num_leaves):
                if membership[i] == children[0]:
                    lchild_leaves_mask[i] = True
                    rchild_leaves_mask[i] = False
                    curr_leaves_mask[i] = True
                elif membership[i] == children[1]:
                    lchild_leaves_mask[i] = False
                    rchild_leaves_mask[i] = True
                    curr_leaves_mask[i] = True
                else:
                    lchild_leaves_mask[i] = False
                    rchild_leaves_mask[i] = False
                    curr_leaves_mask[i] = False

            membership[curr_leaves_mask] = curr

            curr_leaves = np.where(curr_leaves_mask)[0]
            lchild_leaves = np.where(lchild_leaves_mask)[0]
            rchild_leaves = np.where(rchild_leaves_mask)[0]

            prev_num_trellis_nodes = leaves_indptr.size - 1
            (lchild_node_id,
             leaves_indptr,
             leaves_indices) = get_trellis_node_id(
                    leaves_indptr, leaves_indices, lchild_leaves)
            (rchild_node_id,
             leaves_indptr,
             leaves_indices) = get_trellis_node_id(
                    leaves_indptr, leaves_indices, rchild_leaves)
            assert lchild_node_id < prev_num_trellis_nodes
            assert rchild_node_id < prev_num_trellis_nodes

            (curr_node_id,
             leaves_indptr,
             leaves_indices) = get_trellis_node_id(
                    leaves_indptr, leaves_indices, curr_leaves)
            assert curr_node_id <= prev_num_trellis_nodes

            if curr_node_id == prev_num_trellis_nodes:
                child_pairs_indptr = np.append(
                        child_pairs_indptr, child_pairs_indptr[-1]+2)
                child_pairs_indices = np.append(
                        child_pairs_indices,
                        [lchild_node_id, rchild_node_id])
            else:
                ptr = child_pairs_indptr[curr_node_id]
                next_ptr = child_pairs_indptr[curr_node_id+1]
                already_exists = False
                for i in range(ptr, next_ptr, 2):
                    if child_pairs_indices[i] == lchild_node_id:
                        assert child_pairs_indices[i+1] == rchild_node_id
                        already_exists = True
                        break
                if not already_exists:
                    child_pairs_indptr[curr_node_id+1:] += 2
                    child_pairs_indices = np.concatenate((
                            child_pairs_indices[:next_ptr],
                            np.array(sorted([lchild_node_id, rchild_node_id]),
                                     dtype=np.int64),
                            child_pairs_indices[next_ptr:]
                    ))

    return (leaves_indptr,
            leaves_indices,
            child_pairs_indptr,
            child_pairs_indices)


@nb.njit
def convert_Z_to_parents(Z: np.ndarray) -> np.ndarray:
    num_internal_nodes = Z.shape[0]
    parents = np.empty((2 * num_internal_nodes + 1,), dtype=np.int64)
    for i in range(num_internal_nodes):
        parents[int(Z[i, 0])] = i + num_internal_nodes + 1
        parents[int(Z[i, 1])] = i + num_internal_nodes + 1
    parents[-1] = parents.shape[0] - 1
    return parents


class Trellis(object):

    def __init__(self, adj_mx: np.ndarray):
        self.adj_mx = adj_mx
        self.n = adj_mx.shape[0]
        self.topo_order = None

    def fit(self):
        dist_mx = np.triu(1 - self.adj_mx, k=1)
        dist_mx += dist_mx.T
        flat_dist_mx = squareform(dist_mx)
        link_fns = ["average", "single", "complete", "ward", "median", "centroid", "weighted"]
        trees = np.vstack([convert_Z_to_parents(linkage(flat_dist_mx, link_fn))
                           for link_fn in link_fns])

        # build trellis
        (self.leaves_indptr,
         self.leaves_indices,
         self.child_pairs_indptr,
         self.child_pairs_indices) = build_trellis_from_trees(trees)

        # set topological order for internal trellis node iteration
        self.topo_order = np.argsort(np.diff(self.leaves_indptr))[self.n:]

        # set some basic variables
        self.num_nodes = self.leaves_indptr.size - 1

    def internal_nodes_topo_ordered(self):
        if self.topo_order is None:
            raise ValueError('Topological order has not been set.'
                             ' Please run `fit()`.')
        return iter(self.topo_order.tolist())

    def get_child_pairs_iter(self, node_idx: int):
        for i in range(self.child_pairs_indptr[node_idx],
                self.child_pairs_indptr[node_idx+1], 2):
            yield (self.child_pairs_indices[i],
                   self.child_pairs_indices[i+1])
