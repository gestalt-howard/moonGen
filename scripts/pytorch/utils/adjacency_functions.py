# Authors: Aaron Wu / Howard Tai

# Script contains code for defining adjacency-related functions

import numpy as np

from pytorch.utils.full_process_utils import *


def set_diagonal(adj_mat):
    """
    Sets self adjacency (i.e. 1 along diagonal of adjacency matrix)
    """
    for i in range(adj_mat.shape[0]):
        adj_mat[i, i] = 1
    return adj_mat


def norm_adjacency(adj_mat):
    """
    Normalizes the adjacency matrix by implementing D^(-1/2) x A x D^(-1/2)
    """
    # Calculate degrees
    rowsum = np.array(adj_mat.sum(1))

    # D^(-1/2), checking for infinity
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.

    # Placeholder for degree matrix, diagonals are degree values ^ (-0.5)
    r_mat_inv = np.zeros(adj_mat.shape)
    for i in range(len(r_inv)):
        r_mat_inv[i, i] = r_inv[i]

    # D^(-1/2) x A
    norm_adj_mat = np.dot(r_mat_inv, adj_mat)

    # D^(-1/2) x A x D^(-1/2)
    norm_adj_mat = np.dot(norm_adj_mat, r_mat_inv)
    return norm_adj_mat


def gen_adjacency(full_processed, nodes_keys):
    """
    Input: 
        1. full_processed (from full_data_process.py)
            A GraphDataProcess object that neatly contains:
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix

    Output: adj_mat
        An adjacency matrix (to be used for train/test)

    Description:
        1. Takes a set of nodes that represent both problems and holds (nodes_types_map determines node type)
        2. Gets the pmi matrix (hold-hold adjacency) and tfidf dictionary (problem-hold adjacency)
        3. Iterate through nodes:
            1. Determine hold types and retrieve corresponding values
            2. Self Adjacency: if node is compared against itself, 1
            3. Populate an adjacency matrix with values
               (hold_names is needed to determine where in the full pmi matrix to draw from since it is a numpy mat)

    Purpose:
        Generate the pmi/tfidf adjacency matrix to be used for train/test
    """
    # Get map of hold types
    nodes_types_map = get_nodes_types_map(full_processed)

    # Get hold names
    holds_names = get_holds_names(full_processed)

    # Get PMI and TFIDF calculations
    hold_adj = get_hold_adj(full_processed)
    prob_hold_adj = get_prob_hold_adj(full_processed)

    # Placeholder for adjacency matrix
    adj_mat = np.zeros((len(nodes_keys), len(nodes_keys)))

    # Populate adjacency matrix
    for i, node1 in enumerate(nodes_keys):
        node_type1 = nodes_types_map[node1]
        for j, node2 in enumerate(nodes_keys):
            node_type2 = nodes_types_map[node2]
            if j == i:  # Add in self-adjacency later
                continue

            # If problem-problem, skip
            if node_type1 == 'problem' and node_type2 == 'problem':
                continue

            # If problem-hold or hold-problem, populate with TFIDF
            elif node_type1 == 'problem' and node_type2 == 'hold':
                adj_mat[i, j] = prob_hold_adj[node1][node2]
                adj_mat[j, i] = adj_mat[i, j]

            # If hold-hold, populate with PMI
            elif node_type1 == 'hold' and node_type2 == 'hold':
                adj_mat[i, j] = hold_adj[holds_names.index(node1), holds_names.index(node2)]
                adj_mat[j, i] = hold_adj[holds_names.index(node2), holds_names.index(node1)]
    return adj_mat


def binary_adjacency(full_processed, nodes_keys):
    """
    Input:
        1. full_processed (from full_data_process.py)
        2. nodes_keys

    Output: adj_mat
        An adjacency matrix (to be used for train/test)

    Description:
        1. Takes a set of nodes that represent both problems and holds (nodes_types_map determines node type)
        2. Gets the problem-hold mapping from a graphDataProcess object
        3. Iterate through nodes:
            1. If the nodes are problem and a hold corresponding to that problem, then 1
            2. Else 0 (not even self-adjacency)

    Purpose:
        Generate the onehot adjacency matrix to be used for train/test
    """
    # Get hold type map
    nodes_types_map = get_nodes_types_map(full_processed)

    # Get problem-hold maps
    prob_holds_map = get_prob_holds_map(full_processed)

    # Initialize adjacency matrix
    adj_mat = np.zeros((len(nodes_keys), len(nodes_keys)))

    # Fill in adjacency matrix
    for i, node1 in enumerate(nodes_keys):
        node_type1 = nodes_types_map[node1]
        for j, node2 in enumerate(nodes_keys):
            node_type2 = nodes_types_map[node2]
            if j == i:  # Fill in self-adjacency later
                continue

            # Skip problem-problem relationship
            if node_type1 == 'problem' and node_type2 == 'problem':
                continue

            # Set problem-hold and hold-problem to 1
            elif node_type1 == 'problem' and node_type2 == 'hold':
                if node2 in prob_holds_map[node1]:
                    adj_mat[i, j] = 1
                    adj_mat[j, i] = 1

            # Skip hold-hold relationship
            elif node_type1 == 'hold' and node_type2 == 'hold':
                continue
    return adj_mat


def binary_adjacency_diag_norm(full_processed, nodes_keys):
    """
    Input:
        1. full_processed (from full_data_process.py)
        2. nodes_keys

    Output: normalized onehot adjacency with self-adjacency

    Description:
        Runs normalization on onehot adjacency matrix with self-adjacency

    Purpose:
        Needed to get binary adjacency matrix
    """
    return norm_adjacency(set_diagonal(binary_adjacency(full_processed, nodes_keys)))


def gen_adjacency_diag_norm(full_processed, nodes_keys):
    """
    Input:
        1. full_processed (from full_data_process.py)
        2. nodes_keys

    Output: normalized pmi/tfidf adjacency with self-adjacency

    Description:
        Runs normalization on pmi/tfidf adjacency matrix with self-adjacency

    Purpose:
        Needed to get pmi/tfidf adjacency matrix
    """
    return norm_adjacency(set_diagonal(gen_adjacency(full_processed, nodes_keys)))
