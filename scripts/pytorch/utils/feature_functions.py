# Authors: Aaron Wu / Howard Tai

# This script contains processing functions that define node features for input into a graph neural network

import numpy as np

from scripts.pytorch.utils.full_process_utils import *


def gen_onehotfeatures(full_processed, nodes_keys):
    """
    Input:
        1. full_processed (from full_data_process.py)
            An GraphDataProcess object that neatly contains:
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix

    Output: onehot_features
        Diagonal matrix of size nxn (n = number of nodes)

    Description:
        Generates a diagonal matrix as features

    Purpose:
        Simple features for the GCN
    """
    onehot_features = np.zeros((len(nodes_keys), len(nodes_keys)))

    # Fill in diagonal elements with 1
    for i in range(len(nodes_keys)):
        onehot_features[i][i] = 1
    return onehot_features


def gen_multihotfeatures(full_processed, nodes_keys):
    """
    Input:
        1. full_processed (from full_data_process.py)
            A GraphDataProcess object that neatly contains:
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix

    Output: multihot_features
        Matrix of features where the problem rows are represented by a multi-hot of their corresponding holds

    Description:
        1. Diagonal matrix of nodes
        2. For all problem node rows, also one-hot the hold columns

    Purpose:
        Multi-hot features for the GCN
    """
    # Get map of node types (problem or hold)
    nodes_types_map = get_nodes_types_map(full_processed)

    # Get hold names
    hold_names_map = get_holds_names(full_processed)

    # Get map of problems to holds
    prob_hold_map = get_prob_holds_map(full_processed)

    # Placeholder
    multihot_features = np.zeros((len(nodes_keys), len(hold_names_map)))

    # Iterate through nodes
    for i, n in enumerate(nodes_keys):
        node_type = nodes_types_map[n]

        # One-hot for holds
        if 'h' in node_type:
            multihot_features[i][nodes_keys.index(n)] = 1

        # Multi-hot for problems
        if 'p' in node_type:
            hold_nodes = prob_hold_map[n]
            for h in hold_nodes:
                multihot_features[i][nodes_keys.index(h)] = 1
    return multihot_features
