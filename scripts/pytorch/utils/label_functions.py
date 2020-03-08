# Authors: Aaron Wu / Howard Tai

# Script for defining labels in various formats (i.e. flat list, one-hot, etc...)

import numpy as np

from scripts.pytorch.utils.full_process_utils import *


def gen_labels_idxs(full_processed, nodes_keys):
    """
    Input:
        1. full_processed (from full_data_process.py)
            An graphDataProcess object that neatly contains:
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into TFIDF mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix

    Output: labels
        Array of labels (0,1,2,3...) that correspond to the given node_keys

    Description:
        Loads a problem-grade map and uses it to retrieve grades for each problem in a list

    Purpose:
        Generate labels for train/test
    """
    # Get difficulty grades
    grades_dict = get_grades_dict(full_processed)

    # Extract grades for subset of nodes
    labels = np.zeros(len(nodes_keys))
    for i, key in enumerate(nodes_keys):
        if key in grades_dict:
            labels[i] = grades_dict[key]

    # Get set of difficulties
    labels_set = sorted(list(set(labels)))

    # Instantiate dictionary to define re-mapping difficulty to start from 0
    labels_dict = dict()
    for i, label in enumerate(labels_set):
        labels_dict[label] = i

    # Use labels_dict to re-map difficulty
    for i in range(labels.shape[0]):
        labels[i] = labels_dict[labels[i]]
    return labels


def gen_onehot_labels_idxs(full_processed, nodes_keys):
    """
    Input:
        1. full_processed (from full_data_process.py)
            An graphDataProcess object that neatly contains:
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix

    Output: onehot_labels
        Matrix of onehot labels (0 = [1,0,0...]) that corrsepond to the given node_keys

    Description:
        Loads a problem-grade map and uses it to generate onehot grades for each problem in a list

    Purpose:
        Generate labels for train/test
    """
    # Get labels as a list data object
    labels = gen_labels_idxs(full_processed, nodes_keys)

    # Get set of labels
    labels_set = list(set(labels))

    # Instantiate and populate one-hot labels
    onehot_labels = np.zeros((len(nodes_keys), len(labels_set)))
    for i, label in enumerate(labels):
        onehot_labels[i][label] = 1
    return onehot_labels
