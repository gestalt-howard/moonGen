# Authors: Aaron Wu / Howard Tai

# Script defining various utility functions

import os
import pdb
import torch
import pickle

import numpy as np

from scripts.pytorch.utils.label_functions import gen_labels_idxs, gen_onehot_labels_idxs
from scripts.pytorch.utils.feature_functions import *
from scripts.pytorch.utils.adjacency_functions import *


# ----------------------------------------------------------------------------------------------------------------------
# Sampling Functions
# ----------------------------------------------------------------------------------------------------------------------
def sample_nodes_balanced(nodes_grades_dict, params):
    """
    Input:
        1. nodes_grades_dict (dict)
            A dictionary where:
            1. The keys are processed node names (n1, n2, etc) for problems (global IDs)
            2. The values are transformed grade of that problem (0,1,2,3...)
               Grades are mapped from Moonboard scale to V scale to a 0-intercept scale
        2. params (dict)
            1. num_per_core = number of nodes per unique grade (i.e. sampling size)
            2. target_grades = isolated set of grades to run on instead of the full set

    Output: node_samples
        List of shuffled processed node names (n1, n2, etc) sampled from the full set.

    Description:
        1. Maps each grade to a list of nodes
        2. Randomly samples n nodes from each grade
        3. No replacement: if the number of nodes per core exceeds the full set, no more will be retrieved

    Purpose:
        Sample a subset of the full set of nodes for:
            1. Balancing classes
            2. Subsetting a network that might be too large for memory
    """
    # Unpack number of samples per difficulty
    num_per_core = params['num_per_core']

    # Construct dictionary of key: grade - value: list of problem IDs
    grades_dict = {}
    for node in nodes_grades_dict:
        if nodes_grades_dict[node] in grades_dict:
            grades_dict[nodes_grades_dict[node]].append(node)
        else:
            grades_dict[nodes_grades_dict[node]] = [node]

    # Define target grades, optionally load from params
    target_grades = list(grades_dict.keys())
    if 'target_grades' in params:
        target_grades = params['target_grades']

    # Randomly sample (num_per_core) samples from each difficulty grade
    node_samples = []
    for grade in target_grades:
        shuffle = np.random.permutation(len(grades_dict[grade]))
        node_samples += [grades_dict[grade][i] for i in shuffle[:num_per_core]]
    return node_samples


def sample_nodes_balanced_replaced(nodes_grades_dict, params):
    """
    Input:
        1. nodes_grades_dict
            A dictionary where:
            1. The keys are processed node names (n1, n2, etc) for problems
            2. The values are transformed grade of that problem (0,1,2,3...)
               Grades are mapped from MoonBoard scale to V scale to a 0-intercept scale
        2. params
            1. num_per_core = number of nodes per unique grade
            2. target_grades = isolated set of grades to run on instead of the full set

    Output: node_samples
        List of shuffled processed node names (n1, n2, etc) sampled from the full set.

    Description:
        1. Maps each grade to a list of nodes
        2. Randomly samples n nodes from each grade
        3. With replacement: nodes already sampled can still be sampled

    Purpose:
        Sample a subset of the full set of nodes for:
            1. Balancing classes
            2. Sub-setting a network that might be too large for memory
    """
    # Unpack number of samples per difficulty
    num_per_core = params['num_per_core']

    # Construct dictionary of key: grade - value: list of problem IDs
    grades_dict = dict()
    for node in nodes_grades_dict:
        if nodes_grades_dict[node] in grades_dict:
            grades_dict[nodes_grades_dict[node]].append(node)
        else:
            grades_dict[nodes_grades_dict[node]] = [node]

    # Define target grades as either a list of grades or a single grade
    target_grades = list(grades_dict.keys())
    if 'target_grades' in params:
        target_grades = params['target_grades']

    # Instantiate list to store sampled nodes
    node_samples = []

    # Sample from a specific grade / difficulty level
    for grade in target_grades:
        for i in range(num_per_core):
            random_idx = np.random.randint(0, len(grades_dict[grade]))
            node_samples += [grades_dict[grade][random_idx]]  # Take one sample

    return node_samples


def sample_target_nodes_balanced(nodes_grades_dict, params):
    """
    Input:
        1. nodes_grades_dict
            A dictionary where:
            1. The keys are processed node names (n1, n2, etc) for problems
            2. The values are transformed grade of that problem (0,1,2,3...)
               Grades are mapped from MoonBoard scale to V scale to a 0-intercept scale
        2. params
            1. num_per_core = number of nodes per unique grade
            2. target_grade = a single grade to sample from

    Output: node_samples
        List of shuffled processed node names (n1, n2, etc) sampled from the full set.

    Description:
        1. Maps each grade to a list of nodes
        2. Randomly samples n nodes from each grade except the target grade
        3. Takes only m nodes from this set of non-target nodes
        4. Randomly samples m nodes from the target grade

    Purpose:
        Sample a subset of the full set of nodes for:
            1. Balancing number of target nodes and non-target nodes (when it's 1 class vs k classes)
            2. Subsetting a network that might be too large for memory
    """
    # Parse parameters
    num_per_core = params['num_per_core']
    target_grade = params['target_grade']

    # Construct dictionary of key: grade - value: list of problem IDs
    grades_dict = {}
    for node in nodes_grades_dict:
        if nodes_grades_dict[node] in grades_dict:
            grades_dict[nodes_grades_dict[node]].append(node)
        else:
            grades_dict[nodes_grades_dict[node]] = [node]

    # Sample nodes of non-target difficulty
    nontarget_samples = []
    for grade in grades_dict:
        if grade == target_grade:
            continue
        shuffle = np.random.permutation(len(grades_dict[grade]))
        nontarget_samples += [grades_dict[grade][i] for i in shuffle[:num_per_core]]

    # Sample nodes of target difficulty
    target_samples = []
    if target_grade in grades_dict:
        shuffle = np.random.permutation(len(grades_dict[target_grade]))
        target_samples = [grades_dict[target_grade][i] for i in shuffle[:num_per_core]]

    # Shuffle non-target samples and create master node set
    shuffle = np.random.permutation(len(nontarget_samples))[:num_per_core]
    node_samples = target_samples + [nontarget_samples[i] for i in shuffle]

    return node_samples


# ----------------------------------------------------------------------------------------------------------------------
# Train Dev Test Split Functions
# ----------------------------------------------------------------------------------------------------------------------
def split_nodes(node_set, split_ratio):
    """
    Input:
        1. node_set = full set of nodes to be split
        2. split_ratio = ratio to split the set between 2 subsets

    Output:
        1. set1 = set that contains n*split_ratio values
        2. set2 = set that contains n - n*split_ratio values

    Description:
        Randomly splits values in a set based on a split_ratio

    Purpose:
        Splitting between train-dev and test, and train and dev
    """
    num_split = int(len(node_set) * split_ratio)
    shuffle = np.random.permutation(len(node_set))
    set1 = [node_set[i] for i in shuffle[:num_split]]
    set2 = [node_set[i] for i in shuffle[num_split:]]
    return set1, set2


def train_dev_test_split(node_set, split_ratio_dict):
    """
    Input:
        1. node_set = full set of nodes to be split
        2. split_ratio = dictionary that dictates the test split and the dev split

    Output:
        1. train_set = nodes to be considered for train
        2. dev_set = nodes to be considered for validation
        3. test_set = nodes to be considered for test

    Description:
        Splits a set of nodes into train, dev, test.

    Purpose:
        Total data needs to be split in a consistent manner
    """
    train_dev_set, test_set = split_nodes(node_set, split_ratio_dict['test'])
    train_set, dev_set = split_nodes(train_dev_set, split_ratio_dict['dev'])
    return train_set, dev_set, test_set


def get_split_dict(core_nodes_id_dict, hold_nodes_id_dict, split_ratio_dict):
    """
    Input
        1. core_nodes_id_dict:
           A dictionary that maps sampled problem nodes to an index in feature / adjacency / label (from
           sub_data_process.py)
           Note: core nodes can be mapped to multiple indices for up-sampling purposes
        2. hold_nodes_id_dict = dictionary that maps sampled hold nodes to an index in feature/adjacency/label
           (from sub_data_process.py)
        3. split_ratio_dict = dictionary that dictates the test split and the dev split

    Output: split_dict
        Dictionary of train, dev, test feature / adjacency / label indices that correspond to nodes in their set

    Description:
        1. Get core problem nodes and split them into train, dev, test
           * Hold nodes aren't considered since they don't have labels
        2. Get the indices corresponding to the problem nodes (can be multiple for one problem)

    Purpose:
        Total data needs to be split in a consistent manner
    """
    # Instantiate dictionary of train-dev-test splits
    split_dict = dict()

    # Parse out problem nodes, represented using global node IDs
    core_nodes = list(core_nodes_id_dict.keys())

    # Get train, dev, test sets
    train_set, dev_set, test_set = train_dev_test_split(core_nodes, split_ratio_dict)

    # Populate train set IDs (feature / adjacency / label)
    split_dict['train_idxs'] = []
    for i in train_set:
        split_dict['train_idxs'] += core_nodes_id_dict[i]

    # Populate dev set IDs (feature / adjacency / label)
    split_dict['dev_idxs'] = []
    for i in dev_set:
        split_dict['dev_idxs'] += core_nodes_id_dict[i]

    # Populate test set IDs (feature / adjacency / label)
    split_dict['test_idxs'] = []
    for i in test_set:
        split_dict['test_idxs'] += core_nodes_id_dict[i]

    # Parse hold indexes
    holds_idxs = [hold_nodes_id_dict[i] for i in hold_nodes_id_dict]
    split_dict['hold_idxs'] = holds_idxs

    return split_dict


# ----------------------------------------------------------------------------------------------------------------------
# General Utils
# ----------------------------------------------------------------------------------------------------------------------
def load_pickle(path):
    """
    Loads a pickled object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, path):
    """
    Saves a data object in a pickled format
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return None


def set_default(current_val, fail_val, default_val):
    """
    Input:
        1. current_val (string) = current value being considered
        2. fail_val (string) = value used to determine if current_val is a failure
        3. default_val (string) = default assigned value if current_val==fail_val

    Output: current_val

    Description:
        current_val = default_val if failed
        otherwise current_val = default_val

    Purpose:
        Checks and sets a default value
    """
    if current_val == fail_val:
        return default_val
    return current_val


def set_default_dict(current_dict, key, fail_val, default_val):
    """
    Input:
        1. current_dict (dict) = current dictionary being considered
        2. key (string) = current key of current_dict being considered
        3. fail_val (string) = value used to determine if current_dict[key] is a failure
        4. default_val (string) = default assigned value if current_dict[key]==fail_val

    Output: current_dict[key] (string)

    Description:
        current_dict[key] = default_val if failed

    Purpose:
        Checks and sets a default value for dictionary key
    """
    if key not in current_dict or current_dict[key] == fail_val:
        return default_val
    return current_dict[key]


def rev_dict(mapping):
    """
    Inverts the mapping of a dictionary
    """
    rev_mapping = {}
    for m in mapping:
        rev_mapping[mapping[m]] = m
    return rev_mapping


def remove_redo_paths(redo, paths_list):
    """
    Deletes all items in a given list of paths from hard-drive memory.
    Used to help facilitate re-doing a section of the pipeline.

    Input(s):
    - redo (bool): Flag for initiating memory-clearing operation
    - paths_list (list of strings): List of paths
    """
    if redo:
        for p in paths_list:
            if os.path.exists(p):
                os.remove(p)
    return None


def str_to_func(func_str):
    """
    Retrieves the function whose name corresponds to a given string
    """
    try:
        function = globals()[func_str]
    except KeyError:
        pdb.set_trace()
        function = locals()[func_str]
    return function


def get_func_dict(str_dict):
    """
    Turns a dictionary of strings that contain function names into a dictionary that contains those functions
    """
    func_dict = {}
    for func in str_dict:
        func_dict[func] = str_to_func(str_dict[func])
    return func_dict


# ----------------------------------------------------------------------------------------------------------------------
# PyTorch Functions
# ----------------------------------------------------------------------------------------------------------------------
def sample_and_load_pytorch_data(subgraph_data_obj, split_ratio_dict, save_path, target_label=-1, redo=False):
    """
    Input:
        1. subgraph_data_obj (refer to sub_data_process.py)
            A SubGraphProcess object that neatly contains:
            1. Mapping of the problem nodes to their respective indexes in features/adjacency/labels
               (core_nodes_id_dict)
            2. Mapping of the hold nodes to their respective indexes in features/adjacency/labels (hold_nodes_id_dict)
            3. Generated matrix of features for sub-sampled problem nodes + hold nodes (features)
            4. Generated adjacency matrix for sub-sampled problem nodes + hold nodes (adjacency)
            5. Generated list of labels for sub-sampled problem nodes + hold nodes (labels)
        2. split_ratio_dict = dictionary that dictates the test split and the dev split
        3. save_path = path to save some intermediate outputs before training
        4. target_label = target grade for binary classification (-1 for multi-class)
        5. redo = deletes intermediate outputs if True (will load saved intermediates otherwise)

    Output:
        1. features = features to be used by the model
        2. adj = adjacency matrix to be used by the model
        3. labels = labels that correspond to the features
        4. idx_train = indices for training loss
        5. idx_dev = indices for validation during training
        6. idx_test = indices for testing

    Description:
        1. Retrieves the core problem nodes and the hold nodes
        2. Split the data into lists of indices based on given split_dict
        3. Retrieve features/labels/adjacency and convert them to torch tensors
        4. Save intermediate files

    Purpose:
        Sampling done prior to training and testing
    """
    files = [
        'features.pickle',
        'adj.pickle',
        'labels.pickle',
        'idx_train.pickle',
        'idx_dev.pickle',
        'idx_test.pickle'
    ]
    remove_redo_paths(redo, [save_path + f for f in files])

    # Load tensors if they already exist
    if all(os.path.exists(save_path + file) for file in files):
        features = load_pickle(save_path + 'features.pickle')
        adj = load_pickle(save_path + 'adj.pickle')
        labels = load_pickle(save_path + 'labels.pickle')
        idx_train = load_pickle(save_path + 'idx_train.pickle')
        idx_dev = load_pickle(save_path + 'idx_dev.pickle')
        idx_test = load_pickle(save_path + 'idx_test.pickle')
    else:
        # Define data splits
        core_nodes_id_dict = subgraph_data_obj.core_nodes_id_dict
        hold_nodes_id_dict = subgraph_data_obj.hold_nodes_id_dict
        split_dict = get_split_dict(core_nodes_id_dict, hold_nodes_id_dict, split_ratio_dict)

        # Cast features and adjacency to tensors
        features = torch.tensor(subgraph_data_obj.features, dtype=torch.float32)
        adj = torch.tensor(subgraph_data_obj.adjacency, dtype=torch.float32)

        # Choose between binary classification or multi-class (-1)
        if target_label != -1:
            labels = torch.tensor((subgraph_data_obj.labels == target_label) * 1, dtype=torch.int64)
        else:
            labels = torch.tensor(subgraph_data_obj.labels, dtype=torch.int64)

        # Define index of splits and tensors
        idx_train = torch.tensor(split_dict['train_idxs'], dtype=torch.int64)
        idx_dev = torch.tensor(split_dict['dev_idxs'], dtype=torch.int64)
        idx_test = torch.tensor(split_dict['test_idxs'], dtype=torch.int64)

        # Save features
        save_pickle(features, save_path + 'features.pickle')
        save_pickle(adj, save_path + 'adj.pickle')
        save_pickle(labels, save_path + 'labels.pickle')
        save_pickle(idx_train, save_path + 'idx_train.pickle')
        save_pickle(idx_dev, save_path + 'idx_dev.pickle')
        save_pickle(idx_test, save_path + 'idx_test.pickle')

    return features, adj, labels, idx_train, idx_dev, idx_test
