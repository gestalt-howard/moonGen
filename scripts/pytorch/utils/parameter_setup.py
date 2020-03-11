# Author: Howard Tai

# This script contains code for setting up PyTorch experiments

import os

import numpy as np

from scripts.pytorch.utils.utils import *

from scripts.pytorch.utils.label_functions import *
from scripts.pytorch.utils.feature_functions import *
from scripts.pytorch.utils.adjacency_functions import *


# ----------------------------------------------------------------------------------------------------------------------
# Parameter dictionary functions
# ----------------------------------------------------------------------------------------------------------------------
def get_full_processing_params():
    """
    Defines parameters for saving processed outputs of raw mined MoonBoard data. These parameters are fixed.

    Parameters:
    - node_map_name: Mapping of hold and problem IDs to global node IDs
    - holds_names_name: Mapping of local hold IDs to global node IDs
    - problems_names_name: Mapping of local problem IDs to global node IDs
    - holds_mat_name: Mapping of each global problem ID to a set of global hold IDs
    - pmi_name: Adjacency definitions between hold-hold
    - tfidf_name: Adjacency definitions between problem-hold
    """
    params = {
        'node_map_name': 'node_mapping.pickle',
        'holds_names_name': 'holds_names.pickle',
        'problems_names_name': 'problems_names.pickle',
        'holds_mat_name': 'holds_mat.pickle',
        'pmi_name': 'pmi.pickle',
        'tfidf_name': 'tfidf.pickle'
    }
    return params


def get_full_processing_redo_params(mapping=False, adjacency=False, tfidf=False):
    """
    Defines flags for redoing full processing on mined MoonBoard data

    Input(s):
    - mapping (bool): Redoing node mapping?
    - adjacency (bool): Redoing adjacency (PMI) calculation?
    - tfidf (bool): Redoing tfidf calculation?
    """
    params = {
        'mapping_redo': mapping,
        'adjacency_redo': adjacency,
        'tfidf_redo': tfidf
    }
    return params


def get_subset_processing_params():
    """
    Defines parameters for saving outputs of sub-setting and feature processing. These settings are fixed.

    Parameters:
    - core_nodes_name: List of global node IDs that are 'core nodes'
    - feature_name: Numpy array of feature vectors
    - adjacency_name: Numpy array of adjacency matrix
    - labels_name: Numpy array of node labels (note, holds are labeled as 0)
    """
    params = {
        'core_nodes_name': 'core_nodes.pickle',
        'features_name': 'sampled_features.pickle',
        'adjacency_name': 'sampled_adjacency.pickle',
        'labels_name': 'sampled_labels.pickle'
    }
    return params


def get_subset_processing_redo_params(core_nodes=False):
    """
    Defines flags for redoing subset processing on full-processed MoonBoard data

    Input(s):
    - core_nodes (bool): Re-define core nodes? Everything else follows from this single flag
    """
    flag_names = ['core_nodes_redo', 'feature_redo', 'adjacency_redo', 'label_redo']
    if core_nodes:
        return {n: True for n in flag_names}
    else:
        return {n: False for n in flag_names}


def get_processing_function_params(
        feature='gen_multihotfeatures',
        adjacency='gen_adjacency_diag_norm',
        label='gen_labels_idxs',
        sampling='sample_nodes_balanced_replaced'
):
    """
    Defines processing functions for generating features, adjacency, labels, and conducting sampling
    """
    assert feature in ['gen_onehotfeatures', 'gen_multihotfeatures']
    assert adjacency in ['binary_adjacency_diag_norm', 'gen_adjacency_diag_norm', 'gen_adjacency_diag_norm_diag']
    assert label in ['gen_labels_idxs', 'gen_onehot_labels_idxs']
    assert sampling in ['sample_nodes_balanced', 'sample_nodes_balanced_replaced', 'sample_target_nodes_balanced']

    params = {
        'feature': feature,
        'adjacency': adjacency,
        'label': label,
        'sampling': sampling
    }
    return get_func_dict(params)


def get_split_params(test_ratio=0.2, dev_ratio=0.2):
    """
    Gets parameters for generating:
    - train-dev / test split
    - train / dev split
    """
    params = {
        'test': 1 - test_ratio,
        'dev': 1 - dev_ratio
    }
    return params


def get_sampling_params(num_per_class=100):
    """
    Gets parameters for sampling
    """
    params = {
        'num_per_core': num_per_class,
        'target_grades': list(range(4, 15))
    }
    return params


# ----------------------------------------------------------------------------------------------------------------------
# Model-specific parameters
# ----------------------------------------------------------------------------------------------------------------------
def get_dense_params(hidden_layers, num_epochs, dropout=0.2, lr=0.01, weight_decay=5e-4):
    """
    Gets parameters for training a Dense (fully-connected) neural network
    """
    params = {
        'hidden': hidden_layers,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs
    }
    return params


def get_gcn_params(hidden_layers, num_epochs, dropout=0.2, lr=0.01, weight_decay=5e-4):
    """
    Gets parameters for training a Graph Convolutional network
    """
    params = {
        'hidden': hidden_layers,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs
    }
    return params


# ----------------------------------------------------------------------------------------------------------------------
# Root and main parameter setup function
# ----------------------------------------------------------------------------------------------------------------------
def get_general_params(model_type, version, data_dir, result_dir, raw_data_path, full_redo=False, sub_redo=False):
    """
    Setups general parameters for an experiment

    Input(s):
    - model_type (string): Description of model type (i.e. GCN, Dense)
    - version (string): Descriptive identifier of experiment version
    - data_dir (string): Directory path to main data save folder
    - result_dir (string): Directory path to main result save folder
    - raw_data_path (string): Path to location of raw-mined MoonBoard data
    - full_redo (bool)
    - sub_redo (bool)
    """
    data_path = make_sub_dirs(model_type, version, data_dir)
    result_path = make_sub_dirs(model_type, version, result_dir)

    params = {
        'model_type': model_type,
        'ver': version,
        'full_processed_name': 'full_processed.pickle',
        'sub_processed_name': 'sub_processed.pickle',
        'raw_data_path': raw_data_path,
        'data_dir': data_dir,
        'data_path': data_path,
        'result_path': result_path,
        'full_redo': full_redo,
        'sub_redo': sub_redo
    }
    return params


def get_exp_parameters(
        model_type,
        version,
        data_dir,
        result_dir,
        raw_data_path,
        feature_func,
        adjacency_func,
        label_func,
        sampling_func,
        num_per_class,
        nn_params
):
    """
    Main experiment parameter setup function
    """
    assert model_type in ['GCN', 'Dense']

    params_dict = {
        'gen_params': get_general_params(model_type, version, data_dir, result_dir, raw_data_path),
        'full_names_dict': get_full_processing_params(),
        'full_redo_dict': get_full_processing_redo_params(),
        'sub_names_dict': get_subset_processing_params(),
        'sub_redo_dict': get_subset_processing_redo_params(),
        'sub_functions_dict': get_processing_function_params(feature_func, adjacency_func, label_func, sampling_func),
        'split_ratio_dict': get_split_params(),
        'sampling_params': get_sampling_params(num_per_class)
    }
    if model_type == 'GCN':
        params_dict['model_params'] = get_gcn_params(**nn_params)
    else:
        params_dict['model_params'] = get_dense_params(**nn_params)

    # Save parameters
    param_save_path = params_dict['gen_params']['data_path'] + 'params.pickle'
    save_pickle(params_dict, param_save_path)

    return params_dict


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    return None


if __name__ == '__main__':
    main()
