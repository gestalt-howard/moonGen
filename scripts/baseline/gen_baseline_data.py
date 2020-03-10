# Author(s): Howard Tai

# Script for setting up / formatting data for running baseline experiments on. Some key details are:
# - Sampling with replacement for balanced class sizes
# - 2000 samples per class out of 11 classes (V4 - V14)
# - Multi-hot features

import numpy as np

from scripts.pytorch.sub_data_process import SubGraphProcess
from scripts.pytorch.full_data_process import GraphDataProcess

from scripts.pytorch.utils.utils import get_func_dict, sample_and_load_pytorch_data
from scripts.evaluation.eval_utils import path_exists, save_pickle, load_pickle, make_directory


# ----------------------------------------------------------------------------------------------------------------------
# Data processing functions
# ----------------------------------------------------------------------------------------------------------------------
def gen_full_processed_data(raw_data_path, save_data_dir):
    """
    Generates object wrapper on data structure containing processed version raw mined MoonBoard data
    (Reference scripts.pytorch.full_data_process.py)

    Input(s):
    - raw_data_path (string): Path to raw mined MoonBoard data
    - save_data_dir (string): Directory to store processed data

    Output(s):
    GraphDataProcess object
    """
    save_path = save_data_dir + 'full_processed.pickle'

    if path_exists(save_path):
        return load_pickle(save_path), save_path

    full_processed_obj = GraphDataProcess(raw_data_path, save_data_dir)
    full_processed_obj.run_all()

    save_pickle(full_processed_obj, save_path)
    return full_processed_obj, save_path


def gen_sub_processed_data(full_processed_path, save_data_dir):
    """
    Generates object wrapper on data structure containing sampled and feature-processed version of full_processed() data
    (Reference scripts.pytorch.sub_data_process.py)

    Input(s):
    - raw_data_path (string): Path to GraphDataProcess object
    - save_data_dir (string): Directory to store processed data
    """
    # Specify processing functions (strings)
    sub_functions_dict = {
        'feature': 'gen_multihotfeatures',
        'adjacency': 'gen_adjacency_diag_norm',
        'label': 'gen_labels_idxs',
        'sampling': 'sample_nodes_balanced_replaced'
    }
    # Dictionary of processing functions (functions)
    sub_functions_dict = get_func_dict(sub_functions_dict)

    # Specify sampling settings
    sampling_params = {
        'num_per_core': 2000,
        'target_grades': list(range(4, 15)),
        'sample_nodes_path': save_data_dir + 'core_nodes.pickle'
    }

    save_path = save_data_dir + 'sub_processed.pickle'
    if path_exists(save_path):
        return load_pickle(save_path)

    # Define SubGraphProcess object and run processing
    sub_processed_obj = SubGraphProcess(
        full_processed_path,
        save_data_dir,
        functions_dict=sub_functions_dict,
        sampling_params=sampling_params
    )
    sub_processed_obj.run_all()

    save_pickle(sub_processed_obj, save_path)
    return sub_processed_obj


# ----------------------------------------------------------------------------------------------------------------------
# Split formatting functions
# ----------------------------------------------------------------------------------------------------------------------
def get_data_split(features, labels, idx_set):
    """
    Given a tensors of features and labels, return trn-dev-tst splits

    Input(s):
    - features (PyTorch tensor)
    - labels (PyTorch tensor)
    - idx_set (PyTorch tensor)
    """
    x = features[idx_set].numpy()
    y = labels[idx_set].numpy()
    return x, y


def get_features_and_labels(sub_processed_obj, pytorch_data_dir, baseline_data_dir):
    """
    Parses SubGraphProcess object to get features and labels

    Input(s):
    - sub_processed_obj (SubGraphProcess object)
    - save_data_dir (string): Directory to store processed data
    """
    # Define split ratios
    # 80:20 train-dev:test split
    # 80:20 train:dev split
    split_ratio_dict = {'test': .8, 'dev': .8}

    # Get splits
    features, adj, labels, idx_train, idx_dev, idx_test = sample_and_load_pytorch_data(
        sub_processed_obj,
        split_ratio_dict,
        pytorch_data_dir
    )

    x_trn, y_trn = get_data_split(features, labels, idx_train)
    x_dev, y_dev = get_data_split(features, labels, idx_dev)
    x_tst, y_tst = get_data_split(features, labels, idx_test)

    # Save data
    save_path = baseline_data_dir + 'data_dict.pickle'
    data_dict = {
        'trn': {'x': x_trn, 'y': y_trn},
        'dev': {'x': x_dev, 'y': y_dev},
        'tst': {'x': x_tst, 'y': y_tst}
    }
    save_pickle(data_dict, save_path)
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    """
    Executes data setup for baseline models
    """
    # Root path
    root_path = 'C:/Users/chetai/Desktop/'

    # Raw data path
    raw_data_path = root_path + 'moonboard_data.pickle'

    # Sub save directories
    baseline_dir = root_path + 'moonboard_baseline/'
    full_processed_dir = baseline_dir + 'full_processed/'
    sub_processed_dir = baseline_dir + 'sub_processed/'
    pytorch_data_dir = baseline_dir + 'pytorch_data/'
    baseline_data_dir = baseline_dir + 'baseline_data/'

    # Create directories
    dir_list = [baseline_dir, full_processed_dir, sub_processed_dir, pytorch_data_dir, baseline_data_dir]
    for dir_path in dir_list:
        make_directory(dir_path)

    # Get full-processed data
    full_processed_obj, full_processed_path = gen_full_processed_data(raw_data_path, full_processed_dir)

    # Get sub-processed data
    sub_processed_obj = gen_sub_processed_data(full_processed_path, sub_processed_dir)

    # Get data splits
    get_features_and_labels(sub_processed_obj, pytorch_data_dir, baseline_data_dir)
    return None


if __name__ == '__main__':
    main()
