# Author: Howard Tai

# This script contains code for batch-running many PyTorch-based experiments

import copy

from scripts.pytorch.utils.utils import *
from scripts.evaluation.eval_utils import *

from scripts.pytorch.utils.label_functions import *
from scripts.pytorch.utils.feature_functions import *
from scripts.pytorch.utils.adjacency_functions import *

from scripts.pytorch.GCN.GCN import GCN
from scripts.pytorch.Dense.Dense import Dense

from scripts.pytorch.full_data_process import GraphDataProcess
from scripts.pytorch.sub_data_process import SubGraphProcess

from scripts.pytorch.utils.train_test_functions import *


# ----------------------------------------------------------------------------------------------------------------------
# Full and subset processing wrappers
# ----------------------------------------------------------------------------------------------------------------------
def full_graph_process(param_dict):
    """
    Wrapper for processing data on the full mined data-set

    Input(s):
    - param_dict (dict)

    Output(s):
    GraphDataProcess object
    """
    # Parse main save directory, data save path, and names of intermediate files
    data_dir = param_dict['gen_params']['data_dir']
    full_processed_path = data_dir + param_dict['gen_params']['full_processed_name']
    full_names_dict = param_dict['full_names_dict']

    # Parse path to mined data
    raw_data_path = param_dict['gen_params']['raw_data_path']

    # Parse flags for redoing calculation
    full_redo_dict = param_dict['full_redo_dict']

    if not path_exists(full_processed_path):
        # Create processing object and execute
        graph_data_obj = GraphDataProcess(raw_data_path, data_dir, full_names_dict, full_redo_dict)
        graph_data_obj.run_all()
        save_pickle(graph_data_obj, full_processed_path)
    else:
        graph_data_obj = load_pickle(full_processed_path)

    return graph_data_obj


def sub_graph_process(param_dict):
    """
    Wrapper for sampling data subset and organizing model input features

    Input(s):
    - param_dict (dict)

    Output(s):
    SubGraphProcess object
    """
    # Parse main data save path
    data_dir = param_dict['gen_params']['data_dir']
    full_processed_path = data_dir + param_dict['gen_params']['full_processed_name']

    # Parse sub data save path
    data_path = param_dict['gen_params']['data_path']
    sub_processed_path = data_path + param_dict['gen_params']['sub_processed_name']

    # Unpack processing parameters
    sub_names_dict = param_dict['sub_names_dict']
    sub_redo_dict = param_dict['sub_redo_dict']
    sub_functions_dict = param_dict['sub_functions_dict']
    sampling_params = param_dict['sampling_params']

    if not path_exists(sub_processed_path):
        # Create sampling object and execute
        subgraph_data_obj = SubGraphProcess(
            full_processed_path,
            data_path,
            sub_names_dict,
            sub_redo_dict,
            sub_functions_dict,
            sampling_params
        )
        subgraph_data_obj.run_all()
        save_pickle(subgraph_data_obj, sub_processed_path)
    else:
        subgraph_data_obj = load_pickle(sub_processed_path)

    return subgraph_data_obj


# ----------------------------------------------------------------------------------------------------------------------
# Experiment setup functions
# ----------------------------------------------------------------------------------------------------------------------
def setup_experiment(
        model_type: str,
        version: str,
        data_dir: str,
        result_dir: str,
        feature_func: str,
        adjacency_func: str,
        hidden_layers: list,
        num_epochs: int
):
    """
    Sets up an experiment, a wrapper for get_exp_parameters()
    """
    # Set neural network params
    nn_params = {
        'hidden': hidden_layers,
        'dropout': 0.2,
        'lr': 0.01,
        'weight_decay': 0.0005,
        'num_epochs': num_epochs
    }

    # Set meta params
    exp_params = {
        'model_type': model_type,
        'version': version,
        'data_dir': data_dir,
        'result_dir': result_dir,
        'raw_data_path': 'C:/Users/chetai/Desktop/moonboard_data.pickle',
        'feature_func': feature_func,
        'adjacency_func': adjacency_func,
        'label_func': 'gen_labels_idxs',
        'sampling_func': 'sample_nodes_balanced_replaced',
        'num_per_class': 2000,
        'nn_params': nn_params
    }
    return exp_params


def process_data(params):
    """
    Wrapper for full_graph_process() and sub_graph_process()

    Output(s):
    - GraphDataProcess object
    - SubGraphProcess object
    """
    graph_data_obj = full_graph_process(params)
    subgraph_data_obj = sub_graph_process(params)
    return graph_data_obj, subgraph_data_obj


def get_data_splits(subgraph_data_obj, params, target_trade=-1):
    """
    Wrapper for sample_and_load_pytorch_data(). This will save trn-dev-tst indexes in (2) locations: at the respective
    data and result sub-folders.

    Input(s):
    - SubGraphProcess object
    - params (dict)
    - target_grade (int): Target grade for binary classification or -1 (default) for multi-class
    """
    # This step already saves trn-dev-tst indexes in results sub-folder
    features, adj, labels, idx_train, idx_dev, idx_test = sample_and_load_pytorch_data(
        subgraph_data_obj,
        params['split_ratio_dict'],
        params['gen_params']['result_path'],
        target_trade
    )

    # This step saves trn-dev-tst indexes in data sub-folder
    data_path = params['gen_params']['data_path']

    idx_trn_path = data_path + 'idx_train.pickle'
    idx_dev_path = data_path + 'idx_dev.pickle'
    idx_tst_path = data_path + 'idx_test.pickle'

    save_pickle(idx_train.numpy(), idx_trn_path)
    save_pickle(idx_dev.numpy(), idx_dev_path)
    save_pickle(idx_test.numpy(), idx_tst_path)

    return features, adj, labels, idx_train, idx_dev, idx_test


# ----------------------------------------------------------------------------------------------------------------------
# Model setup functions
# ----------------------------------------------------------------------------------------------------------------------
def get_model(params, num_classes, num_features):
    """
    Defines a neural network model, given input parameters
    """
    model_type = params['gen_params']['model_type']

    if model_type == 'GCN':
        model = GCN(
            nfeatures=num_features,
            nhidden_layer_list=params['nn_params']['hidden'],
            nclass=num_classes,
            dropout=params['nn_params']['dropout']
        )

    else:
        model = Dense(
            nfeatures=num_features,
            nhidden_layer_list=params['nn_params']['hidden'],
            nclass=num_classes,
            dropout=params['nn_params']['dropout']
        )

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['nn_params']['lr'],
        weight_decay=params['nn_params']['weight_decay']
    )

    return model, optimizer


# ----------------------------------------------------------------------------------------------------------------------
# Experiment running functions
# ----------------------------------------------------------------------------------------------------------------------
def


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    return None


if __name__ == '__main__':
    main()
