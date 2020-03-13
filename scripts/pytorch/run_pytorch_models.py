# Author: Howard Tai

# This script contains code for batch-running many PyTorch-based experiments

import copy

import numpy as np

from scripts.pytorch.utils.utils import *
from scripts.evaluation.eval_utils import *
from scripts.evaluation.evaluation_tools import *

from scripts.pytorch.utils.label_functions import *
from scripts.pytorch.utils.feature_functions import *
from scripts.pytorch.utils.adjacency_functions import *

from scripts.pytorch.GCN.GCN import GCN
from scripts.pytorch.Dense.Dense import Dense

from scripts.pytorch.sub_data_process import SubGraphProcess
from scripts.pytorch.full_data_process import GraphDataProcess

from scripts.pytorch.utils.parameter_setup import *
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
        print('Loading full-process object...')
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
        print('Loading sub-process object...')
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
        num_per_class: int,
        hidden_layers: list,
        num_epochs: int
):
    """
    Sets up an experiment, a wrapper for get_exp_parameters()
    """
    # Set neural network params
    nn_params = {
        'hidden_layers': hidden_layers,
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
        'num_per_class': num_per_class,
        'nn_params': nn_params
    }
    return get_exp_parameters(**exp_params)


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
            nhidden_layer_list=params['model_params']['hidden'],
            nclass=num_classes,
            dropout=params['model_params']['dropout']
        )
    else:
        model = Dense(
            nfeatures=num_features,
            nhidden_layer_list=params['model_params']['hidden'],
            nclass=num_classes,
            dropout=params['model_params']['dropout']
        )

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['model_params']['lr'],
        weight_decay=params['model_params']['weight_decay']
    )

    return model, optimizer


# ----------------------------------------------------------------------------------------------------------------------
# Experiment running functions
# ----------------------------------------------------------------------------------------------------------------------
def get_evaluation_save_paths(params, data_type):
    """
    Gets evaluation plots save parameters for a specific data type

    Input(s):
    - params (dict)
    - data_type (string): Either one of 'train' or 'test'
    """
    assert(data_type in ['train', 'test'])

    result_path = params['gen_params']['result_path']
    save_root = result_path + '%s/' % data_type
    make_directory(save_root)

    save_dict = {
        'description': ' '.join([data_type, params['gen_params']['model_type'], params['gen_params']['version']]),
        'corr_fig_save': save_root + 'fig_correlation.png',
        'farp_fig_save': save_root + 'fig_farpa.png',
        'farp_stats_save': save_root + 'stats_farpa.pickle',
        'confusion_fig_save': save_root + 'fig_confusion.png',
        'global_stats_save': save_root + 'stats_global.pickle'
    }
    return save_dict


def run_experiment(params):
    """
    Runs an NN experiment from data generation to model training and evaluation
    """
    print_header('EXPERIMENT: %s --- %s' % (params['gen_params']['model_type'], params['gen_params']['version']))

    # Get data-processing objects
    print_header('GETTING DATA-PROCESSING OBJECTS...')
    graph_data_obj, subgraph_data_obj = process_data(params)

    # Get data splits
    print_header('SPLITTING DATA...')
    features, adj, labels, idx_train, idx_dev, idx_test = get_data_splits(subgraph_data_obj, params)

    # Get model
    print_header('DEFINING MODEL...')
    num_classes = len(np.unique(np.asarray(labels)))
    num_features = features.shape[-1]
    model, optimizer = get_model(params, num_classes, num_features)

    # Train model
    # ------------------------------------------------------------------------------------------------------------------
    print_header('TRAINING MODEL...')
    train_dict = {
        'optimizer': optimizer,
        'features': features,
        'adj': adj,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_dev,
        'num_epochs': params['model_params']['num_epochs']
    }
    model = run_train(model, train_dict)

    # Save model
    save_pickle(model, params['gen_params']['result_path'] + 'model.pickle')

    # Evaluation
    # ------------------------------------------------------------------------------------------------------------------
    # Accuracy on test set
    print_header('EVALUATING MODEL...')
    test_dict = {'features': features, 'adj': adj, 'labels': labels, 'idx_test': idx_test}
    test(model, test_dict)

    # Forward pass on network (inference)
    print('\nRunning inference...')
    output = model(features, adj)

    # Train / Test predictions
    y_pred_trn = np.exp(output[idx_train].detach().numpy())[:, 1:]                  # Drop class 0 (holds)
    y_true_trn = onehot_labels(labels.numpy()[idx_train] - 1, y_pred_trn.shape[1])  # Shift labels by 1

    y_pred_tst = np.exp(output[idx_test].detach().numpy())[:, 1:]                  # Drop class 0 (holds)
    y_true_tst = onehot_labels(labels.numpy()[idx_test] - 1, y_pred_tst.shape[1])  # Shift labels by 1

    # Generate evaluation plots / stats
    trn_save_dict = get_evaluation_save_paths(params, 'train')
    tst_save_dict = get_evaluation_save_paths(params, 'test')

    print('Evaluating train...')
    evaluate_predictions(y_true_trn, y_pred_trn, trn_save_dict)
    print('Evaluating test...')
    evaluate_predictions(y_true_tst, y_pred_tst, tst_save_dict)

    return None


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    return None


if __name__ == '__main__':
    main()
