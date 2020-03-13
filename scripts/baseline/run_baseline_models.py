# Author(s): Howard Tai

# Script for running a battery of baseline models on preprocessed baseline data
# For baseline data generation, reference (gen_baseline_data.py)

import numpy as np

# Baseline model imports
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from scripts.evaluation.eval_utils import *
from scripts.evaluation.evaluation_tools import *


# ----------------------------------------------------------------------------------------------------------------------
# Data dictionary parsing functions
# ----------------------------------------------------------------------------------------------------------------------
def get_train_data(data_dict):
    """
    Assembles training data from data_dict
    """
    x_trn = data_dict['trn']['x']
    x_dev = data_dict['dev']['x']

    y_trn = data_dict['trn']['y'] - 1
    y_dev = data_dict['dev']['y'] - 1

    return np.concatenate([x_trn, x_dev], axis=0), np.concatenate([y_trn, y_dev], axis=0)


def get_test_data(data_dict):
    """
    Assembles test data from data_dict
    """
    return data_dict['tst']['x'], data_dict['tst']['y'] - 1


# ----------------------------------------------------------------------------------------------------------------------
# Setup, prediction, and evaluation wrappers
# ----------------------------------------------------------------------------------------------------------------------
def assemble_save_paths(dir_path, output_names_dict, description):
    """
    Input(s):
    - dir_path (string)
    - output_names_dict (dict)
    """
    save_dict = {'description': description}
    for save_name, output_name in output_names_dict.items():
        save_dict[save_name] = dir_path + output_name
    return save_dict


def get_experiment_params(models_path, description):
    """
    Sets up sub-folders and save paths for a single experiment
    """
    model_path = models_path + description + '/'
    make_directory(model_path)

    trn_root = model_path + 'train/'
    tst_root = model_path + 'test/'

    make_directory(trn_root)
    make_directory(tst_root)

    output_names_dict = {
        'corr_fig_save': 'fig_correlation.png',
        'farp_fig_save': 'fig_farp.png',
        'farp_stats_save': 'stats_farp.pickle',
        'confusion_fig_save': 'fig_confusion.png',
        'global_stats_save': 'stats_global.pickle'
    }

    settings_dict = {
        'model_path': model_path,
        'trn_settings': assemble_save_paths(trn_root, output_names_dict, description),
        'tst_settings': assemble_save_paths(tst_root, output_names_dict, description)
    }
    return settings_dict


def train_model(model, param_dict, data_dict):
    """
    Trains an input model
    """
    model_out_path = param_dict['model_path'] + 'model.pickle'
    if path_exists(model_out_path):
        print('\nModel already exists!')
        model = load_pickle(model_out_path)
    else:
        print('\nTraining model...')
        x_trn, y_trn_true = get_train_data(data_dict)
        model.fit(x_trn, y_trn_true)
        save_pickle(model, model_out_path)
    return model


def get_model_preds(model, x_trn, x_tst):
    """
    Gets model predictions on train and test data-sets and formats into one-hot
    """
    y_trn_pred = model.predict_proba(x_trn)
    y_tst_pred = model.predict_proba(x_tst)
    return y_trn_pred, y_tst_pred


def run_evaluation(settings_dict):
    """
    Determine need for re-running evaluation
    """
    save_paths = []
    for name, path in settings_dict.items():
        if 'save' in name:
            save_paths.append(path)

    if all([path_exists(p) for p in save_paths]):
        print('Evaluation results already exist!')
        return False
    else:
        return True


def get_model_evals(model, param_dict, data_dict):
    """
    Runs model evaluation
    """
    print('\nEvaluating model...')
    x_trn, y_trn = get_train_data(data_dict)
    x_tst, y_tst = get_test_data(data_dict)

    # Run inference
    y_trn_pred, y_tst_pred = get_model_preds(model, x_trn, x_tst)

    # One-hot labels
    y_trn_true = onehot_labels(y_trn, y_trn_pred.shape[1])
    y_tst_true = onehot_labels(y_tst, y_tst_pred.shape[1])

    # Train evaluation
    print('\nTrain evaluation...')
    if run_evaluation(param_dict['trn_settings']):
        evaluate_predictions(y_trn_true, y_trn_pred, param_dict['trn_settings'])

    # Test evaluation
    print('\nTest evaluation...')
    if run_evaluation(param_dict['tst_settings']):
        evaluate_predictions(y_tst_true, y_tst_pred, param_dict['tst_settings'])
    return None


def fit_and_evaluate(model, param_dict, data_dict):
    """
    Wrapper for train_model() and get_model_evals()
    """
    # Fit model, if required
    model = train_model(model, param_dict, data_dict)

    # Model evaluation
    get_model_evals(model, param_dict, data_dict)
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Experiment functions
# ----------------------------------------------------------------------------------------------------------------------
def run_logistic_regression(param_dict, data_dict):
    """
    Defines a logistic regression experiment

    Input(s):
    - param_dict (dict)
    - data_dict (dict)
    """
    # Define logistic regression model
    lr_params = {
        'penalty': 'l2',
        'max_iter': 1000,
        'C': 10,  # Inverse of regularization weight
        'verbose': 1,
        'n_jobs': -1,
        'random_state': 7
    }
    model = LogisticRegression(**lr_params)

    fit_and_evaluate(model, param_dict, data_dict)
    return None


def run_svm(param_dict, data_dict):
    """
    Defines a SVM experiment
    """
    # Define SVM model
    svm_params = {
        'probability': True,
        'max_iter': -1,
        'C': 2,  # Inverse of regularization weight
        'verbose': 1,
        'random_state': 7
    }
    model = SVC(**svm_params)

    fit_and_evaluate(model, param_dict, data_dict)
    return None


def run_random_forest(param_dict, data_dict):
    """
    Defines a random forest experiment
    """
    # Define random forest model
    rf_params = {
        'n_estimators': 500,
        'n_jobs': -1,
        'verbose': 1,
        'random_state': 7,
        'bootstrap': True,
        'max_samples': 0.8
    }
    model = RandomForestClassifier(**rf_params)

    fit_and_evaluate(model, param_dict, data_dict)
    return None


def run_boosting(param_dict, data_dict):
    """
    Defines a boosting experiment
    """
    # Define boosting model
    boosting_params = {
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'verbose': 1,
        'random_state': 7
    }
    model = GradientBoostingClassifier(**boosting_params)

    fit_and_evaluate(model, param_dict, data_dict)
    return None


def run_mlp(param_dict, data_dict):
    """
    Defines a Multi-Layer Perceptron experiment
    """
    # Define MLP model
    mlp_params = {
        'hidden_layer_sizes': (200, 100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'batch_size': 2000,
        'shuffle': True,
        'verbose': 1,
        'random_state': 7
    }
    model = MLPClassifier(**mlp_params)

    fit_and_evaluate(model, param_dict, data_dict)
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    """
    Runs baseline experiments
    """
    # Root path
    root_path = 'C:/Users/chetai/Desktop/'  # CHANGE TO PROPER DIRECTORY
    baseline_dir = root_path + 'moonboard_baseline/'

    # Models path
    models_path = baseline_dir + 'models/'
    make_directory(models_path)

    # Load data
    data_dir = baseline_dir + 'baseline_data/'
    data_path = data_dir + 'data_dict.pickle'
    data_dict = load_pickle(data_path)

    # Setup baseline experiments
    exp_dict = {
        0: 'logistic_regression',
        1: 'svm',
        2: 'random_forest',
        3: 'boosting',
        4: 'mlp'
    }
    params_dict = {}
    for i, description in exp_dict.items():
        # Get parameters
        params_dict[i] = get_experiment_params(models_path, description)

    # Run baseline experiments
    print_header('LOGISTIC REGRESSION')
    run_logistic_regression(params_dict[0], data_dict)

    print_header('SUPPORT VECTOR MACHINE')
    run_svm(params_dict[1], data_dict)

    print_header('RANDOM FOREST')
    run_random_forest(params_dict[2], data_dict)

    print_header('GRADIENT BOOSTING')
    run_boosting(params_dict[3], data_dict)

    print_header('MULTI-LAYER PERCEPTRON')
    run_mlp(params_dict[4], data_dict)
    return None


if __name__ == '__main__':
    main()
