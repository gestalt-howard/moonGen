# Author: Howard Tai

# Script containing evaluation tools for assessing model performance

import pdb

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from scripts.evaluation.eval_utils import path_exists
from scripts.evaluation.eval_utils import save_pickle, load_pickle


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation helper functions
# ----------------------------------------------------------------------------------------------------------------------
def aggregate_prob_with_window(y_true, y_pred, window_size=0):
    """
    For each class, re-define probabilities and labels in terms of a sliding window
    Used for calculation of per-class recall and precision

    Input(s):
    - y_true: (numpy array) True difficulty label shape [m x num_classes], one-hot vector
    - y_pred: (numpy array) Predicted difficulty labels shape [m x num_classes], vector of probabilities
    - window_size: (int) Sliding window size

    Output(s):
    - Aggregate true labels and aggregate prediction probabilities
    """
    m, n = y_true.shape
    assert(m == y_pred.shape[0])
    assert(n == y_pred.shape[1])

    if window_size == 0:
        return y_true, y_pred

    aggregate_trues = np.empty_like(y_true)
    aggregate_preds = np.empty_like(y_pred)

    # Iterate over features
    for i in range(n):
        s_idx = max(0, i - window_size)
        e_idx = min(n-1, i + window_size) + 1

        aggregate_trues[:, i] = np.sum(y_true[:, s_idx:e_idx], axis=1)
        aggregate_preds[:, i] = np.sum(y_pred[:, s_idx:e_idx], axis=1)

    return aggregate_trues, aggregate_preds


def argmax_with_window(y_true, y_pred, window_size=0):
    """
    Recalculates predicted labels, factoring in for a specified window size
    Used for adapting sliding window concept for confusion matrices and correlation plots

    Input(s):
    - y_true: (numpy array) True difficulty label shape [m x num_classes], one-hot vector
    - y_pred: (numpy array) Predicted difficulty labels shape [m x num_classes], vector of probabilities
    - window_size: (int) Sliding window size

    Output(s):
    - True labels shape [m] and corrected predicted labels shape [m]
    """
    # Get most-likely classes
    y_true_cls = np.argmax(y_true, axis=1).reshape(-1) + 4  # Add 4 to adjust to V-scale difficulties
    y_pred_cls = np.argmax(y_pred, axis=1).reshape(-1) + 4

    y_pred_template = np.empty_like(y_pred_cls)

    if window_size == 0:
        return y_true_cls, y_pred_cls
    else:
        # Find difference between true and predicted classes
        true_pred_diff = np.abs(np.subtract(y_true_cls, y_pred_cls))

        # Find which predictions are within a tolerable range
        match_idx = np.where(true_pred_diff <= window_size)[0]
        diff_idx = np.where(true_pred_diff > window_size)[0]

        # Re-define predictions based on diff against true labels
        y_pred_template[match_idx] = y_true_cls[match_idx]
        y_pred_template[diff_idx] = y_pred_cls[diff_idx]

        return y_true_cls, y_pred_template


def find_index_balanced_rp(recalls, precisions):
    """
    Given a list of recalls and precisions, find the index corresponding to the least difference between the two

    Input(s):
    - recalls: (list)
    - precisions: (list)

    Output(s):
    - match_index: (int)
    """
    r_array = np.asarray(recalls)
    p_array = np.asarray(precisions)

    diff = np.abs(np.subtract(r_array, p_array))
    return np.argmin(diff)


# ----------------------------------------------------------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------------------------------------------------------
def plot_label_correlation(y_true, y_pred, title, ax, window_size=0):
    """
    Plots correlation between true and predicted labels

    Input(s):
    - y_true: (numpy array) True difficulty label shape [m x num_classes], one-hot vector
    - y_pred: (numpy array) Predicted difficulty labels shape [m x num_classes], vector of probabilities
    - title: (string) Title of correlation plot
    - ax: Matplotlib axis object
    - window_size: (int) Size of sliding window
    """
    # Collapsing class probabilities
    agg_true, agg_pred = argmax_with_window(y_true, y_pred, window_size=window_size)

    # Fit linear trend line
    z = np.polyfit(agg_true, agg_pred, 1)
    p = np.poly1d(z)

    # Extract trend statistics
    slope, bias = p.c
    corr, pval = pearsonr(agg_true, agg_pred)
    title_text = '%s\nslope: %.3f, bias %.3f\ncorr: %.3f, pval: %.3f' % (title, slope, bias, corr, pval)

    # Plot correlation
    ax.scatter(agg_true, agg_pred, s=4, c='red')
    ax.plot(agg_true, p(agg_true))
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    ax.set_title(title_text)

    return None


# ----------------------------------------------------------------------------------------------------------------------
# FARPA
# ----------------------------------------------------------------------------------------------------------------------
def get_farp_scores(y_true, y_pred, window_size=0):
    """
    Generates F1 score, accuracy, recall, and precision factoring in sliding windows over correct labels
    Also calculates AUC

    Input(s):
    - y_true: (numpy array) True difficulty label shape [m x num_classes], one-hot vector
    - y_pred: (numpy array) Predicted difficulty labels shape [m x num_classes], vector of probabilities
    - ax: Matplotlib axis object
    - title: (string) Title of correlation plot
    - window_size: (int)

    Output(s):
    - Dictionary of statistics for each difficulty class
    """
    farpa_dict = {}

    # Define thresholds
    thresholds = np.linspace(0.01, 0.99, 99)

    # Get aggregate labels and predictions
    agg_true, agg_pred = aggregate_prob_with_window(y_true, y_pred, window_size=window_size)

    # Get FARPA scores for each class
    for i in range(agg_true.shape[1]):
        f_scores = []
        a_scores = []
        r_scores = []
        p_scores = []

        for t in thresholds:
            true_t = agg_true[:, i]
            pred_t = agg_pred[:, i] >= t

            f_scores.append(f1_score(true_t, pred_t, zero_division=0))
            a_scores.append(accuracy_score(true_t, pred_t))
            r_scores.append(recall_score(true_t, pred_t, zero_division=0))
            p_scores.append(precision_score(true_t, pred_t, zero_division=0))

        # AUC calculation
        try:
            auc_score = roc_auc_score(agg_true[:, i], agg_pred[:, i])
        except ValueError:
            auc_score = np.nan

        farpa_dict['V%s' % (i+4)] = {
            'thresholds': thresholds,
            'f_scores': f_scores,
            'a_scores': a_scores,
            'r_scores': r_scores,
            'p_scores': p_scores,
            'auc_score': auc_score
        }
    return farpa_dict


def plot_single_farp_curve(v_dict, ax, title):
    """
    Plots a single FARP curve

    Input(s):
    - v_dict: (dict) Dictionary of calculated FARP scores for a given difficulty class
    - ax: Matplotlib axis object
    - title (string) Title of plot
    """
    ax.plot(v_dict['thresholds'], v_dict['f_scores'], label='F1')
    ax.plot(v_dict['thresholds'], v_dict['a_scores'], label='Accuracy')
    ax.plot(v_dict['thresholds'], v_dict['r_scores'], label='Recall')
    ax.plot(v_dict['thresholds'], v_dict['p_scores'], label='Precision')
    ax.set_xlabel('Thresholds')
    ax.set_title(title)
    ax.legend()
    return None


def get_global_metrics(farp_dict):
    """
    Finds global metrics for model performance across difficulty classes

    Input(s):
    - farp_dict: (dict)

    Output(s):
    - local_stats: (dict of dicts)
    - global_stats: (dict)
    """
    running_f = []
    running_a = []
    running_r = []
    running_p = []
    running_auc = []

    for v_class, v_dict in farp_dict.items():
        index = find_index_balanced_rp(v_dict['r_scores'], v_dict['p_scores'])

        running_f.append(v_dict['f_scores'][index])
        running_a.append(v_dict['a_scores'][index])
        running_r.append(v_dict['r_scores'][index])
        running_p.append(v_dict['p_scores'][index])
        running_auc.append(v_dict['auc_score'])

    # Global metrics aggregations
    global_stats = {
        'F1': np.nanmean(running_f),
        'Accuracy': np.nanmean(running_a),
        'Recall': np.nanmean(running_r),
        'Precision': np.nanmean(running_p),
        'AUC': np.nanmean(running_auc)
    }

    # Local metrics
    local_stats = dict()
    for i, v_class in enumerate(farp_dict.keys()):
        tmp_metrics = {
            'F1': running_f[i],
            'Accuracy': running_a[i],
            'Recall': running_r[i],
            'Precision': running_p[i],
            'AUC': running_auc[i]
        }
        local_stats[v_class] = tmp_metrics

    return local_stats, global_stats


# ----------------------------------------------------------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title, ax, window_size=0):
    """
    Plots confusion matrix between true and predicted labels

    Input(s):
    - y_true: (numpy array) True difficulty label shape [m x num_classes], one-hot vector
    - y_pred: (numpy array) Predicted difficulty labels shape [m x num_classes], vector of probabilities
    - title: (string) Title of correlation plot
    - ax: Matplotlib axis object
    - window_size: (int) Size of sliding window
    """
    agg_true, agg_pred = argmax_with_window(y_true, y_pred, window_size=window_size)

    # Get confusion matrix
    confusion = confusion_matrix(agg_true, agg_pred)

    # Cast as pandas dataframe
    v_grades = ['V%s' % (i+4) for i in range(y_true.shape[1])]
    confusion_df = pd.DataFrame(confusion, index=v_grades, columns=v_grades)

    # Plot confusion matrix
    sn.heatmap(confusion_df, annot=True, fmt='d', ax=ax)
    ax.set_ylabel('True')
    ax.set_xlabel('Pred')
    ax.set_title(title)

    return None


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation wrappers
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_correlation(y_true, y_pred, title, fig_save_path):
    """
    Correlation plot wrapper
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.flatten()

    plot_label_correlation(y_true, y_pred, 'Window Size 0', ax[0], window_size=0)
    plot_label_correlation(y_true, y_pred, 'Window Size 1', ax[1], window_size=1)
    fig.suptitle(title)

    plt.savefig(fig_save_path)
    plt.clf()
    plt.close()
    return None


def evaluate_farp(y_true, y_pred, title, fig_save_path):
    """
    FARP plot wrapper
    """
    _, n = y_true.shape
    fig_dim = 8

    fig, ax = plt.subplots(n, 2, figsize=(2*(fig_dim+1), n*fig_dim))
    ax = ax.flatten()

    # Get FARP curves
    farp_dict_s0 = get_farp_scores(y_true, y_pred, window_size=0)
    farp_dict_s1 = get_farp_scores(y_true, y_pred, window_size=1)

    v_grades = ['V%s' % (i+4) for i in range(n)]

    # Plot FARP curves
    cnt = 0
    for v_grade in v_grades:
        plot_single_farp_curve(farp_dict_s0[v_grade], ax[cnt], '%s Window Size 0' % v_grade)
        cnt += 1
        plot_single_farp_curve(farp_dict_s1[v_grade], ax[cnt], '%s Window Size 1' % v_grade)
        cnt += 1

    fig.suptitle(title)
    plt.savefig(fig_save_path)
    plt.clf()
    plt.close()

    return farp_dict_s0, farp_dict_s1


def evaluate_confusion(y_true, y_pred, title, fig_save_path):
    """
    Confusion plot wrapper
    """
    fig_dim = 12

    fig, ax = plt.subplots(1, 2, figsize=(fig_dim*2, fig_dim-2))
    ax = ax.flatten()

    plot_confusion_matrix(y_true, y_pred, 'Window Size 0', ax[0], window_size=0)
    plot_confusion_matrix(y_true, y_pred, 'Window Size 1', ax[1], window_size=1)
    fig.suptitle(title)

    plt.savefig(fig_save_path)
    plt.clf()
    plt.close()

    return None


def evaluate_global(farp_dict_s0, farp_dict_s1):
    """
    Global statistics wrapper
    """
    local_s0, global_s0 = get_global_metrics(farp_dict_s0)
    local_s1, global_s1 = get_global_metrics(farp_dict_s1)

    global_stats = {
        'Window 0 Local': local_s0,
        'Window 0 Global': global_s0,
        'Window 1 Local': local_s1,
        'Window 1 Global': global_s1
    }
    return global_stats


# ----------------------------------------------------------------------------------------------------------------------
# Main evaluation function
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Main evaluation function
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, exp_settings):
    """
    Performs model evaluation

    Input(s):
    - y_true: (numpy array) True difficulty label shape [m x num_classes], one-hot vector
    - y_pred: (numpy array) Predicted difficulty labels shape [m x num_classes], vector of probabilities
    - exp_settings: (dict) Dictionary of experiment settings
    """
    description = exp_settings['description']

    # Correlation
    evaluate_correlation(y_true, y_pred, description, exp_settings['corr_fig_save'])

    # FARP curves
    fd0, fd1 = evaluate_farp(y_true, y_pred, description, exp_settings['farp_fig_save'])
    farp_stats = {
        'Window 0': fd0,
        'Window 1': fd1
    }
    save_pickle(farp_stats, exp_settings['farp_stats_save'])

    # Confusion matrices
    evaluate_confusion(y_true, y_pred, description, exp_settings['confusion_fig_save'])

    # Global stats
    global_stats = evaluate_global(fd0, fd1)
    save_pickle(global_stats, exp_settings['global_stats_save'])

    return None
