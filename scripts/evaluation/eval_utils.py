# Author: Howard Tai

# Script containing general utility functions

import os
import h5py
import pickle

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# General helper functions
# ----------------------------------------------------------------------------------------------------------------------
def path_exists(path):
    """
    True if path exists, False else
    """
    if os.path.exists(path):
        return True
    return False


def make_directory(path):
    """
    Makes a directory if path doesn't exist
    """
    if not path_exists(path):
        os.mkdir(path)
    return None


def save_h5(data, fname):
    """
    Saves data as h5py format
    """
    with h5py.File(fname, 'w') as h5f:
        h5f.create_dataset('data', data=data)
    return None


def load_h5(fname):
    """
    Loads data from h5py format
    """
    with h5py.File(fname, 'r') as h5f:
        return h5f['data'][:]


def maybe_h5(data, fname):
    """
    Saves h5 dataset if it doesn't already exist
    """
    if not path_exists(fname):
        save_h5(data, fname)
    return None


def save_pickle(data, fname):
    """
    Saves data as pickle format
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    return None


def load_pickle(fname):
    """
    Loads data from pickle format
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)


def maybe_pickle(data, fname):
    """
    Saves pickle dataset if it doesn't already exist
    """
    if not path_exists(fname):
        save_pickle(data, fname)
    return None


def print_header(text):
    """
    Prints header block
    """
    print('\n' + '-'*40 + '\n' + text + '\n' + '-'*40)
    return None


def onehot_labels(labels, num_labels):
    """
    Casts flat labels as one-hot vectors
    """
    onehot = np.zeros((len(labels), num_labels))
    for i, l in enumerate(labels):
        onehot[i][l] = 1
    return onehot


# ----------------------------------------------------------------------------------------------------------------------
# Function defining difficulty map
# ----------------------------------------------------------------------------------------------------------------------
def get_difficulty_map():
    """
    Defines difficulty maps for conversions, relative to Aaron's grade_map.pickle
    """
    map_dict = {
        2: {'font_scale': '6B', 'v_scale': 4},
        3: {'font_scale': '6B+', 'v_scale': 4},
        4: {'font_scale': '6C', 'v_scale': 5},
        5: {'font_scale': '6C+', 'v_scale': 5},
        6: {'font_scale': '7A', 'v_scale': 6},
        7: {'font_scale': '7A+', 'v_scale': 7},
        8: {'font_scale': '7B', 'v_scale': 8},
        9: {'font_scale': '7B+', 'v_scale': 8},
        10: {'font_scale': '7C', 'v_scale': 9},
        11: {'font_scale': '7C+', 'v_scale': 10},
        12: {'font_scale': '8A', 'v_scale': 11},
        13: {'font_scale': '8A+', 'v_scale': 12},
        14: {'font_scale': '8B', 'v_scale': 13},
        15: {'font_scale': '8B+', 'v_scale': 14}
    }
    return map_dict
