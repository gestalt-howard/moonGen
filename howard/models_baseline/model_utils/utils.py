# Author: Howard Tai

# Script containing general utility functions

import os
import h5py
import pickle


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


# ----------------------------------------------------------------------------------------------------------------------
# Data loading functions
# ----------------------------------------------------------------------------------------------------------------------
def load_trn_tst_dicts(root_path):
    """
    Loads dictionaries containing train and test information, given a root path
    """
    trn_dict = load_pickle(root_path + 'trn_data.pickle')
    tst_dict = load_pickle(root_path + 'tst_data.pickle')

    return trn_dict, tst_dict
