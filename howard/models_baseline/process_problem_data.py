# Author: Howard Tai

# Script for processing MoonBoard problem data from scraped version

import os
import pandas as pd
import numpy as np

from pprint import pprint

from sklearn.preprocessing import OneHotEncoder

from model_utils.utils import path_exists
from model_utils.utils import load_pickle, save_pickle, maybe_pickle
from model_utils.utils import load_h5, save_h5, maybe_h5


def cast_difficulty_vscale(labels):
    """
    Maps difficulty to V-scale
    """
    map_dict = {  # References scale defined by Aaron's grade_map.pickle
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
    v_labels = []
    for label in labels:
        v_labels.append(map_dict[label]['v_scale'])

    return np.asarray(v_labels)


def holds_to_grid(single_problem_dict):
    """
    Converts holds coordinate to a grid
    """
    num_cols = 11
    num_rows = 18
    holds_grid = np.zeros((num_rows, num_cols))

    holds_tags = ['start', 'mid', 'end']
    holds_list = []

    for tag in holds_tags:
        holds_list += single_problem_dict.get(tag, [])

    for coordinate in holds_list:
        x, y = coordinate
        y = num_rows - y - 1
        holds_grid[y, x] = 1

    return holds_grid


def flatten_holds_grid(holds_grid):
    """
    Flattens 2D grid to 1D vector
    """
    return holds_grid.reshape(1, -1)


def get_train_test_splits(uids, data, v_labels, v_array, num_test=1000):
    """
    Splits data into train and test splits
    """
    uids_len = uids.shape[0]
    data_len = data.shape[0]
    label_len = v_labels.shape[0]
    array_len = v_array.shape[0]

    assert(uids_len == data_len)
    assert(uids_len == label_len)
    assert(uids_len == array_len)

    shuffled_idx = np.random.permutation(uids_len)
    trn_idx = shuffled_idx[num_test:]
    tst_idx = shuffled_idx[:num_test]

    trn_dict = {'uids': uids[trn_idx], 'labels': v_labels[trn_idx], 'array': v_array[trn_idx], 'data': data[trn_idx]}
    tst_dict = {'uids': uids[tst_idx], 'labels': v_labels[tst_idx], 'array': v_array[tst_idx], 'data': data[tst_idx]}

    return trn_dict, tst_dict


def create_datasets(data_dict, trn_data_save, tst_data_save):
    """
    Creates data-set from scraping results
    """
    uid_list = sorted(list(data_dict.keys()))

    labels_list = []
    problems_list = []

    for uid in uid_list:
        problem_dict = data_dict[uid]
        labels_list.append(problem_dict['grade'])
        problems_list.append(flatten_holds_grid(holds_to_grid(problem_dict)))

    # Cast as numpy objects
    uid_list = np.asarray(uid_list)
    labels_list = np.asarray(labels_list)
    v_labels_list = cast_difficulty_vscale(labels_list)
    x_problems = np.vstack(problems_list)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse=False)
    v_labels_array = encoder.fit_transform(v_labels_list.reshape(-1, 1))
    print('V-Scale Label Categories:')
    print(encoder.categories_)

    # Get train test splits
    trn_dict, tst_dict = get_train_test_splits(uid_list, x_problems, v_labels_list, v_labels_array)

    save_pickle(trn_dict, trn_data_save)
    save_pickle(tst_dict, tst_data_save)

    return None


def main():
    """
    Testing data processing
    """
    data_root = 'C:/Users/chetai/Documents/Projects/data/moonGen/'

    data_path = data_root + 'moonboard_data.pickle'
    data_dict = load_pickle(data_path)

    processed_root = data_root + 'processed_data/'
    trn_data_save = processed_root + 'trn_data.pickle'
    tst_data_save = processed_root + 'tst_data.pickle'

    # Get preprocessed dataset
    create_datasets(data_dict, trn_data_save, tst_data_save)

    return None


if __name__ == '__main__':
    main()
