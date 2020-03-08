# Authors: Aaron Wu / Howard Tai

# This script defines a class for processing data into a format suitable for input into a graph neural network. Key
# elements of processing include:
#  1. Defining a node indexing system that assigns global IDs to holds and problems
# 2a. Defining the hold-to-hold PMI adjacency definition
# 2b. Defining the hold-to-problem TFIDF adjacency definition

import os
import pickle

import numpy as np

from scripts.pytorch.utils.tfidfHolds import TfidfHolds
from scripts.pytorch.utils.nodeMapping import NodeMapping
from scripts.pytorch.utils.nodeAdjacency import NodeAdjacency

from scripts.pytorch.utils.utils import *


class GraphDataProcess:
    """
    A GraphDataProcess object contains:
    1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
    2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
    3. The TFIDF model to transform problems into TFIDF mappings (refer to tfidfHolds.py)
    """

    def __init__(self, raw_data_path='', save_data_dir='', names_dict=None, redo_dict=None):
        """
        Input(s):
        - raw_data_path (string): Path to mined data-set
        - save_data_dir (string): Path to directory of saved data files
        - names_dict (dict): Dictionary of
        - redo_dict (dict): Dictionary of flags for re-performing calculations
        """
        if redo_dict is None:
            redo_dict = {}
        if names_dict is None:
            names_dict = {}

        self.raw_data_path = raw_data_path
        self.save_data_dir = save_data_dir
        self.names_dict = names_dict
        self.redo_dict = redo_dict

        # Initializing feature objects
        self.nodeMapping_obj = None
        self.nodeAdjacency_obj = None
        self.tfidf_obj = None

        # Default settings
        self.default_raw = '/home/ds-team/aaron/other/MoonBoard/data/processed_data/moonboard_data.pickle'
        self.default_save = '/home/ds-team/aaron/other/MoonBoard/data/train_test/pytorch/graphNet/'
        self.default_names = {
            'node_map_name': 'node_mapping.pickle',
            'holds_names_name': 'holds_names.pickle',
            'problems_names_name': 'problems_names.pickle',
            'holds_mat_name': 'holds_mat.pickle',
            'pmi_name': 'pmi.pickle',
            'tfidf_name': 'tfidf.pickle'
        }
        self.default_redo = {
            'mapping_redo': False,
            'adjacency_redo': False,
            'tfidf_redo': False
        }

        # Set defaults
        self.set_default_paths()
        self.set_default_params()

    def set_default_paths(self):
        """
        Initializes file paths to default if given paths are empty
        """
        self.raw_data_path = set_default(self.raw_data_path, '', self.default_raw)
        self.save_data_dir = set_default(self.save_data_dir, '', self.default_save)
        return None

    def set_default_params(self):
        """
        Initializes settings (dictionary values) to default if given settings are empty
        """
        for name in self.default_names:
            set_default_dict(self.names_dict, name, '', self.default_names[name])
        for redo in self.default_redo:
            set_default_dict(self.redo_dict, redo, '', self.default_redo[redo])
        return None

    def get_node_mapping(self):
        """
        Processes raw mined MoonBoard data to get node maps
        """
        print('\nMapping nodes...')

        # Set node map save path
        node_map_path = self.save_data_dir + self.names_dict['node_map_name']
        remove_redo_paths(self.redo_dict['mapping_redo'], [node_map_path])

        # Define node mapping object and get maps
        self.nodeMapping_obj = NodeMapping(self.raw_data_path, node_map_path)
        self.nodeMapping_obj.get_map()

        print('Finished mapping nodes!')
        return None

    def get_adj_mapping(self):
        """
        Processes node maps to get PMI adjacency definitions
        """
        print('\nMapping adjacency...')

        # Get node map
        if self.nodeMapping_obj is None:
            self.get_node_mapping()

        # Parse out master dictionary of maps
        nodes_map = self.nodeMapping_obj.maps_dict

        # Define save paths for maps, clears memory
        holds_names_path = self.save_data_dir + self.names_dict['holds_names_name']
        problems_names_path = self.save_data_dir + self.names_dict['problems_names_name']
        holds_mat_path = self.save_data_dir + self.names_dict['holds_mat_name']
        pmi_path = self.save_data_dir + self.names_dict['pmi_name']

        map_paths = [holds_names_path, problems_names_path, holds_mat_path, pmi_path]
        remove_redo_paths(self.redo_dict['adjacency_redo'], map_paths)

        # Define node adjacency object and get PMI matrix
        self.nodeAdjacency_obj = NodeAdjacency(
            nodes_map,
            holds_names_path,
            problems_names_path,
            holds_mat_path,
            pmi_path
        )
        self.nodeAdjacency_obj.load_pmi()

        print('Finished mapping adjacency!')
        return None

    def get_tfidf(self):
        """
        Calculates TFIDF of holds relative to the corpus of problems. The direct analogy to NLP can be made by linking
        holds to words and problems to documents. In the context of MoonBoard problems, TFIDF reduces to IDF because
        term frequency can only take on values of 1 or 0.
        """
        print('\nTraining TFIDF...')

        # Get adjacency map
        if self.nodeAdjacency_obj is None:
            self.get_adj_mapping()

        # Define save path for TFIDF, clear memory
        tfidf_path = self.save_data_dir + self.names_dict['tfidf_name']
        remove_redo_paths(self.redo_dict['tfidf_redo'], [tfidf_path])

        # Unpack node adjacency attributes
        holds_names = self.nodeAdjacency_obj.holds_names
        holds_mat = self.nodeAdjacency_obj.holds_mat
        problems_names = self.nodeAdjacency_obj.problems_names

        # Get TFIDF object
        if os.path.exists(tfidf_path):
            self.tfidf_obj = pickle.load(open(tfidf_path, 'rb'))
        else:
            self.tfidf_obj = TfidfHolds(holds_names, holds_mat, problems_names)
            self.tfidf_obj.get_tfidf_dict()
            save_pickle(self.tfidf_obj, tfidf_path)

        print('Finished training TFIDF!')
        return None

    def run_all(self):
        """
        Executes all feature processing methods
        """
        self.get_node_mapping()
        self.get_adj_mapping()
        self.get_tfidf()
        return None
