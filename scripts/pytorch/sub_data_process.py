# Authors: Aaron Wu / Howard Tai

import os
import pickle

import numpy as np

from scripts.pytorch.utils.label_functions import *
from scripts.pytorch.utils.feature_functions import *
from scripts.pytorch.utils.adjacency_functions import *

from scripts.pytorch.utils.utils import *


class SubGraphProcess:
    """
    A subGraphProcess object contains:
    1. Mapping of the problem nodes to their respective indexes in features/adjacency/labels (core_nodes_id_dict)
    2. Mapping of the hold nodes to their respective indexes in features/adjacency/labels (hold_nodes_id_dict)
    3. Generated matrix of features for sub-sampled problem nodes + hold nodes (features)
    4. Generated adjacency matrix for sub-sampled problem nodes + hold nodes (adjacency)
    5. Generated list of labels for sub-sampled problem nodes + hold nodes (labels)
    """

    def __init__(
            self,
            full_processed_path='',
            save_data_dir='',
            names_dict=None,
            redo_dict=None,
            functions_dict=None,
            sampling_params=None
    ):
        # Setting default dictionary inputs
        if sampling_params is None:
            sampling_params = {}
        if functions_dict is None:
            functions_dict = {}
        if redo_dict is None:
            redo_dict = {}
        if names_dict is None:
            names_dict = {}

        # File paths
        self.full_processed_path = full_processed_path
        self.save_data_dir = save_data_dir
        self.raw_data_path = ''

        # Define settings as constructor inputs
        self.names_dict = names_dict
        self.redo_dict = redo_dict
        self.functions_dict = functions_dict
        self.sampling_params = sampling_params

        # Initializations
        self.full_processed = None
        self.core_nodes = None
        self.hold_nodes = None
        self.nodes = None
        self.features = None
        self.adjacency = None
        self.labels = None
        self.problem_idxs = None

        self.core_nodes_id_dict = dict()
        self.hold_nodes_id_dict = dict()

        # Default settings
        self.default_full_processed = '/home/ds-team/aaron/other/MoonBoard/data/train_test/pytorch/graphNet' \
                                      '/full_processed.pickle '
        self.default_save = '/home/ds-team/aaron/other/MoonBoard/data/train_test/pytorch/graphNet/'
        self.default_names = {
            'core_nodes_name': 'core_nodes.pickle',
            'features_name': 'sampled_features.pickle',
            'adjacency_name': 'sampled_adjacency.pickle',
            'labels_name': 'sampled_labels.pickle',
            'problem_idxs_name': 'sampled_problem_idxs.pickle'
        }
        self.default_redo = {
            'core_nodes_redo': False,
            'feature_redo': False,
            'adjacency_redo': False,
            'label_redo': False
        }
        self.default_func = {  # Processing function names
            'feature': gen_onehotfeatures,
            'adjacency': gen_adjacency,
            'label': gen_labels_idxs,
            'sampling': sample_nodes_balanced
        }
        self.default_sampling_params = {
            'num_per_core': 100,               # Number of samples for each class
            'target_grades': list(range(4, 15)),  # Range of difficulty classes
            'sample_nodes_path': self.save_data_dir + '/' + self.default_names['core_nodes_name']
        }

        # Set defaults
        self.set_default_paths()
        self.set_default_params()

    def set_default_paths(self):
        """
        Initializes file paths to default if given paths are empty
        """
        self.raw_data_path = set_default(self.full_processed_path, '', self.default_full_processed)
        self.save_data_dir = set_default(self.save_data_dir, '', self.default_save)
        return None

    def set_default_params(self):
        """
        Initializes settings (dictionary values) to default if given settings are empty
        """
        for name in self.default_names:
            self.names_dict[name] = set_default_dict(self.names_dict, name, '', self.default_names[name])
        for redo in self.default_redo:
            self.redo_dict[redo] = set_default_dict(self.redo_dict, redo, '', self.default_redo[redo])
        for func in self.default_func:
            self.functions_dict[func] = set_default_dict(self.functions_dict, func, None, self.default_func[func])

        # Default sampling parameters
        for param in self.default_sampling_params:
            self.sampling_params[param] = set_default_dict(
                self.sampling_params,
                param,
                None,
                self.default_sampling_params[param]
            )
        return None

    def load_full_processed(self):
        """
        Retrieves processed data
        """
        self.full_processed = load_pickle(self.full_processed_path)
        return

    def get_core_nodes(self):
        """
        Sample a subset of problem nodes from each difficulty class
        """
        print('\nSampling core nodes...')

        # Get save path
        core_nodes_path = self.save_data_dir + self.names_dict['core_nodes_name']

        # Define difficulty map
        prob_grade_map = self.full_processed.nodeMapping_obj.maps_dict['prob_grade_map']

        # Sample core nodes and save
        self.core_nodes = self.functions_dict['sampling'](prob_grade_map, self.sampling_params)
        save_pickle(self.core_nodes, core_nodes_path)
        return None

    def get_hold_nodes(self):
        """
        Parses hold names from a node adjacency object where hold names are represented as global node IDs
        """
        self.hold_nodes = self.full_processed.nodeAdjacency_obj.holds_names
        return None

    def set_nodes(self):
        """
        After obtaining a subset of problem nodes, assign a unique, incrementing integer identifier for both sampled
        problem nodes and hold nodes

        Creates:
        - core_nodes_id_dict: key (global ID), value (feature / adjacency / label ID)
        - hold_nodes_id_dict: key (global ID), value (feature / adjacency / label ID)
        """
        # Total nodes = {subset of problem nodes} + {set of hold nodes}
        self.nodes = self.core_nodes + self.hold_nodes

        # Give each problem (core) node a unique incrementing identifier
        for i in range(len(self.core_nodes)):
            if self.core_nodes[i] in self.core_nodes_id_dict:
                self.core_nodes_id_dict[self.core_nodes[i]].append(i)
            else:
                self.core_nodes_id_dict[self.core_nodes[i]] = [i]

        # Give each hold node a unique incrementing identifier, starting from last problem node
        for i in range(len(self.hold_nodes)):
            self.hold_nodes_id_dict[self.hold_nodes[i]] = i + len(self.core_nodes)
        return None

    def process_nodes(self, name_key, redo_key, func_key):
        """
        Method for processing nodes using a specified function
        """
        # Define data save path, clear memory
        save_path = self.save_data_dir + self.names_dict[name_key]
        remove_redo_paths(self.redo_dict[redo_key], [save_path])

        # If processed data already exists
        if os.path.exists(save_path):
            return load_pickle(save_path)
        else:
            # Processes data using a specific processing function
            output = self.functions_dict[func_key](self.full_processed, self.nodes)
            save_pickle(output, save_path)
        return output

    def get_features(self):
        """
        Gets node features
        """
        print('\nGetting samples node features...')
        self.features = self.process_nodes('features_name', 'feature_redo', 'feature')
        return None

    def get_adjacency(self):
        """
        Gets adjacency matrix
        """
        print('\nGetting samples node adjacency...')
        self.adjacency = self.process_nodes('adjacency_name', 'adjacency_redo', 'adjacency')
        return None

    def get_labels(self):
        """
        Gets labels for nodes
        """
        print('\nGetting samples node labels...')
        self.labels = self.process_nodes('labels_name', 'label_redo', 'label')
        return None

    def run_all(self):
        """
        Executes all processing methods
        """
        self.load_full_processed()
        self.get_hold_nodes()
        self.get_core_nodes()
        self.set_nodes()

        # Process features
        self.get_features()
        self.get_adjacency()
        self.get_labels()
        return None
