# Authors: Aaron Wu / Howard Tai

# This script contains code defining a class that creates the adjacency relationship amongst holds and problems in a
# MoonBoard-specific problem graph

import os
import pdb
import numpy as np

from scripts.pytorch.utils.utils import *


class NodeAdjacency:
    """
    Object that calculates and contains PMI adjacency information
    """

    def __init__(self, maps_dict=None, holds_names_path='', problems_names_path='', holds_mat_path='', pmi_path=''):
        """
        Constructor for PMI adjacency-defining object

        Required:
        - maps_dict (dict): nodeMapping object's self.map_dict

        Optional: (paths to intermediate files)
        - holds_names_path (string)
        - problems_names_path (string)
        - holds_mat_path (string)
        - pmi_path (string)
        """
        # Intermediate files
        self.problems_names_path = problems_names_path
        self.holds_names_path = holds_names_path
        self.holds_mat_path = holds_mat_path
        self.pmi_path = pmi_path

        # Load maps dict
        self.maps_dict = maps_dict

        # Instantiate PMI-related attributes
        self.holds_names = None
        self.problems_names = None
        self.holds_mat = None
        self.hold_probs = None
        self.joint_hold_probs = None
        self.pmi_mat = None

    def load_holds_names(self):
        """
        Retrieves all relevant hold names (i.e. actually a part of a specific MoonBoard configuration); not all pegs on
        the MoonBoard canvas are populated by holds
        """
        if os.path.exists(self.holds_names_path):
            self.holds_names = load_pickle(self.holds_names_path)
        else:
            self.get_holds_names()
            if self.holds_names_path != '':
                save_pickle(self.holds_names, self.holds_names_path)
        return None

    def load_problems_names(self):
        """
        Retrieves all problem names as global node IDs
        """
        if os.path.exists(self.problems_names_path):
            self.problems_names = load_pickle(self.problems_names_path)
        else:
            self.get_problems_names()
            if self.problems_names_path != '':
                save_pickle(self.problems_names, self.problems_names_path)
        return None

    def load_holds_mat(self):
        """
        Serves 2 purposes:
        1. Wrapper for load_hold_names() and load_problems_names()
        2. Retrieves multi-hot representation of problems
        """
        if self.holds_names is None:  # Load global hold IDs
            self.load_holds_names()
        if self.problems_names is None:  # Load global problem IDs
            self.load_problems_names()

        if os.path.exists(self.holds_mat_path):
            self.holds_mat = load_pickle(self.holds_mat_path)
        else:
            self.make_problem_matrix()
            if self.holds_mat_path != '':
                save_pickle(self.holds_mat, self.holds_mat_path)
        return None

    def load_pmi(self):
        """
        Retrieves PMI calculations
        """
        if self.holds_mat is None:  # Load multi-hot problems matrix
            self.load_holds_mat()

        if os.path.exists(self.pmi_path):
            self.pmi_mat = load_pickle(self.pmi_path)
        else:
            self.calc_hold_probs()
            self.calc_joint_hold_probs()
            self.calc_pmi()

            if self.pmi_path != '':
                save_pickle(self.pmi_mat, self.pmi_path)
        return None

    def get_holds_names(self):
        """
        Find all the holds that are used in MoonBoard problems (holds represented as global node IDs)
        """
        problems = sorted(list(self.maps_dict['prob_hold_map'].keys()))

        all_holds = []
        for problem in problems:
            all_holds += self.maps_dict['prob_hold_map'][problem]
        self.holds_names = sorted(list(set(all_holds)))
        return None

    def get_problems_names(self):
        """
        Collects all MoonBoard problems as global node IDs
        """
        nodes_dict = self.maps_dict['nodes_map']
        self.problems_names = [nodes_dict[n] for n in nodes_dict if 'p' in n]
        return None

    def make_problem_matrix(self):
        """
        Creates a problem matrix, where each row is multi-hot vector with 1 / 0 indicating presence / absence of a
        specific hold

        Rows: problems
        Cols: hold indexes

        Overall shape: [num_problems, num_holds]
        """
        # Initialize placeholder
        self.holds_mat = np.zeros((len(self.problems_names), len(self.holds_names)))

        # Loop over problems
        for i, p_name in enumerate(self.problems_names):
            # Fill in hold indexes for specific problem
            for h_name in self.maps_dict['prob_hold_map'][p_name]:
                self.holds_mat[i][self.holds_names.index(h_name)] = 1
        return None

    def calc_hold_probs(self):
        """
        Calculates hold-wise probabilities

        Intuitively: If choosing a problem at random, what's the probability of seeing a specific hold?
        """
        self.hold_probs = np.sum(self.holds_mat, axis=0) / float(self.holds_mat.shape[0])
        return None

    def calc_joint_hold_probs(self):
        """
        Calculate joint probability amongst holds, shape [num_holds, num_holds]
        """
        # Initialize placeholder
        joint_hold_probs = np.zeros((self.holds_mat.shape[1], self.holds_mat.shape[1]))

        # Calculate global co-occurrence counts
        for holds in self.holds_mat:
            joint_hold_probs = joint_hold_probs + np.outer(holds, holds)

        # Calculate co-occurrence probabilities
        self.joint_hold_probs = joint_hold_probs / float(self.holds_mat.shape[0])
        return None

    def calc_pmi(self):
        """
        Calculate PMI as follows:

        - Calculate independence condition with outer product
        - Calculate PMI
        - Perform ReLU clipping

        Note that PMI implementation defines window as a single problem
        """
        outer_hold_probs = np.outer(self.hold_probs, self.hold_probs)
        self.pmi_mat = np.log2(np.divide(self.joint_hold_probs + .0001, outer_hold_probs + .0001))
        self.pmi_mat = np.clip(self.pmi_mat, a_min=0, a_max=None)
        return None
