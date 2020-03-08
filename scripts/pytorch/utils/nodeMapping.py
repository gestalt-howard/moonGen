# Authors: Aaron Wu / Howard Tai

# Script containing code defining a class for pre-processing data for representation as graph nodes

import os
import pickle
import numpy as np

from scripts.pytorch.utils.utils import *


class NodeMapping:
    """
    Object that defines various mappings from the original raw data to a form usable by other functions
    """

    def __init__(self, data_path='', maps_path='', num_rows=18, num_cols=11, vmap=True):
        # Initialize data paths
        self.data_path = data_path
        self.maps_path = maps_path
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.vmap = vmap

        if vmap:  # Defines Fontainebleau to V-Scale difficulty mapping
            self.vmap_dict = {
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

        # Initialize class attributes
        self.data = None
        self.holds = None
        self.problems = None
        self.holds_map = None
        self.problems_map = None
        self.nodes_map = None
        self.problem_holds_map = None
        self.problem_grade_map = None
        self.holds_idxs_map = None
        self.maps_dict = None

    def load_data(self):
        """
        Loads original mined data (should be a dictionary) from pickle format
        """
        self.data = load_pickle(self.data_path)
        return None

    def get_holds_set(self):
        """
        Initializes hold coordinates as x, y (Cartesian) pairs
        """
        self.holds = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.holds.append([j, i])
        return None

    def get_problems_set(self):
        """
        Unpacks original (mined) problems data into a list of problem dictionaries consisting of:

        - Problem URL (string)
        - Problem holds (list of tuples)
        - Problem difficulty (int)
        """
        self.problems = []
        data_keys = sorted(list(self.data.keys()))

        for key in data_keys:
            problem = dict()
            problem['url'] = self.data[key]['url']
            problem['holds'] = self.data[key]['start'] + self.data[key]['mid'] + self.data[key]['end']
            problem['grade'] = self.data[key]['grade']

            # Cast difficulty (grade) to V-Scale
            if self.vmap:
                problem['grade'] = self.vmap_dict[problem['grade']]['v_scale']
            self.problems.append(problem)
        return None

    def map_holds(self):
        """
        Assign each Cartesian coordinate pair a unique hold ID (indexed @ 0)
        """
        self.holds_map = dict()
        for i, hold in enumerate(self.holds):
            self.holds_map['%s_%s' % (hold[0], hold[1])] = 'h%s' % i
        return None

    def map_problems(self):
        """
        Assign each problem URL a unique ID (indexed @ 0)
        """
        self.problems_map = dict()
        for i, problem in enumerate(self.problems):
            self.problems_map[problem['url']] = 'p%s' % i
        return None

    def map_nodes(self):
        """
        Assigns each unique hold ID and unique problem ID to a unique (global) node ID
        """
        self.nodes_map = dict()
        current_i = 0

        # Map holds to global ID
        for key in sorted(list(self.holds_map.keys())):
            self.nodes_map[self.holds_map[key]] = 'n%s' % current_i
            current_i += 1

        # Map problems to global ID
        for key in sorted(list(self.problems_map.keys())):
            self.nodes_map[self.problems_map[key]] = 'n%s' % current_i
            current_i += 1

        return None

    def problems2nodes(self):
        """
        Assigns to each problem its respective set of holds in terms of unique global node IDs
        """
        self.problem_holds_map = dict()

        # Loop through all problems
        for i, problem in enumerate(self.problems):
            problem_id = self.problems_map[problem['url']]  # Get problem ID
            problem_node_id = self.nodes_map[problem_id]    # Get problem's global node ID

            # Instantiate empty list for problem's holds
            holds_list = []

            # Iterate through problem holds (Cartesian)
            for hold in problem['holds']:
                hold_id = self.holds_map['%s_%s' % (hold[0], hold[1])]
                holds_list.append(self.nodes_map[hold_id])

            # Add problem
            self.problem_holds_map[problem_node_id] = holds_list
        return None

    def grade_problems(self):
        """
        Assigns a difficulty grade to each MoonBoard problem (identified via problem's global node ID)
        """
        self.problem_grade_map = dict()

        for i, problem in enumerate(self.problems):
            problem_id = self.problems_map[problem['url']]
            problem_node_id = self.nodes_map[problem_id]
            self.problem_grade_map[problem_node_id] = problem['grade']
        return None

    def map_holds_idxs(self):
        """
        Flatten Cartesian coordinates and map to global node ID for each node

        Example conversion:
        - With 11 columns and 18 rows and origin @ lower-left (0, 0)
        - A hold at coordinate (3, 7): 4th row, 8th column
        - Coordinate (3, 7) maps to: 3 * 11 + 7 = index 40 out of 198 (11 * 18)
        """
        self.holds_idxs_map = dict()

        for i, hold in enumerate(self.holds):
            hold_id = self.holds_map['%s_%s' % (hold[0], hold[1])]  # Get local ID
            hold_node_id = self.nodes_map[hold_id]                  # Get global ID
            self.holds_idxs_map[hold_node_id] = hold[0] * self.num_cols + hold[1]
        return None

    def map_all(self):
        """
        Executes every mapping protocol (method) and wraps everything into a dictionary
        """
        self.map_holds()
        self.map_problems()
        self.map_nodes()
        self.problems2nodes()
        self.grade_problems()
        self.map_holds_idxs()

        # Wraps into dictionary
        self.maps_dict = dict()
        self.maps_dict['holds_map'] = self.holds_map
        self.maps_dict['holds_idxs_map'] = self.holds_idxs_map
        self.maps_dict['problems_map'] = self.problems_map
        self.maps_dict['nodes_map'] = self.nodes_map
        self.maps_dict['prob_hold_map'] = self.problem_holds_map
        self.maps_dict['prob_grade_map'] = self.problem_grade_map
        return None

    def get_map(self):
        """
        Wrapper for map_all() method with logic for loading pre-computed results
        """
        if os.path.exists(self.maps_path):
            self.maps_dict = load_pickle(self.maps_path)
        else:
            self.load_data()         # Load mined data
            self.get_holds_set()     # Load hold set as Cartesian pairs
            self.get_problems_set()  # Load problem set from mined data
            self.map_all()           # Get master map dictionary

            if not self.maps_path == '':
                save_pickle(self.maps_dict, self.maps_path)
        return None
