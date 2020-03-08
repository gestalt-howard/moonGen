# Author(s): Aaron Wu / Howard Tai

# This script defines a class that calculates the TFIDF values of holds within a corpus of MoonBoard problems

import os
import pickle

import numpy as np


class TfidfHolds:
    """
    Object that contains dictionaries of problems that are mapped to holds.
    Each mapped hold contains the tfidf value of that hold with respect to the parent problem.
    """

    def __init__(self, holds_names=None, holds_mat=None, problems_names=None):
        self.idf = None
        self.counts = None
        self.fitted = False
        self.fit_data = None

        self.holds_names = holds_names
        self.holds_mat = holds_mat
        self.problems_names = problems_names

        self.tfidf_dict = dict()

    def set_holds_names(self, holds_names):
        """
        Resets hold names
        """
        self.holds_names = holds_names
        return

    def set_holds_mat(self, holds_mat):
        """
        Resets matrix of problems, shape [num_problems, num_holds]
        """
        self.holds_mat = holds_mat
        return

    def calc_counts(self):
        """
        Computes the frequency of each hold across the entire corpus of problems
        Output shape [num_holds]
        """
        self.counts = np.sum(self.holds_mat, axis=0)
        return None

    def calc_idf(self):
        """
        Performs TFIDF calculation:
        log(v)
        v = number of documents / number of documents containing word
        """
        self.idf = np.log2(self.holds_mat.shape[0] / (np.sum(self.holds_mat, axis=0) + 1))
        return None

    def fit(self):
        """
        Fit TFIDF model
        """
        self.calc_counts()
        self.calc_idf()
        self.fitted = True
        return None

    def transform(self, new_holds_mat):
        """
        Multiplies IDF weights into each problem's holds

        Input(s):
        - new_holds_mat (numpy ndarray): Multi-hot matrix of problems, shape [num_problems, num_holds]
        """
        if not self.fitted:
            print('Need to fit model to some data...')
            return None

        tf = new_holds_mat
        if len(new_holds_mat.shape) == 2:
            tiled_idf = np.tile(self.idf, (new_holds_mat.shape[0], 1))  # Broadcast IDF to all problems
            tfidf = np.multiply(tf, tiled_idf)
        else:
            tfidf = np.multiply(tf, self.idf)
        return tfidf

    def fit_transform(self):
        """
        Wrapper for fit() and transform() methods
        """
        self.fit()
        self.fit_data = self.transform(self.holds_mat)
        return None

    def get_tfidf_dict(self):
        """
        Creates a dictionary mapping each hold in each problem to a specific TFIDF weight. Each problem name (identified
        using a global node identifier) has a dictionary attached. This sub-dictionary's has key-value pairs defined as
        [hold ID, TFIDF score], where hold ID is also in a global node ID representation.
        """
        self.fit_transform()
        assert (len(self.problems_names) == self.fit_data.shape[0])  # Sanity check

        # Assign TFIDF weights to each hold of each problem
        for i, pname in enumerate(self.problems_names):
            self.tfidf_dict[pname] = dict()
            for j, hname in enumerate(self.holds_names):
                self.tfidf_dict[pname][hname] = self.fit_data[i][j]
        return None
