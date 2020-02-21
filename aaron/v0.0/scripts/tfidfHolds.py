import numpy as np
import pickle
import os

class tfidfHolds:
    '''
    Object that contains dictionaries of problems that are mapped to holds.
    Each mapped hold contains the tfidf value of that hold with respect to the parent problem.
    '''
    def __init__(self, holds_names=None, holds_mat=None, problems_names=None):
        self.idf=None
        self.counts=None
        self.fitted=False
        self.fit_data=None
        self.holds_names=holds_names
        self.holds_mat=holds_mat
        self.problems_names = problems_names
        self.tfidf_dict={}
        return

    def set_holds_names(self, holds_names):
        self.holds_names = holds_names
        return

    def set_holds_mat(self, holds_mat):
        self.holds_mat = holds_mat
        return

    def calc_counts(self):
        self.counts = np.sum(self.holds_mat, axis=0)
        return

    def calc_idf(self):
        self.idf = np.log2(self.holds_mat.shape[0]/(np.sum(self.holds_mat, axis=0)+1))
        return

    def fit(self):
        self.calc_counts()
        self.calc_idf()
        self.fitted=True
        return

    def transform(self, new_holds_mat):
        if not self.fitted:
            print('Need to fit to some data')
            return
        tf = new_holds_mat
        if len(new_holds_mat.shape)==2:
            tiled_idf = np.tile(self.idf,(new_holds_mat.shape[0],1))
            tfidf = np.multiply(tf, tiled_idf)
        else:
            tfidf = np.multiply(tf, self.idf)
        return tfidf
    
    def fit_transform(self):
        self.fit()
        self.fit_data = self.transform(self.holds_mat)
        return
    
    def get_tfidf_dict(self):
        self.fit_transform()
        assert len(self.problems_names)==self.fit_data.shape[0]
        for i,pname in enumerate(self.problems_names):
            self.tfidf_dict[pname] = {}
            for j,hname in enumerate(self.holds_names):
                self.tfidf_dict[pname][hname] = self.fit_data[i][j]
        return