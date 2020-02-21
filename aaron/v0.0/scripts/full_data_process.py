import os
import numpy as np
import pickle

from nodeMapping import nodeMapping
from nodeAdjacency import nodeAdjacency
from tfidfHolds import tfidfHolds
from utils import *

class graphDataProcess:
    '''
    A graphDataProcess object contains: 
    1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
    2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
    3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
    '''
    def __init__(self, raw_data_path='', save_data_dir='', names_dict={}, redo_dict={}):
        self.raw_data_path = raw_data_path
        self.save_data_dir = save_data_dir
        self.names_dict = names_dict
        self.redo_dict = redo_dict
        
        self.nodeMapping_obj = None
        self.nodeAdjacency_obj = None
        self.tfidf_obj = None
        
        
        #-------------------------------------------------------------
        # Default Initializations
        #-------------------------------------------------------------
        self.default_raw = '/home/ds-team/aaron/other/MoonBoard/data/processed_data/moonboard_data.pickle'
        self.default_save = '/home/ds-team/aaron/other/MoonBoard/data/train_test/pytorch/graphNet/'
        self.default_names = {'node_map_name':'node_mapping.pickle',
                              'holds_names_name':'holds_names.pickle',
                              'problems_names_name':'problems_names.pickle',
                              'holds_mat_name':'holds_mat.pickle',
                              'pmi_name':'pmi.pickle',
                              'tfidf_name':'tfidf.pickle'}
        self.default_redo = {'mapping_redo':False,
                             'adjacency_redo':False,
                             'tfidf_redo':False}
        #-------------------------------------------------------------
        
        self.set_default_paths()
        self.set_default_params()
        return
    
    def set_default_paths(self):
        self.raw_data_path = set_default(self.raw_data_path, '', self.default_raw)
        self.save_data_dir = set_default(self.save_data_dir, '', self.default_save)
        return
    
    def set_default_params(self):
        for name in self.default_names:
            set_default_dict(self.names_dict, name, '', self.default_names[name])
        for redo in self.default_redo:
            set_default_dict(self.redo_dict, redo, '', self.default_redo[redo])
        return
    
    def get_node_mapping(self):
        print('Mapping nodes')
        node_map_path = self.save_data_dir + self.names_dict['node_map_name']
        remove_redo_paths(self.redo_dict['mapping_redo'], [node_map_path])
        self.nodeMapping_obj = nodeMapping(self.raw_data_path, node_map_path)
        self.nodeMapping_obj.get_map()
        return

    def get_adj_mapping(self):
        print('Mapping adjacency')
        if self.nodeMapping_obj==None:
            self.get_node_mapping()
        nodes_map = self.nodeMapping_obj.maps_dict
        holds_names_path = self.save_data_dir + self.names_dict['holds_names_name']
        problems_names_path = self.save_data_dir + self.names_dict['problems_names_name']
        holds_mat_path = self.save_data_dir + self.names_dict['holds_mat_name']
        pmi_path = self.save_data_dir + self.names_dict['pmi_name']
        remove_redo_paths(self.redo_dict['adjacency_redo'], [holds_names_path, problems_names_path, holds_mat_path, pmi_path])
        self.nodeAdjacency_obj = nodeAdjacency(nodes_map, holds_names_path, problems_names_path, holds_mat_path, pmi_path)
        self.nodeAdjacency_obj.load_pp_pmi()
        return

    def get_tfidf(self):
        print('Training TFIDF')
        if self.nodeAdjacency_obj==None:
            self.get_adj_mapping()
        tfidf_path = self.save_data_dir + self.names_dict['tfidf_name']
        holds_names = self.nodeAdjacency_obj.holds_names
        holds_mat = self.nodeAdjacency_obj.holds_mat
        problems_names = self.nodeAdjacency_obj.problems_names
        remove_redo_paths(self.redo_dict['tfidf_redo'], [tfidf_path])

        if os.path.exists(tfidf_path):
            self.tfidf_obj = pickle.load(open(tfidf_path,'rb'))
        else:
            self.tfidf_obj = tfidfHolds(holds_names, holds_mat, problems_names)
            self.tfidf_obj.get_tfidf_dict()
            pickle.dump(self.tfidf_obj, open(tfidf_path,'wb'))
        return
    
    def run_all(self):
        self.get_node_mapping()
        self.get_adj_mapping()
        self.get_tfidf()
        return

