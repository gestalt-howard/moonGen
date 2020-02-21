import pickle
import numpy as np
import os
from adjacency_functions import *
from feature_functions import *
from label_functions import *
from utils import *

class subGraphProcess:
    '''
    A subGraphProcess object contains: 
    1. Mapping of the problem nodes to their respective idxs in features/adjacency/labels (core_nodes_id_dict)
    2. Mapping of the hold nodes to their respective idxs in features/adjacency/labels (hold_nodes_id_dict)
    3. Generated matrix of features for subsampled problem nodes + hold nodes (features)
    4. Generated adjacency matrix for subsampled problem nodes + hold nodes (adjacency)
    5. Generated list of labels for subsampled problem nodes + hold nodes (labels) 
    '''
    def __init__(self, full_processed_path='', save_data_dir='', names_dict={}, redo_dict={}, functions_dict={}, sampling_params={}):
        self.full_processed_path = full_processed_path
        self.save_data_dir = save_data_dir
        self.names_dict = names_dict
        self.redo_dict = redo_dict
        self.functions_dict = functions_dict
        self.sampling_params = sampling_params
        
        self.full_processed = None
        self.core_nodes = None
        self.hold_nodes = None
        self.nodes = None
        self.core_nodes_id_dict={}
        self.hold_nodes_id_dict={}
        self.features = None
        self.adjacency = None
        self.labels = None
        self.problem_idxs = None
        
        #-------------------------------------------------------------
        # Default Initializations
        #-------------------------------------------------------------
        self.default_full_processed = '/home/ds-team/aaron/other/MoonBoard/data/train_test/pytorch/graphNet/full_processed.pickle'
        self.default_save = '/home/ds-team/aaron/other/MoonBoard/data/train_test/pytorch/graphNet/'
        self.default_names = {'core_nodes_name':'core_nodes.pickle',
                              'features_name':'sampled_features.pickle',
                              'adjacency_name':'sampled_adjacency.pickle',
                              'labels_name':'sampled_labels.pickle',
                              'problem_idxs_name':'sampled_problem_idxs.pickle'}
        self.default_redo = {'core_nodes_redo':False,
                             'feature_redo':False,
                             'adjacency_redo':False,
                             'label_redo':False}
        self.default_func = {'feature':gen_onehotfeatures,
                             'adjacency':gen_adjacency,
                             'label':gen_labels_idxs,
                             'sampling':sample_nodes_balanced}
        self.default_sampling_params = {'num_per_core':100,
                                        'target_grade':0,
                                        'target_grades':list(range(16)),
                                        'sample_nodes_path':self.save_data_dir + '/' + self.default_names['core_nodes_name'],
                                        'unbalance_multiplier':10}
        #-------------------------------------------------------------
        
        self.set_default_paths()
        self.set_default_params()
        return
    
    def set_default_paths(self):
        self.raw_data_path = set_default(self.full_processed_path, '', self.default_full_processed)
        self.save_data_dir = set_default(self.save_data_dir, '', self.default_save)
        return
    
    def set_default_params(self):
        for name in self.default_names:
            set_default_dict(self.names_dict, name, '', self.default_names[name])
        for redo in self.default_redo:
            set_default_dict(self.redo_dict, redo, '', self.default_redo[redo])
        for func in self.default_func:
            set_default_dict(self.functions_dict, func, None, self.default_func[func])
        for param in self.sampling_params:
            set_default_dict(self.sampling_params, param, None, self.default_sampling_params[param])
        return

    def load_full_processed(self):
        self.full_processed = pickle.load(open(self.full_processed_path,'rb'))
        return
    
    def get_core_nodes(self):
        print('Sampling core nodes')
        core_nodes_path = self.save_data_dir + self.names_dict['core_nodes_name']
        prob_grade_map = self.full_processed.nodeMapping_obj.maps_dict['prob_grade_map']
        self.core_nodes = self.functions_dict['sampling'](prob_grade_map, self.sampling_params)
        pickle.dump(self.core_nodes, open(core_nodes_path,'wb'))
#         remove_redo_paths(self.redo_dict['core_nodes_redo'], [core_nodes_path])
#         if os.path.exists(core_nodes_path):
#             self.core_nodes = pickle.load(open(core_nodes_path,'rb'))
#         else:
#             prob_grade_map = self.full_processed.nodeMapping_obj.maps_dict['prob_grade_map']
#             self.core_nodes = self.functions_dict['sampling'](prob_grade_map, self.sampling_params)
#             pickle.dump(self.core_nodes, open(core_nodes_path,'wb'))
        return
    
    def get_hold_nodes(self):
        self.hold_nodes = self.full_processed.nodeAdjacency_obj.holds_names
        return
    
    def set_nodes(self):
        self.nodes = self.core_nodes + self.hold_nodes
        for i in range(len(self.core_nodes)):
            if self.core_nodes[i] in self.core_nodes_id_dict:
                self.core_nodes_id_dict[self.core_nodes[i]].append(i)
            else:
                self.core_nodes_id_dict[self.core_nodes[i]] = [i]
        for i in range(len(self.hold_nodes)):
            self.hold_nodes_id_dict[self.hold_nodes[i]]=i + len(self.core_nodes)
        return
    
    def process_nodes(self, name_keys, redo_key, func_key):
        save_paths = [self.save_data_dir + self.names_dict[name_key] for name_key in name_keys]
        remove_redo_paths(self.redo_dict[redo_key], save_paths)
        if all(os.path.exists(save_path) for save_path in save_paths):
            return [pickle.load(open(save_path,'rb')) for save_path in save_paths]
        else:
            outputs = self.functions_dict[func_key](self.full_processed, self.nodes)
            if type(outputs)!=list:
                outputs = [outputs]
            for i,output in enumerate(outputs):
                pickle.dump(output, open(save_paths[i],'wb'))
        return outputs
        
    def get_features(self):
        print('Getting samples node features')
        self.features = self.process_nodes(['features_name'], 'feature_redo', 'feature')[0]
        return
    
    def get_adjacency(self):
        print('Getting samples node adjacency')
        self.adjacency = self.process_nodes(['adjacency_name'], 'adjacency_redo', 'adjacency')[0]
        return
    
    def get_labels(self):
        print('Getting samples node labels')
        self.labels = self.process_nodes(['labels_name'], 'label_redo', 'label')[0]
        return

    def run_all(self):
        self.load_full_processed()
        self.get_hold_nodes()
        self.get_core_nodes()
        self.set_nodes()
        self.get_features()
        self.get_adjacency()
        self.get_labels()
        return