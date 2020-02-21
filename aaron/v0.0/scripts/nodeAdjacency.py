import pickle
import numpy as np
import os
import pdb

class nodeAdjacency:
    '''
    Object that calculates and contains PMI adjacency information
    '''
    def __init__(self, maps_dict=None, holds_names_path='', problems_names_path='', holds_mat_path='', pmi_path=''):
        self.problems_names_path=problems_names_path
        self.holds_names_path=holds_names_path
        self.holds_mat_path=holds_mat_path
        self.pmi_path=pmi_path
        
        self.maps_dict=maps_dict
        self.holds_names=None
        self.problems_names=None
        self.holds_mat=None
        self.hold_probs=None
        self.joint_hold_probs=None
        self.pmi_mat=None
        return
    
    def load_holds_names(self):
        if os.path.exists(self.holds_names_path):
            self.holds_names = pickle.load(open(self.holds_names_path,'rb'))
        else:
            self.get_holds_names()
            if not self.holds_names_path=='':
                pickle.dump(self.holds_names, open(self.holds_names_path,'wb'))
        return
    
    def load_problems_names(self):
        if os.path.exists(self.problems_names_path):
            self.problems_names = pickle.load(open(self.problems_names_path,'rb'))
        else:
            self.get_problems_names()
            if not self.problems_names_path=='':
                pickle.dump(self.problems_names, open(self.problems_names_path,'wb'))
        return
    
    def load_holds_mat(self):
        if self.holds_names==None:
            self.load_holds_names()
        if self.problems_names==None:
            self.load_problems_names()
        if os.path.exists(self.holds_mat_path):
            self.holds_mat = pickle.load(open(self.holds_mat_path,'rb'))
        else:
            self.make_holds_mat()
            if not self.holds_mat_path=='':
                pickle.dump(self.holds_mat, open(self.holds_mat_path,'wb'))
        return
    
    def load_pp_pmi(self):
        if self.holds_mat==None:
            self.load_holds_mat()
        if os.path.exists(self.pmi_path):
            self.pmi_mat = pickle.load(open(self.pmi_path,'rb'))
        else:
            self.calc_hold_probs()
            self.calc_joint_hold_probs()
            self.calc_pp_pmi()
            if not self.pmi_path=='':
                pickle.dump(self.pmi_mat, open(self.pmi_path,'wb'))
        return
    
    def get_holds_names(self)
        problems = sorted(list(self.maps_dict['prob_hold_map'].keys()))
        all_holds = []
        for problem in problems:
            all_holds+=self.maps_dict['prob_hold_map'][problem]
        self.holds_names = sorted(list(set(all_holds)))
        return
    
    def get_problems_names(self):
        nodes_dict = self.maps_dict['nodes_map']
        self.problems_names = [nodes_dict[n] for n in nodes_dict if 'p' in n]
        return

    def make_holds_mat(self):
        self.holds_mat = np.zeros((len(self.problems_names),len(self.holds_names)))
        for i,pname in enumerate(self.problems_names):
            for hname in self.maps_dict['prob_hold_map'][pname]:
                self.holds_mat[i][self.holds_names.index(hname)]=1
        return
    
    def calc_hold_probs(self):
        self.hold_probs = np.sum(self.holds_mat, axis=0)/float(self.holds_mat.shape[0])
        return

    def calc_joint_hold_probs(self):
        joint_hold_probs = np.zeros((self.holds_mat.shape[1],self.holds_mat.shape[1]))
        for holds in self.holds_mat:
            joint_hold_probs = joint_hold_probs + np.outer(holds,holds)
        self.joint_hold_probs = joint_hold_probs/float(self.holds_mat.shape[0])
        return 

    def calc_pp_pmi(self):
        outer_hold_probs = np.outer(self.hold_probs,self.hold_probs)
        self.pmi_mat = np.log2(np.divide(self.joint_hold_probs + .0001, outer_hold_probs + .0001))
        self.pmi_mat = np.clip(self.pmi_mat, a_min=0, a_max=None)
        return