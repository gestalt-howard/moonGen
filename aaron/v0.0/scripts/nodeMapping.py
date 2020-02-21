import pickle
import numpy as np
import os

class nodeMapping:
    '''
    Object that contains various mappings from the original raw data to a form more usable by other functions
    '''
    def __init__(self, data_path='', maps_path='', num_rows=18, num_cols=11, vmap=True):
        self.data_path = data_path
        self.maps_path = maps_path
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.vmap = vmap
        if vmap:
            self.load_vscale()
        
        self.data = None
        self.holds = None
        self.problems = None
        self.holds_map = None
        self.problems_map = None
        self.nodes_map = None
        self.problem_holds_map = None
        self.maps_dict = None
        return
    
    def load_data(self):
        self.data = pickle.load(open(self.data_path,'rb'))
        return
    
    def load_vscale(self):
        self.vmap_dict = {  # References scale defined by Aaron's grade_map.pickle
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
        return
    
    def get_holds_set(self):
        self.holds = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.holds.append([j,i])
        return

    def get_problems_set(self):
        self.problems = []
        data_keys = sorted(list(self.data.keys()))
        for k in data_keys:
            problem = {}
            problem['url'] = self.data[k]['url']
            problem['holds'] = self.data[k]['start'] + self.data[k]['mid'] + self.data[k]['end']
            problem['grade'] = self.data[k]['grade']
            if self.vmap:
                problem['grade'] = self.vmap_dict[problem['grade']]['v_scale']
            self.problems.append(problem)
        return
    
    def map_holds(self):
        self.holds_map = {}
        for i,hold in enumerate(self.holds):
            self.holds_map[str(hold[0])+'_'+str(hold[1])] = 'h'+str(i)
        return

    def map_problems(self):
        self.problems_map = {}
        for i,problem in enumerate(self.problems):
            self.problems_map[problem['url']] = 'p'+str(i)
        return

    def map_nodes(self):
        self.nodes_map = {}
        current_i = 0
        for k in sorted(list(self.holds_map.keys())):
            self.nodes_map[self.holds_map[k]] = 'n'+str(current_i)
            current_i+=1
        for k in sorted(list(self.problems_map.keys())):
            self.nodes_map[self.problems_map[k]] = 'n'+str(current_i)
            current_i+=1   
        return

    def problems2nodes(self):
        self.problem_holds_map = {}
        for i,problem in enumerate(self.problems):
            pID = self.problems_map[problem['url']]
            pID_node = self.nodes_map[pID]
            self.problem_holds_map[pID_node] = []
            for hold in problem['holds']:
                hID = self.holds_map[str(hold[0])+'_'+str(hold[1])]
                self.problem_holds_map[pID_node].append(self.nodes_map[hID])
        return
    
    def grade_problems(self):
        self.problem_grade_map = {}
        for i,problem in enumerate(self.problems):
            pID = self.problems_map[problem['url']]
            pID_node = self.nodes_map[pID]
            self.problem_grade_map[pID_node] = problem['grade']
        return
    
    def map_holds_idxs(self):
        self.holds_idxs_map = {}
        for i,holds in enumerate(self.holds):
            hID = self.holds_map[str(holds[0])+'_'+str(holds[1])]
            hID_node = self.nodes_map[hID]
            self.holds_idxs_map[hID_node] = holds[0]*self.num_cols + holds[1]
        return

    def map_all(self):
        self.map_holds()
        self.map_problems()
        self.map_nodes()
        self.problems2nodes()
        self.grade_problems()
        self.map_holds_idxs()
        self.maps_dict={}
        self.maps_dict['holds_map'] = self.holds_map
        self.maps_dict['holds_idxs_map'] = self.holds_idxs_map
        self.maps_dict['problems_map'] = self.problems_map
        self.maps_dict['nodes_map'] = self.nodes_map
        self.maps_dict['prob_hold_map'] = self.problem_holds_map
        self.maps_dict['prob_grade_map'] = self.problem_grade_map
        return
    
    def get_map(self):
        if os.path.exists(self.maps_path):
            self.maps_dict = pickle.load(open(self.maps_path,'rb'))
        else:
            self.load_data()
            self.get_holds_set()
            self.get_problems_set()
            self.map_all()
            if not self.maps_path=='':
                pickle.dump(self.maps_dict, open(self.maps_path,'wb'))
        return