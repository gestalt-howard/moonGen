import numpy as np
import torch
import os
import pickle

from adjacency_functions import *
from feature_functions import *
from label_functions import *
from utils import *

#-------------------------------------------------------------
# Sampling Functions
#-------------------------------------------------------------
def sample_nodes_balanced(nodes_grades_dict, params):
    '''
    Input:
        1. nodes_grades_dict
            A dictionary where:
            1. The keys are processed node names (n1, n2, etc) for problems
            2. The values are transformed grade of that problem (0,1,2,3...)
               Grades are mapped from Moonboard scale to V scale to a 0-intercept scale
        2. params
            1. num_per_core = number of nodes per unique grade
            2. target_grades = isolated set of grades to run on instead of the full set
    Output: node_samples
        List of shuffled processed node names (n1, n2, etc) sampled from the full set.
    Description:
        1. Maps each grade to a list of nodes
        2. Randomly samples n nodes from each grade
        3. No replacement: if the number of nodes per core exceeds the full set, no more will be retrieved
    Purpose:
        Sample a subset of the full set of nodes for:
            1. Balancing classes
            2. Subsetting a network that might be too large for memory
    '''
    num_per_core = params['num_per_core']
    node_samples = []
    grades_dict = {}
    for node in nodes_grades_dict:
        if nodes_grades_dict[node] in grades_dict:
            grades_dict[nodes_grades_dict[node]].append(node)
        else:
            grades_dict[nodes_grades_dict[node]] = [node]
            
    target_grades = list(grades_dict.keys())
    if 'target_grades' in params:
        target_grades = params['target_grades']
    for grade in target_grades:
        shuffle = np.random.permutation(len(grades_dict[grade]))
        node_samples+=[grades_dict[grade][i] for i in shuffle[:num_per_core]]
    return node_samples

def sample_nodes_balanced_replaced(nodes_grades_dict, params):
    '''
    Input:
        1. nodes_grades_dict
            A dictionary where:
            1. The keys are processed node names (n1, n2, etc) for problems
            2. The values are transformed grade of that problem (0,1,2,3...)
               Grades are mapped from Moonboard scale to V scale to a 0-intercept scale
        2. params
            1. num_per_core = number of nodes per unique grade
            2. target_grades = isolated set of grades to run on instead of the full set
    Output: node_samples
        List of shuffled processed node names (n1, n2, etc) sampled from the full set.
    Description:
        1. Maps each grade to a list of nodes
        2. Randomly samples n nodes from each grade
        3. With replacement: nodes already sampled can still be sampled
    Purpose:
        Sample a subset of the full set of nodes for:
            1. Balancing classes
            2. Subsetting a network that might be too large for memory
    '''
    num_per_core = params['num_per_core']
    node_samples = []
    grades_dict = {}
    for node in nodes_grades_dict:
        if nodes_grades_dict[node] in grades_dict:
            grades_dict[nodes_grades_dict[node]].append(node)
        else:
            grades_dict[nodes_grades_dict[node]] = [node]
    target_grades = list(grades_dict.keys())
    if 'target_grades' in params:
        target_grades = params['target_grades']
    for grade in target_grades:
        for i in range(num_per_core):
            shuffle = np.random.permutation(len(grades_dict[grade]))
            node_samples+=[grades_dict[grade][shuffle[0]]]
    return node_samples

def sample_target_nodes_balanced(nodes_grades_dict, params):
    '''
    Input:
        1. nodes_grades_dict
            A dictionary where:
            1. The keys are processed node names (n1, n2, etc) for problems
            2. The values are transformed grade of that problem (0,1,2,3...)
               Grades are mapped from Moonboard scale to V scale to a 0-intercept scale
        2. params
            1. num_per_core = number of nodes per unique grade
            2. target_grade = a single grade to sample from
    Output: node_samples
        List of shuffled processed node names (n1, n2, etc) sampled from the full set.
    Description:
        1. Maps each grade to a list of nodes
        2. Randomly samples n nodes from each grade except the target grade
        3. Takes only m nodes from this set of non-target nodes
        4. Randomly samples m nodes from the target grade
    Purpose:
        Sample a subset of the full set of nodes for:
            1. Balancing number of target nodes and non-target nodes (when it's 1 class vs k classes)
            2. Subsetting a network that might be too large for memory
    '''
    num_per_core = params['num_per_core']
    target_grade = params['target_grade']
    node_samples = []
    target_samples = []
    grades_dict = {}
    for node in nodes_grades_dict:
        if nodes_grades_dict[node] in grades_dict:
            grades_dict[nodes_grades_dict[node]].append(node)
        else:
            grades_dict[nodes_grades_dict[node]] = [node]
    nontarget_samples = []
    for grade in grades_dict:
        if grade==target_grade:
            continue
        shuffle = np.random.permutation(len(grades_dict[grade]))
        nontarget_samples+=[grades_dict[grade][i] for i in shuffle[:num_per_core]]
    if target_grade in grades_dict:
        shuffle = np.random.permutation(len(grades_dict[target_grade]))
        target_samples = [grades_dict[target_grade][i] for i in shuffle[:num_per_core]]
    shuffle = np.random.permutation(len(nontarget_samples))[:num_per_core]
    node_samples = target_samples + [nontarget_samples[i] for i in shuffle]
    return node_samples

def load_upsample_nodes(nodes_grades_dict, params):
    '''
    Input:
        1. nodes_grades_dict
            A dictionary where:
            1. The keys are processed node names (n1, n2, etc) for problems
            2. The values are transformed grade of that problem (0,1,2,3...)
               Grades are mapped from Moonboard scale to V scale to a 0-intercept scale
        2. params
            1. sample_nodes_path = path to previously sampled nodes to upsample (for direct comparison)
            2. target_grade = a single grade to upsample relative to other grades
            3. unbalance_multiplier = number of times a target grade set is repeated
    Output: node_samples
        List of shuffled processed node names (n1, n2, etc) sampled from the full set.
    Description:
        1. Loads a previously sampled set of nodes
        2. Map each node to a grade and extract the nodes of a target grade
        3. Upsample the target nodes and add them back to the original list
    Purpose:
        For testing the effects of unbalanced sampling
    '''
    nodes_path = params['sample_nodes_path']
    target_grade = params['target_grade']
    multiplier = params['unbalance_multiplier']
    node_samples = pickle.load(open(nodes_path,'rb'))
    target_nodes = [node for node in node_samples if nodes_grades_dict[node]==target_grade]
    added_nodes = list(target_nodes)*(multiplier-1)
    node_samples+=added_nodes
    return node_samples
    
#-------------------------------------------------------------

#-------------------------------------------------------------
# Train Dev Test Split Functions
#-------------------------------------------------------------
def split_nodes(node_set, split_ratio):
    '''
    Input:
        1. node_set = full set of nodes to be split
        2. split_ratio = ratio to split the set between 2 subsets
    Output:
        1. set1 = set that contains n*split_ratio values
        2. set2 = set that contains n - n*split_ratio values
    Description:
        Randomly splits values in a set based on a split_ratio
    Purpose:
        Splitting between train/dev and test, and train and dev
    '''
    num_split = int(len(node_set)*split_ratio)
    shuffle = np.random.permutation(len(node_set))
    set1 = [node_set[i] for i in shuffle[:num_split]]
    set2 = [node_set[i] for i in shuffle[num_split:]]
    return set1, set2

def train_dev_test_split(node_set, split_ratio_dict):
    '''
    Input:
        1. node_set = full set of nodes to be split
        2. split_ratio = dictionary that dictates the test split and the dev split
    Output:
        1. train_set = nodes to be considered for train
        2. dev_set = nodes to be considered for validation
        3. test_set = nodes to be considered for test
    Description:
        Splits a set of nodes into train, dev, test.
    Purpose:
        Total data needs to be split in a consistent manner
    '''
    train_dev_set, test_set = split_nodes(node_set, split_ratio_dict['test'])
    train_set, dev_set = split_nodes(train_dev_set, split_ratio_dict['dev'])
    return train_set, dev_set, test_set

def get_split_dict(core_nodes_id_dict, hold_nodes_id_dict, split_ratio_dict):
    '''
    Input
        1. core_nodes_id_dict = dictionary that maps sampled problem nodes to an index in feature/adjacency/label (from sub_data_process.py)
            Note: core nodes can be mapped to multiple indices for upsampling purposes
        2. hold_nodes_id_dict = dictionary that maps sampled hold nodes to an index in feature/adjacency/label (from sub_data_process.py)
        3. split_ratio_dict = dictionary that dictates the test split and the dev split
    Output: split_dict
        Dictionary of train, dev, test feature/adjacency/label indices that correspond to nodes in their set
    Description:
        1. Get core problem nodes and split them into train, dev, test
            Hold nodes aren't considered since they don't have labels
        2. Get the indices corrseponding to the problem nodes (can be multiple for one problem)
    Purpose:
        Total data needs to be split in a consistent manner
    '''
    split_dict = {}
    core_nodes = list(core_nodes_id_dict.keys())
    train_set, dev_set, test_set = train_dev_test_split(core_nodes, split_ratio_dict)
    holds_idxs = [hold_nodes_id_dict[i] for i in hold_nodes_id_dict]
    split_dict['train_idxs'] = []
    for i in train_set:
        split_dict['train_idxs']+=core_nodes_id_dict[i]
    split_dict['dev_idxs'] = []
    for i in dev_set:
        split_dict['dev_idxs']+=core_nodes_id_dict[i]
    split_dict['test_idxs'] = []
    for i in test_set:
        split_dict['test_idxs']+=core_nodes_id_dict[i]
    split_dict['hold_idxs'] = holds_idxs
    return split_dict
#-------------------------------------------------------------

#-------------------------------------------------------------
# General Utils
#-------------------------------------------------------------
def set_default(current_val, fail_val, default_val):
    '''
    Input:
        1. current_val = current value being considered
        2. fail_val = value used to determine if current_val is a failure
        3. default_val = default assigned value if current_val==fail_val
    Output: current_val
    Description:
        current_val = default_val if failed
        otherwise current_val = default_val
    Purpose:
        Checks and sets a default value
    '''
    if current_val==fail_val:
        return default_val
    return current_val
    
def set_default_dict(current_dict, key, fail_val, default_val):
    '''
    Input:
        1. current_dict = current dictionary being considered
        2. key = current key of current_dict being considered
        3. fail_val = value used to determine if current_dict[key] is a failure
        4. default_val = default assigned value if current_dict[key]==fail_val
    Output: current_dict[key]
    Description:
        current_dict[key] = default_val if failed
    Purpose:
        Checks and sets a default value for dictionary key
    '''
    if key not in current_dict:
        return default_val
    if current_dict[key]==fail_val:
        return default_val
    return current_dict[key]

def rev_dict(mapping):
    '''
    Inverts the mapping of a dictionary
    '''
    rev_mapping={}
    for m in mapping:
        rev_mapping[mapping[m]] = m
    return rev_mapping

def set_paths(model_type, ver, data_dir, result_dir):
    '''
    Sets some default paths and makes default directories if they don't exist
    '''
    data_path = data_dir + model_type + '/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    data_path = data_path + ver + '/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    result_path = result_dir + model_type + '/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_path = result_path + ver + '/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    return data_path, result_path

def remove_redo_paths(redo, paths_list):
    '''
    Deletes all items in a given list of paths from hard-drive memory.
    Used to help facilitate re-doing a section of the pipeline.
    '''
    if redo:
        for p in paths_list:
            if os.path.exists(p):
                os.remove(p)
    return

def str_to_func(func_str):
    '''
    Retrieves the function whose name corresponds to a given string
    '''
    function = globals()[func_str]
    return function

def get_func_dict(str_dict):
    '''
    Turns a dictionary of strings that contain function names into a dictionary that contains those functions
    '''
    func_dict = {}
    for func in str_dict:
        func_dict[func] = str_to_func(str_dict[func])
    return func_dict
#-------------------------------------------------------------

#-------------------------------------------------------------
# Pytorch Functions
#-------------------------------------------------------------
def sample_and_load_pytorch_data(subgraph_data_obj, split_ratio_dict, save_path, target_label=-1, redo=False):
    '''
    Input:
        1. subgraph_data_obj (refer to sub_data_process.py)
            A subGraphProcess object that neatly contains: 
            1. Mapping of the problem nodes to their respective idxs in features/adjacency/labels (core_nodes_id_dict)
            2. Mapping of the hold nodes to their respective idxs in features/adjacency/labels (hold_nodes_id_dict)
            3. Generated matrix of features for subsampled problem nodes + hold nodes (features)
            4. Generated adjacency matrix for subsampled problem nodes + hold nodes (adjacency)
            5. Generated list of labels for subsampled problem nodes + hold nodes (labels)
        2. split_ratio_dict = dictionary that dictates the test split and the dev split
        3. save_path = path to save some intermediate outputs before training
        4. target_label = target grade for binary classification (-1 for multiclass)
        5. redo = deletes intermediate outputs if True (will load saved intermediates otherwise)
    Output:
        1. features = features to be used by the model
        2. adj = adjacency matrix to be used by the model
        3. labels = labels that correspond to the features 
        4. idx_train = indices for training loss
        5. idx_dev = indices for validation during training
        6. idx_test = indices for testing
    Description:
        1. Retrieves the core problem nodes and the hold nodes
        2. Split the data into lists of indices based on given split_dict
        3. Retrieve features/labels/adjacency and convert them to torch tensors
        4. Save intermediate files
    Purpose:
        Sampling done prior to training and testing
    '''
    files = ['features.pickle', 'adj.pickle', 'labels.pickle', 'idx_train.pickle',
             'idx_dev.pickle', 'idx_test.pickle']
    remove_redo_paths(redo, [save_path+f for f in files])
    if all(os.path.exists(save_path+file) for file in files):
        features = pickle.load(open(save_path+'features.pickle','rb'))
        adj = pickle.load(open(save_path+'adj.pickle','rb'))
        labels = pickle.load(open(save_path+'labels.pickle','rb'))
        idx_train = pickle.load(open(save_path+'idx_train.pickle','rb'))
        idx_dev = pickle.load(open(save_path+'idx_dev.pickle','rb'))
        idx_test = pickle.load(open(save_path+'idx_test.pickle','rb'))
    else:
        core_nodes_id_dict = subgraph_data_obj.core_nodes_id_dict
        hold_nodes_id_dict = subgraph_data_obj.hold_nodes_id_dict
        split_dict = get_split_dict(core_nodes_id_dict, hold_nodes_id_dict, split_ratio_dict)

        features = torch.FloatTensor(np.array(subgraph_data_obj.features))
        adj = torch.FloatTensor(subgraph_data_obj.adjacency)
        if target_label!=-1:
            labels = torch.LongTensor((subgraph_data_obj.labels==target_label)*1)
        else:    
            labels = torch.LongTensor(subgraph_data_obj.labels)

        idx_train = torch.LongTensor(split_dict['train_idxs'])
        idx_dev = torch.LongTensor(split_dict['dev_idxs'])
        idx_test = torch.LongTensor(split_dict['test_idxs'])
        
        pickle.dump(features, open(save_path+'features.pickle','wb'))
        pickle.dump(adj, open(save_path+'adj.pickle','wb'))
        pickle.dump(labels, open(save_path+'labels.pickle','wb'))
        pickle.dump(idx_train, open(save_path+'idx_train.pickle','wb'))
        pickle.dump(idx_dev, open(save_path+'idx_dev.pickle','wb'))
        pickle.dump(idx_test, open(save_path+'idx_test.pickle','wb'))

    return features, adj, labels, idx_train, idx_dev, idx_test
#-------------------------------------------------------------