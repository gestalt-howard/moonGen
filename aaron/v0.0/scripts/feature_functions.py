import numpy as np

def get_nodes_types_map(full_processed):
    '''
    Input: full_processed (from full_data_process.py)
        An graphDataProcess object that neatly contains: 
        1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
        2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
        3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
    Output: nodes_types_map
        A dictionary where:
        1. The keys are processed node names (n1, n2, etc)
        2. The values are the type for that node (hold vs problem)
    Description:
        Takes node data inherently in a nodeMapping_obj and makes the type explicit.
        The original hold and problem names have 'h' or 'p' in them (h1, h2, p1, p2, etc) and are mapped to processed node names
    Purpose:
        Make it easier to identify node type in other functions
    '''
    nodes_map = full_processed.nodeMapping_obj.maps_dict['nodes_map']
    nodes_types_map = {}
    for node in nodes_map:
        if 'h' in node:
            nodes_types_map[nodes_map[node]] = 'hold'
        if 'p' in node:
            nodes_types_map[nodes_map[node]] = 'problem'
    return nodes_types_map

def get_prob_holds_map(full_processed):
    '''
    Input: full_processed (from full_data_process.py)
    Output: prob_holds_map
        A dictionary where:
        1. The keys are processed node names (n1, n2, etc) for problems
        2. The values are processed node names (n3, n4, etc) for holds
    Description:
        Retrieves problem-hold mapping from a graphDataProcess object
    Purpose:
        Makes the retrieval shorter and neater in code
    '''
    prob_holds_map = full_processed.nodeMapping_obj.maps_dict['prob_hold_map']
    return prob_holds_map

def gen_onehotfeatures(full_processed, nodes_keys):
    '''
    Input: 
        1. full_processed (from full_data_process.py)
            An graphDataProcess object that neatly contains: 
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix
    Output: onehot_features
        Diagonal matrix of size nxn (n = number of nodes)
    Description:
        Generates a diagonal matrix as features
    Purpose:
        Simple features for the GCN
    '''
    onehot_features = np.zeros((len(nodes_keys), len(nodes_keys)))
    for i in range(len(nodes_keys)):
        onehot_features[i][i] = 1
    return onehot_features

def gen_multihotfeatures(full_processed, nodes_keys):
    '''
    Input: 
        1. full_processed (from full_data_process.py)
            An graphDataProcess object that neatly contains: 
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix
    Output: multihot_features
        Matrix of features where the problem rows are repesented by a one-hot of their corresponding holds
    Description:
        1. Diagonal matrix of nodes
        2. For all problem node rows, also onehot the hold columns
    Purpose:
        Multihot features for the GCN
    '''
    nodes_types_map = get_nodes_types_map(full_processed)
    prob_hold_map = get_prob_holds_map(full_processed)
    multihot_features = np.zeros((len(nodes_keys), len(nodes_keys)))
    for i,n in enumerate(nodes_keys):
        node_type = nodes_types_map[n]
        multihot_features[i][i] = 1
        if 'p' in node_type:
            hold_nodes = prob_hold_map[n]
            for h in hold_nodes:
                multihot_features[i][nodes_keys.index(h)] = 1
    return multihot_features