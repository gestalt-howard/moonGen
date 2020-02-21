import numpy as np

def get_nodes_types_map(full_processed):
    '''
    Input: full_processed (from full_data_process.py)
        A graphDataProcess object that neatly contains: 
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

def get_holds_names(full_processed):
    '''
    Input: full_processed (from full_data_process.py)
    Output: holds_names
        List of processed node names (n1, n2, etc) that correspond to holds only.
        This list are restricted to only holds that are used by the problem set and are sorted by name.
    Description:
        Retrieves hold names from a graphDataProcess object
    Purpose:
        Makes the retrieval shorter and neater in code
    '''
    holds_names = full_processed.nodeAdjacency_obj.holds_names
    return holds_names

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

def get_hold_adj(full_processed):
    '''
    Input: full_processed (from full_data_process.py)
    Output: hold_adj
        The full pointwise mutual information matrix that has already been calculated for the full set
    Description:
        Retrieves pmi matrix from a graphDataProcess object
    Purpose:
        Makes the retrieval shorter and neater in code
    '''
    hold_adj = full_processed.nodeAdjacency_obj.pmi_mat
    return hold_adj

def get_prob_hold_adj(full_processed):
    '''
    Input: full_processed (from full_data_process.py)
    Output: prob_hold_adj
        A dictionary where:
        1. The keys are processed node names (n1, n2, etc) for problems
        2. The values are dictionaries of processed node names (n3, n4, etc) of holds are mapped to tfidf values
    Description:
        Retrieves a dictionary of tfidf vectors from a graphDataProcess object
    Purpose:
        Makes the retrieval shorter and neater in code
    '''
    prob_hold_adj = full_processed.tfidf_obj.tfidf_dict
    return prob_hold_adj

def norm_adjacency(adj_mat):
    '''
    Input: adj_mat
        An adjacency matrix (to be used for train/test)
    Output: norm_adj_mat
        Normalized adjacency matrix
    Description:
        Norm_A = (D^-1/2) x A  x (D^-1/2)
        D = degree matrix
        A = adjacency matrix
        Norm_A = normalized adjacency matrix
    Purpose:
        Normalize an adjacency matrix to account for differing number of neighbors
    '''
    norm_degree_mat = np.zeros((adj_mat.shape))
    normed = 1/np.sqrt(np.sum(adj_mat+.001, axis=0))
    for i in range(len(normed)):
        norm_degree_mat[i,i] = normed[i]
    norm_adj_mat = np.matmul(np.matmul(norm_degree_mat,adj_mat),norm_degree_mat)
    return norm_adj_mat

def gen_adjacency(full_processed, nodes_keys):
    '''
    Input: 
        1. full_processed (from full_data_process.py)
            An graphDataProcess object that neatly contains: 
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix
    Output: adj_mat
        An adjacency matrix (to be used for train/test)
    Description:
        1. Takes a set of nodes that represent both problems and holds (nodes_types_map determines node type)
        2. Gets the pmi matrix (hold-hold adjacency) and tfidf dictionary (problem-hold adjacency)
        3. Iterate through nodes:
            1. Determine hold types and retrieve corresponding values
            2. Self Adjacency: if node is compared against itself, 1
            3. Populate an adjacency matrix with values
               (hold_names is needed to determine where in the full pmi matrix to draw from since it is a numpy mat)
    Purpose:
        Generate the pmi/tfidf adjacency matrix to be used for train/test
    '''
    nodes_types_map = get_nodes_types_map(full_processed)
    holds_names = get_holds_names(full_processed)
    hold_adj = get_hold_adj(full_processed)
    prob_hold_adj = get_prob_hold_adj(full_processed)
    
    adj_mat = np.zeros((len(nodes_keys),len(nodes_keys)))

    for i,node1 in enumerate(nodes_keys):
        node_type1 = nodes_types_map[node1]
        for j,node2 in enumerate(nodes_keys[i:]):
            if not adj_mat[i,j+i]==0:
                continue
            node_type2 = nodes_types_map[node2]
            if j==0:
                adj_mat[i,i]=1
            if node_type1=='problem' and node_type2=='problem':
                continue
            elif node_type1=='problem' and node_type2=='hold':
                adj_mat[i,j+i] = prob_hold_adj[node1][node2]
                adj_mat[j+i,i] = adj_mat[i,j+i]
            elif node_type1=='hold' and node_type2=='problem':
                adj_mat[i,j+i] = prob_hold_adj[node2][node1]
                adj_mat[j+i,i] = adj_mat[i,j+i]
            elif node_type1=='hold' and node_type2=='hold':
                adj_mat[i,j+i] = hold_adj[holds_names.index(node1),holds_names.index(node2)]
                adj_mat[j+i,i] = hold_adj[holds_names.index(node2),holds_names.index(node1)]
    return adj_mat

def gen_adjacency_norm(full_processed, nodes_keys):
    '''
    Input: 
        1. full_processed (from full_data_process.py)
        2. nodes_keys
    Output: normalized pmi/tfidf adjacency
    Description:
        Runs normalization on pmi/tfidf adjacency matrix
    Purpose:
        Needed for a specific invoking of norm separate from regular adjacency
    '''
    return norm_adjacency(gen_adjacency(full_processed, nodes_keys))

def onehot_adjacency(full_processed, nodes_keys):
    '''
    Input:
        1. full_processed (from full_data_process.py)
        2. nodes_keys
    Output: adj_mat
        An adjacency matrix (to be used for train/test)
    Description:
        1. Takes a set of nodes that represent both problems and holds (nodes_types_map determines node type)
        2. Gets the problem-hold mapping from a graphDataProcess object
        3. Iterate through nodes:
            1. If the nodes are problem and a hold corresponding to that problem, then 1
            2. Else 0 (not even self-adjacency)
    Purpose:
        Generate the onehot adjacency matrix to be used for train/test
    '''
    nodes_types_map = get_nodes_types_map(full_processed)
    prob_holds_map = get_prob_holds_map(full_processed)
    
    adj_mat = np.zeros((len(nodes_keys),len(nodes_keys)))
    
    for i,node1 in enumerate(nodes_keys):
        node_type1 = nodes_types_map[node1]
        for j,node2 in enumerate(nodes_keys[i:]):
            node_type2 = nodes_types_map[node2]
            if j==0:
                continue
            if node_type1=='problem' and node_type2=='problem':
                continue
            elif node_type1=='problem' and node_type2=='hold':
                if node2 in prob_holds_map[node1]:
                    adj_mat[i,j+i] = 1
            elif node_type1=='hold' and node_type2=='hold':
                continue
    return adj_mat

def onehot_adjacency_norm(full_processed, nodes_keys):
    '''
    Input:
        1. full_processed (from full_data_process.py)
        2. nodes_keys
    Output: normalized onehot adjacency
    Description:
        Runs normalization on onehot adjacency matrix
    Purpose:
        Needed for a specific invoking of norm separate from regular adjacency
    '''
    return norm_adjacency(onehot_adjacency(full_processed, nodes_keys))