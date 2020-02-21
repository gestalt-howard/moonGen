import numpy as np

def get_grades_dict(full_processed):
    '''
    Input: full_processed (from full_data_process.py)
        An graphDataProcess object that neatly contains: 
        1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
        2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
        3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
    Output: grades_dict
        A dictionary where:
        1. The keys are processed node names (n1, n2, etc) for problems
        2. The values are transformed grade of that problem (0,1,2,3...)
           Grades are mapped from Moonboard scale to V scale to a 0-intercept scale
    Description:
        Retrieves problem-grade map from a graphDataProcess object
    Purpose:
        Makes the retrieval shorter and neater in code
    '''
    grades_dict = full_processed.nodeMapping_obj.maps_dict['prob_grade_map']
    return grades_dict

def gen_labels_idxs(full_processed, nodes_keys):
    '''
    Input: 
        1. full_processed (from full_data_process.py)
            An graphDataProcess object that neatly contains: 
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix
    Output: labels
        Array of labels (0,1,2,3...) that corrsepond to the given node_keys
    Description:
        Loads a problem-grade map and uses it to retrieve grades for each problem in a list
    Purpose:
        Generate labels for train/test
    '''
    grades_dict = get_grades_dict(full_processed)
    labels = np.zeros(len(nodes_keys))
    for i,key in enumerate(nodes_keys):
        if key in grades_dict:
            labels[i] = grades_dict[key]
    labels_set = sorted(list(set(labels)))
    labels_dict = {}
    for i,label in enumerate(labels_set):
        labels_dict[label] = i
    for i in range(labels.shape[0]):
        labels[i] = labels_dict[labels[i]]
    return labels

def gen_onehot_labels_idxs(full_processed, nodes_keys):
    '''
    Input: 
        1. full_processed (from full_data_process.py)
            An graphDataProcess object that neatly contains: 
            1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
            2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
            3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)
        2. nodes_keys
            Subset of processed node names (n1, n2, etc) to be used to generate adjacency matrix
    Output: onehot_labels
        Matrix of onehot labels (0 = [1,0,0...]) that corrsepond to the given node_keys
    Description:
        Loads a problem-grade map and uses it to generate onehot grades for each problem in a list
    Purpose:
        Generate labels for train/test
    '''
    labels = gen_labels_idxs(full_processed, nodes_keys)
    labels_set = list(set(labels))
    labels_dict = {}
    for i,label in enumerate(labels_set):
        labels_dict[label] = i
    onehot_labels = np.zeros((len(nodes_keys),len(labels_set)))
    for i,l in enumerate(labels):
        onehot_labels[i][labels_dict[l]] = 1
    return onehot_labels