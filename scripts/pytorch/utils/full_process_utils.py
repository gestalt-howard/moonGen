# Authors: Aaron Wu / Howard Tai

# Script defining various helper functions for parsing a GraphDataProcess object


# ----------------------------------------------------------------------------------------------------------------------
# GraphDataProcess object parsing functions
# ----------------------------------------------------------------------------------------------------------------------
def get_nodes_types_map(full_processed):
    """
    Input: full_processed (from full_data_process.py)
        A GraphDataProcess object that neatly contains:
        1. All the problem-hold mappings, and problem-grade mappings (refer to nodeMapping.py)
        2. The full adjacency matrix (PMI for hold-hold, TFIDF for problem-hold) (refer to nodeAdjacency.py)
        3. The tfidf model to transform problems into tfidf mappings (refer to tfidfHolds.py)

    Output: nodes_types_map
        A dictionary where:
        1. The keys are processed node names (n1, n2, etc)
        2. The values are the type for that node (hold vs problem)

    Description:
        Takes node data inherently in a nodeMapping_obj and makes the type explicit.
        The original hold and problem names have 'h' or 'p' in them (h1, h2, p1, p2, etc) and are mapped to processed
        node names

    Purpose:
        Make it easier to identify node type in other functions
    """
    nodes_map = full_processed.nodeMapping_obj.maps_dict['nodes_map']
    nodes_types_map = dict()
    for node in nodes_map:
        if 'h' in node:
            nodes_types_map[nodes_map[node]] = 'hold'
        if 'p' in node:
            nodes_types_map[nodes_map[node]] = 'problem'
    return nodes_types_map


def get_holds_names(full_processed):
    """
    Input: full_processed (from full_data_process.py)

    Output: holds_names
        List of processed node names (n1, n2, etc) that correspond to holds only.
        This list are restricted to only holds that are used by the problem set and are sorted by name.

    Description:
        Retrieves hold names from a graphDataProcess object

    Purpose:
        Makes the retrieval shorter and neater in code
    """
    holds_names = full_processed.nodeAdjacency_obj.holds_names
    return holds_names


def get_prob_holds_map(full_processed):
    """
    Input: full_processed (from full_data_process.py)

    Output: prob_holds_map
        A dictionary where:
        1. The keys are processed node names (n1, n2, etc) for problems
        2. The values are processed node names (n3, n4, etc) for holds

    Description:
        Retrieves problem-hold mapping from a GraphDataProcess object

    Purpose:
        Makes the retrieval shorter and neater in code
    """
    prob_holds_map = full_processed.nodeMapping_obj.maps_dict['prob_hold_map']
    return prob_holds_map


def get_hold_adj(full_processed):
    """
    Input: full_processed (from full_data_process.py)

    Output: hold_adj
        The full PMI matrix that has already been calculated for the full set of holds

    Description:
        Retrieves PMI matrix from a GraphDataProcess object

    Purpose:
        Makes the retrieval shorter and neater in code
    """
    hold_adj = full_processed.nodeAdjacency_obj.pmi_mat
    return hold_adj


def get_prob_hold_adj(full_processed):
    """
    Input: full_processed (from full_data_process.py)

    Output: prob_hold_adj
        A dictionary where:
        1. The keys are processed node names (n1, n2, etc) for problems
        2. The values are dictionaries of processed node names (n3, n4, etc) of holds are mapped to TFIDF values

    Description:
        Retrieves a dictionary of TFIDF vectors from a GraphDataProcess object

    Purpose:
        Makes the retrieval shorter and neater in code
    """
    prob_hold_adj = full_processed.tfidf_obj.tfidf_dict
    return prob_hold_adj


def get_grades_dict(full_processed):
    """
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
    """
    grades_dict = full_processed.nodeMapping_obj.maps_dict['prob_grade_map']
    return grades_dict
