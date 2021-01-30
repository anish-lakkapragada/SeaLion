from collections import Counter
import numpy as np
cimport numpy as np
import math

cdef list labels
cpdef dict probabilities(list labels) :
    cdef dict count_labels = {label : num_counts/len(labels) for label, num_counts in dict(Counter(labels)).items()}
    return count_labels #{label : num_times}

cpdef double gini_impurity(list labels) :
    cdef dict class_probabilities = probabilities(labels)
    cdef double _, prob
    return 1 - sum([prob ** 2 for _, prob in class_probabilities.items()])

cpdef find_best_split(list data, list labels, int min_samples) :
    """
    :param data: Incoming data
    :return: 2-element tuple (x, y) where x is the index of the feature, and y is the value of that feature for that split.
    We measure how good a split is by how low the new entropy is.
    """

    cdef double incoming_impurity = gini_impurity(labels)
    cdef dict impurity_to_split = {} #impurity : store the (x, y) tuple split]
    cdef dict impurity_to_num_samples = {} #store each impurity to its num_samples
    cdef double label, percent_true, percent_false, impurity_of_split, best_impurity, num_true, num_false
    cdef list input, true_branch, false_branch, observation
    cdef tuple current_split
    cdef int feature
    for input in data :
        for feature in range(len(input)) :
            '''entropy calculated as num_false * false_impurity + num_true * true_impurity'''
            current_split = (feature, input[feature])
            true_branch, false_branch = [], []
            for observation, label in zip(data, labels) :
                if isinstance(observation[current_split[0]], list) :
                    '''this means that it is a one-hot-encoded label, so a data is in the true branch if it is equal to it'''
                    if observation[current_split[0]] == current_split[1] : true_branch.append(label)
                    else : false_branch.append(label)
                elif isinstance(float(observation[current_split[0]]), float) :
                    if observation[current_split[0]] >= current_split[1] : true_branch.append(label)
                    else : false_branch.append(label)
            num_true = len(true_branch)
            num_false = len(false_branch)
            percent_true = num_true/len(data)
            percent_false = num_false/len(data)
            assert percent_false + percent_true == 1, "should be equal to 1"
            impurity_of_split = percent_true * gini_impurity(true_branch) + percent_false * gini_impurity(false_branch)
            impurity_to_split[impurity_of_split] = current_split
            impurity_to_num_samples[impurity_of_split] = len(false_branch) + len(true_branch)

    #find the split ( x, y combo) leading to the least entropy
    best_impurity = min(impurity_to_split)
    if best_impurity < incoming_impurity :
        return impurity_to_split[best_impurity]
    if incoming_impurity >= best_impurity or impurity_to_num_samples[best_impurity] < min_samples :
        '''no improvement, stop and create a leaf node by returning the most common label'''
        return Counter(labels).most_common(1)[0][0]

cpdef tuple branch_data(list data, list labels, tuple split) :
    """
    :param data: inputs
    :param split: (x, y) combination to split data
    :return: true_branches, true_labels, false_branches, false_labels (containing the inputs which were false and true on the given split)
    """
    cdef list true_branch, false_branch, true_labels, false_labels
    true_branch, false_branch = [], []
    true_labels, false_labels = [], []
    cdef list observation
    cdef double label
    for observation, label in zip(data, labels):
        if isinstance(split[1], list) :
            if observation[split[0]] == split[1] :
                true_branch.append(observation)
                true_labels.append(label)
            else :
                false_branch.append(observation)
                false_labels.append(label)
        elif isinstance(float(split[1]), float) :
            if observation[split[0]] >= split[1] :
                true_branch.append(observation)
                true_labels.append(label)
            else :
                false_branch.append(observation)
                false_labels.append(label)
    return true_branch, true_labels, false_branch, false_labels



cpdef double go_down_the_tree(list data_point, dict recursive_tree):
        '''takes in only one data point. Goes down the tree by finding whether it is on the true or false side of each branch and going down
        down that direction recursively until it hits a leaf node'''
        try:
            data_point = np.array(data_point).tolist()
        except Exception:
            pass
        cdef tuple split = list(recursive_tree.keys())[0]  # get the current spit
        cdef bint true_or_false
        try:
            if isinstance(float(split[1]), float):
                true_or_false = data_point[split[0]] >= split[1]
        except Exception:
            if isinstance(anti_tupelize(split[1]), tuple):
                true_or_false = list(data_point[split[0]]) == list(split[1])
        subtree = recursive_tree[split][true_or_false]
        try:
            if isinstance(float(subtree), float):
                return subtree  # its a leaf node
        except Exception:
            return go_down_the_tree(data_point, subtree)


cpdef np.ndarray[np.float64_t, ndim=1] chunk_predict(list data_chunk_with_tree):
    cdef dict tree
    cdef list data_chunk
    data_chunk, tree = data_chunk_with_tree

    cdef list predictions = [go_down_the_tree(data_point, tree) for data_point in data_chunk]
    return np.array(predictions)


cpdef tuple tuplelize(tpl) :
    """Change tpl (1, [2, 3]) to (1, (2, 3)) so it can be stored in a dictionary (the tree dict)"""
    cdef list new_tuple = []
    for element in tpl :
        if isinstance(element, list) :
            new_tuple.append(tuple(element))
        else : new_tuple.append(element)
    return tuple(new_tuple)

cpdef tuple anti_tupelize(tuple tpl) :
    """Take in tuple (1, (2, 3)) for instance in the tree and make it (1, [2, 3]) as the y (which is the value) is a list, not tuple in given dataset"""
    cdef list new_tuple = []
    for element in tpl :
        if isinstance(element, tuple) :
            new_tuple.append(list(element))
        else : new_tuple.append(element)
    return tuple(new_tuple)


cdef class CythonTrainDecisionTraining() :
    cdef dict splits_to_most_common_labels
    cdef dict data_points_to_count
    def __cinit__(self) :
        self.splits_to_most_common_labels = {}
        self.data_points_to_count = {}

    cpdef build_tree(self, list data, list labels, int min_samples, double max_branches = math.inf):
        '''build tree recursively inside this method'''
        try :
            data = np.array(data).tolist()
            labels = np.array(labels).tolist()
        except Exception :
            pass
        best_split = find_best_split(data, labels, min_samples)
        cdef list true_branch, true_labels, false_branch, false_labels
        cdef double most_common_label
        try:
            x = best_split/2
            return best_split
        except Exception:
            true_branch, true_labels, false_branch, false_labels = branch_data(data, labels, best_split)
            self.splits_to_most_common_labels[tuplelize(best_split)] = \
            Counter(true_labels + false_labels).most_common(1)[0][0]
            for point in true_branch :
                if tuplelize(point) in self.data_points_to_count :
                    self.data_points_to_count[tuplelize(point)] += 1
                    if self.data_points_to_count[tuplelize(point)] >= max_branches :
                        most_common_label = Counter(list(labels)).most_common(1)[0][0]
                        return most_common_label
                else :
                    self.data_points_to_count[tuplelize(point)] = 1
            tree = {tuplelize(best_split): {True: CythonTrainDecisionTraining.build_tree(self, true_branch, true_labels, min_samples,  max_branches),
                                            False: CythonTrainDecisionTraining.build_tree(self, false_branch, false_labels, min_samples, max_branches)}}
            return tree

    cpdef dict get_data_points_to_count(self) :
        return dict(self.data_points_to_count)

    cpdef dict get_splits_to_MCLs(self) :
        return self.splits_to_most_common_labels

