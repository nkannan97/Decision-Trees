from scipy.io import arff
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
from Node import dtreeNode

'''
calculates the conditional entropy for nominal features
'''


def compute_entropy_on_thresholds(d, idx, thresh_val,feature_range):
    # stores values less than the threshold
    instances_less_thresh = []
    # stores values greater than the threshold
    instances_greater_thresh = []

    # subset_labels = []
    conditional_entropy = 0

    # iterate through the dataset
    for instance_train in d:
        # add instances to the list depending on the threshold value
        if instance_train[idx] <= thresh_val:
            instances_less_thresh.append(instance_train)
        else:
            instances_greater_thresh.append(instance_train)

    # list of list to maintain subsets
    data_sub = [instances_less_thresh, instances_greater_thresh]
    # iterate through the sub lists
    for sub_data in data_sub:
        subset_labels = []
        for sub_instance in sub_data:
            subset_labels.append(sub_instance[-1])
        probability_subset = 1.0 * len(subset_labels) / len(d)
        # calculates H(Y|feature)
        conditional_entropy += probability_subset * compute_entropy_on_labels(np.array(sub_data),feature_range)

    return conditional_entropy

    # iterate through the sub lists

'''
Finds the best threshold value to split on when the feature is numeric
'''
def find_best_numeric_feature(df, index,feature_range):
    # to store values for each of the feature
    value_lst = []
    #
    for instance in df:
        value_lst.append(instance[index])

    minimum_entropy = float("inf")
    optimal_threshold = 0

    # finding the thresholds for each feature
    value_array = np.sort(np.unique(np.array(value_lst)))
    thresholds = np.divide(np.add(value_array[:-1], value_array[1:]), 2)
    # iterate through the thresholds array to find the threshold which provides maximum info gain
    for threshold in range(len(thresholds)):
        cond_entropy = compute_entropy_on_thresholds(df, index, thresholds[threshold],feature_range)
        if minimum_entropy > cond_entropy:
            minimum_entropy = cond_entropy
            optimal_threshold = thresholds[threshold]

    return minimum_entropy, optimal_threshold

'''
Finds the conditional entropy for nominal value
'''
def find_best_nominal_feature(train_df, index, attribute_range,feature_range):
    data_labels = []
    nominal_entropy = 0

    if len(train_df) == 0:
        return 0, 0
    for diff_val in attribute_range:
        subset1 = []
        for train_instance in train_df:
            if str(train_instance[index], 'utf-8') == diff_val:
                subset1.append(train_instance)

        for sub_subset in subset1:
            data_labels.append(sub_subset[-1])

        nominal_probability = 1.0 * len(data_labels) / len(train_df)
        nominal_entropy += nominal_probability * compute_entropy_on_labels(np.array(subset1),feature_range)

        del data_labels[:]

    return nominal_entropy, 0

'''
Method to calculate conditional entropy for all the features
'''
def compute_conditional_entropy(metadata, used_feature, dataset,feature_range):
    # number of features in the dataset
    num_features = len(metadata.types()) - 1
    # array to store the entropy value for each of the features, used to decide which feature to split
    feature_entropy = np.zeros(shape=num_features)
    opt_threshold = np.zeros(shape=num_features)
    # iterate through the features
    for i in range(num_features):

        if used_feature[i] == True:
            continue

        if metadata.types()[i] == 'numeric':
            min_entropy, opt_thresh = find_best_numeric_feature(dataset, i,feature_range)
            opt_threshold[i] = opt_thresh
        else:
            min_entropy, opt_thresh = find_best_nominal_feature(dataset, i, feature_range[i],feature_range)
            opt_threshold[i] = opt_thresh

        feature_entropy[i] = min_entropy

    return feature_entropy, opt_threshold

'''
calculate H(Y)
'''
def compute_entropy_on_labels(data,feature_range):

    label_range = feature_range[-1]
    # check if the labels aree binary
    if len(label_range) != 2:
        sys.exit('labels are not binary')
    # both variables used to calculate H(Y)
    count_positive = 0
    count_negative = 0

    # iterate through each training instance
    for training in data:
        # increment counter
        t = training[-1].decode()
        if t == label_range[0]:
            count_negative += 1
        else:
            count_positive += 1

    if (count_negative == len(data) or count_negative == 0) or (count_positive == len(data) or count_positive == 0):
        return 0
    else:
        probability_negative = 1.0 * count_negative / len(data)
        probability_positive = 1.0 * count_positive / len(data)
        entropy_value = (-probability_negative * np.log2(probability_negative)) + (
                -probability_positive * np.log2(probability_positive))

    return entropy_value

'''
Splits the numeric dataset into 2 halves for building decision tree
'''
def split_numeric_data(data_tree, split_thresh, b_index):
    left_subtree = []
    right_subtree = []

    for instance in data_tree:
        if instance[b_index] <= split_thresh:
            left_subtree.append(instance)
        else:
            right_subtree.append(instance)
    return [left_subtree, right_subtree]

'''
splits the nominal dataset into subsets depending on the feature range
'''
def split_nominal_data(nominal_treedata, bst_ind, nominal_range):
    combined_data = []
    for nominal_feature in nominal_range:
        nominal_attributes = []
        for nominal_instance in nominal_treedata:

            if str(nominal_instance[bst_ind], 'utf-8') == nominal_feature:
                nominal_attributes.append(nominal_instance)

        combined_data.append(nominal_attributes)
    return combined_data

'''
stopping criteria for building tree
'''
def stop_growing(new_data, info, attributes_used, m):
    flag = 0
    labels_lst = []
    for data_instance in new_data:
        labels_lst.append(data_instance[-1])

    unique_labels = set(labels_lst)

    if len(unique_labels) == 1:
        flag = 1

    if flag == 1 or len(new_data) < m or all(attributes_used) or all(np.equal(info, 0)):
        return True

    return False

'''
calculates info gain 
'''
def information_gain(training_data,features_remaining,feature_range,metadata):
    # function call to compute H(Y)

    class_label_entropy = compute_entropy_on_labels(training_data,feature_range)

    # function call to compute H(Y|feature)
    c_en, opt_t = compute_conditional_entropy(metadata, features_remaining, training_data,feature_range)

    info_gain = np.subtract(class_label_entropy, c_en)

    return info_gain, opt_t

'''
gets the feature with maximum info gain
'''
def get_max_info_gain(training_d, remain_f,feature_range,metadata,flag):
    get_infogain, get_thresh = information_gain(training_d,remain_f,feature_range,metadata)
    if flag == 'check':
        return get_infogain
    elif flag == 'calculate':
        feature_index = np.argmax(get_infogain)
        return feature_index

'''
counts the number of positive and negative labels
'''
def num_labels(labeled_data, feature_range):
    neg_count = 0
    pos_count = 0

    for labeled_instance in labeled_data:
        if str(labeled_instance[-1], 'utf-8') == feature_range[-1][0]:
            neg_count += 1
        else:
            pos_count += 1

    return neg_count, pos_count

'''
creates node for the tree
'''
def create_node(d, attribute_name, attribute_threshold, features_used,feature_range,tree_parent, root, leaf, left):
    f_name = attribute_name
    f_thresh = attribute_threshold
    parent = tree_parent
    label_range = feature_range[-1]
    label_name = ''
    node = dtreeNode()
    node.set_featureName_featureThresh(f_name, f_thresh)
    node.update_feature_array(features_used)
    node.set_parent(parent)
    neg_count, pos_count = num_labels(d, feature_range)

    if neg_count > pos_count:
        label_name = label_range[0]
    elif neg_count < pos_count:
        label_name = label_range[1]
    else:
        label_name = tree_parent.get_label_name()

    node.set_root(root)
    node.set_label_pos_count(pos_count)
    node.set_label_neg_count(neg_count)
    node.set_label_name(label_name)
    if leaf:
        node.set_leaf()
    if left:
        node.set_left()

    return node


'''
Builds the decision tree
'''


def build_tree(new_data, feature_name, feature_threshold, remaining_features, parent,m,feature_range,metadata,root=False, leaf=False,
               left=False):

    check_info_gain = get_max_info_gain(new_data, remaining_features,feature_range,metadata,'check')

    if stop_growing(new_data, check_info_gain, remaining_features,m):

        leaf = True
        root = True
        return create_node(new_data, feature_name, feature_threshold, remaining_features,feature_range,parent, root, leaf, left)
    else:

        leaf = False
        root = True
        # retrieve the index for the feature with maximum info gain
        best_feature_index = get_max_info_gain(new_data, remaining_features,feature_range,metadata,'calculate')

        node = create_node(new_data, feature_name, feature_threshold, remaining_features,feature_range,parent, True, leaf, left)

        feature_name = metadata.names()[best_feature_index]
        # check if the feature is a numeric feature
        if metadata.types()[best_feature_index] == 'numeric':

            # retrieve the best threshold value for the feature
            entrpy_feature, best_feature_thresh = find_best_numeric_feature(new_data, best_feature_index,feature_range)


            # splitting the data based on the best threshold value for the feature
            [left_subtree_data, right_subtree_data] = split_numeric_data(new_data, best_feature_thresh,
                                                                         best_feature_index)

            node_left = build_tree(left_subtree_data, feature_name, best_feature_thresh, remaining_features, node,m,feature_range,
                                   metadata,root,False, True)
            node.set_child(node_left)

            node_right = build_tree(right_subtree_data, feature_name, best_feature_thresh, remaining_features, node,m,feature_range,
                                    metadata,root, False, False)
            node.set_child(node_right)

        else:

            subdivided_data = split_nominal_data(new_data, best_feature_index, feature_range[best_feature_index])

            num_features = len(feature_range[best_feature_index])

            for i in range(num_features):


                sub_node = build_tree(subdivided_data[i], feature_name, feature_range[best_feature_index][i],
                                      remaining_features, node,m,feature_range,metadata,root, False, False)
                node.set_child(sub_node)

    return node


def printing(feature_node,metadata,level=-1):
    # checking if the node is a root node
    if feature_node.isRoot() == True:

        if feature_node.getFeatureName() != None:
            if metadata[feature_node.getFeatureName()][0] == 'numeric':
                if feature_node.left == True:
                    sign = '<='
                    if feature_node.leaf == True:
                        print(level * '|\t' + '{} {} {:0.6f} [{} {}]: {}'.format(feature_node.getFeatureName(), sign,
                                                                                 feature_node.getFeatureType(),
                                                                                 feature_node.get_label_neg_count(),
                                                                                 feature_node.get_label_pos_count(),
                                                                                 feature_node.get_label_name()))

                    else:
                        print(level * '|\t' + '{} {} {:0.6f} [{} {}]'.format(feature_node.getFeatureName(), sign,
                                                                             feature_node.getFeatureType(),
                                                                             feature_node.get_label_neg_count(),
                                                                             feature_node.get_label_pos_count()))

                else:
                    sign = '>'
                    if feature_node.leaf == True:
                        print(level * '|\t' + '{} {} {:0.6f} [{} {}]: {}'.format(feature_node.getFeatureName(), sign,
                                                                                 feature_node.getFeatureType(),
                                                                                 feature_node.get_label_neg_count(),
                                                                                 feature_node.get_label_pos_count(),
                                                                                 feature_node.get_label_name()))
                    else:
                        print(level * '|\t' + '{} {} {:0.6f} [{} {}]'.format(feature_node.getFeatureName(), sign,
                                                                             feature_node.getFeatureType(),
                                                                             feature_node.get_label_neg_count(),
                                                                             feature_node.get_label_pos_count()))
            else:
                sign = '='
                if feature_node.leaf == True:
                    print(level * '|\t' + '{} {} {} [{} {}]: {}'.format(feature_node.getFeatureName(), sign,
                                                                        feature_node.getFeatureType(),
                                                                        feature_node.get_label_neg_count(),
                                                                        feature_node.get_label_pos_count(),
                                                                        feature_node.get_label_name()))
                else:
                    print(level * '|\t' + '{} {} {} [{} {}]'.format(feature_node.getFeatureName(), sign,
                                                               feature_node.getFeatureType(),feature_node.get_label_neg_count(),
                                                                feature_node.get_label_pos_count()))

        level = level + 1

    for i in feature_node.children:
        printing(i,metadata,level)


def classifier(test, node,metadata,f_name):
    pred = ''
    if node.leaf:
        return node.get_label_name()

    for child in node.children:

        index = f_name.index(child.getFeatureName())
        if metadata.types()[index] == 'nominal':
            if str(test[index], 'utf-8') == child.getFeatureType():
                pred = classifier(test, child,metadata,f_name)
        else:
            if child.left == True:
                if test[index] <= child.getFeatureType():
                    pred = classifier(test, child,metadata,f_name)
            else:
                if test[index] > child.getFeatureType():
                    pred = classifier(test, child,metadata,f_name)
    return pred


def printing_classifier_result(testing, num_labels_count, dt_tree,metadata,f_name):
    for i in range(len(testing)):

        actual_label = testing[i][-1]
        y_pred = classifier(testing[i], dt_tree,metadata,f_name)
        if str(actual_label, 'utf-8') == y_pred:
            num_labels_count += 1
        print('{}: {}: {} {}: {}'.format(i + 1, 'Actual', str(actual_label, 'utf-8'), 'Predicted', y_pred))

    return num_labels_count


def feature_names(metadata):
    features = []
    for feature in metadata.names():
        features.append(feature)
    return features

def read_input_arguments():
    train = str(sys.argv[1])
    test = str(sys.argv[2])
    m = int(str(sys.argv[3]))

    return train, test, m

def read_training_data(training_file):
    # to store tuple of different values taken by the features
    diff_feature_values = []

    # reads in the data and metadata from training file
    feature_set, metadata = arff.loadarff(training_file)

    for feature_names in metadata.names():
        diff_feature_values.append(metadata[feature_names][1])

    # constructing boolean array to maintain a list of features that have been exhausted for info gain
    features_remaining = np.zeros(shape=len(metadata.types()) - 1, dtype=bool)

    return feature_set, diff_feature_values, features_remaining, metadata


'''
Main method to call the functions
'''