from scipy.io import arff
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
from Node import dtreeNode
from dt_learn import *
if __name__ == '__main__':
    training_file, testing_file, m = read_input_arguments()
    features, feature_range, features_left, metadata = read_training_data(training_file)
    #label_range = feature_range[-1]
    dt_tree = build_tree(features, None, None, features_left, None, m, feature_range, metadata)
    printing(dt_tree,metadata)
    testing_set, test_metadata = arff.loadarff(testing_file)
    f_name = feature_names(metadata)
    print('<Predictions for the Test Set Instances>')
    num_labels_count = printing_classifier_result(testing_set, 0, dt_tree,metadata,f_name)
    print('Number of correctly classified: {} Total number of test instances: {}'.format(num_labels_count,
                                                                                         len(testing_set)))