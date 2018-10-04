import numpy as np


class dtreeNode():

    def __init__(self, feature_name=None, feature_type=None,threshold=None, parent=None, feature_used=None, leaf=False, left=False):

        # name of the feature
        self.feature_name = feature_name
        # type of feature = {numeric,nominal}
        self.feature_type = feature_type
        # threshold values for numeric feature
        self.threshold = threshold
        # parent node
        self.parent = parent
        # array maintaining the features used for candidate splits
        self.feature_used = feature_used
        # check if the node is a leaf node
        self.leaf = leaf
        # children, for nominal features
        self.children = []
        # root node
        self.root = False
        # helps to build the tree, if true build from left or build from right
        self.left = left

        self.labels_pos_count = 0

        self.labels_neg_count = 0

        self.label_name = ' '

    def getFeatureName(self):
        return self.feature_name

    def getFeatureType(self):
        return self.feature_type

    def getThreshold(self):
        return self.threshold

    def get_label_pos_count(self):
        return self.labels_pos_count

    def get_label_neg_count(self):
        return self.labels_neg_count

    def get_label_name(self):
        return self.label_name

    def isRoot(self):
        if self.root:
            return True
        else:
            return False

    def isLeaf(self):
        if self.leaf:
            return True
        else:
            return False

    def get_child(self):
        return self.children

    def get_left(self):
        return self.left

    def set_parent(self, p):
        self.parent = p

    def set_label_pos_count(self, pos_count):
        self.labels_pos_count = pos_count

    def set_label_neg_count(self,neg_count):
        self.labels_neg_count = neg_count

    def set_label_name(self, name):
        self.label_name = name

    def update_feature_array(self, used):
        self.feature_used = used

    def set_child(self, ch):
        self.children.append(ch)

    def set_leaf(self):
        self.leaf = True

    def set_root(self, r):
        self.root = r

    def set_left(self):
        self.left=True

    def set_featureName_featureThresh(self, fname, thresh):
        self.feature_type = thresh
        self.feature_name = fname
