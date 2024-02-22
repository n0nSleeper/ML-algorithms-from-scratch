import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2 as cv
from collections import Counter

class Node:
    def __init__(self, feature = None, left = None, right = None, threshold = None,*, value = None ):
        self.feature = feature
        self.left = left 
        self.right = right
        self.threshold = threshold
        self.value = value 

    def _is_leaf(self):
        return self.value != None
    
class Decision_Tree:
    def __init__ (self, min_sample_split = 2, max_depth = 10, n_features = None):
        self.min_sample_split = min_sample_split 
        self.max_depth = max_depth 
        self.n_features = n_features

        self.root = None


    def fit(self, X, Y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, Y, depth = 0)

    def _grow_tree(self, X, Y, depth):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(Y))

        #terminate conditions: 
        if (depth >= self.max_depth or n_samples < self.min_sample_split or  n_labels == 1):
            leaf_value = self._most_common(Y)
            return Node(value = leaf_value)
        
        features_idxs = np.random.choice(n_feats, self.n_features, replace= False)

        best_feature, best_threshold = self._best_split(X, Y, features_idxs)

        left_idx, right_idx = self._split(X, best_feature, best_threshold)

        left = self._grow_tree(X[left_idx, :], Y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx, :], Y[right_idx], depth+1)

        return Node(best_feature, left, right, best_threshold)


    def _most_common(self, Y):
        counter = Counter(Y)
        return counter.most_common(1)[0][0]

    def _best_split(self, X, Y, feature_idxs):
        best_feature_idx = None
        best_threshold = None 
        best_information_gain = -1

        for feature_idx in feature_idxs:
            thresholds = np.unique(X[:, feature_idx])

            for thr in thresholds:
                information_gain = self._information_gain(X, Y, thr, feature_idx)

                if (information_gain > best_information_gain):
                    best_information_gain = information_gain
                    best_feature_idx = feature_idx
                    best_threshold = thr 
        return best_feature_idx, best_threshold
    
    def _information_gain(self, X, Y, threshold, feature_idx):
        
        #parent entropy 
        parent_entropy = self._entropy(Y)

        #weighted children entropy 

        left_idxs, right_idxs = self._split(X, feature_idx, threshold)
        n_left_samples = len(left_idxs)
        n_right_samples = len(right_idxs)

        if (n_left_samples == 0 or n_right_samples == 0):
            return 0

        left_child_entropy = self._entropy(Y[left_idxs])
        right_child_entropy = self._entropy(Y[right_idxs])

        #information gain
        IG = parent_entropy - (n_left_samples/len(Y))*left_child_entropy - (n_right_samples/len(Y))*right_child_entropy
        return IG 

    def _entropy(self, Y):
        hist = np.bincount(Y)
        Py = hist/len(Y)
        entropy = -np.sum([py*np.log(py) for py in Py if py > 0])
        return entropy 
    
    def _split(self, X, feature_idx, threshold):
        left = np.argwhere(X[:, feature_idx] <= threshold).flatten()
        right = np.argwhere(X[:, feature_idx] > threshold).flatten ()

        return left, right 

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return predictions
    
    def _traverse_tree(self, x, node):
        if node._is_leaf():
            return node.value 
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

             
        

    
