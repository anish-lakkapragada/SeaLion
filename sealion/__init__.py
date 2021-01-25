"""
SeaLion is a simple machine learning and data science library.

=============================================================

Designed with beginners in mind, it has rich documentation via Pydoc and algorithms that span from the most basic
to more modern approaches. It is meant to help beginners navigate it all, and the documentation not only explains t
he models and their respective functions but also what they are and when to use them. Emphasis was also put on creating
new functions to make it interesting for those who are just getting started and seasoned ml-engineers alike.

I hope you enjoy it!
- Anish Lakkapragada 2021
"""

from sealion.DimensionalityReduction import tSNE, PCA
from sealion.utils import one_hot, revert_one_hot, revert_softmax, confusion_matrix
from sealion.decision_trees import DecisionTree
from sealion.ensemble_learning import RandomForest, EnsembleClassifier
from sealion.unsupervised_clustering import KMeans, DBSCAN
from sealion.regression import LinearRegression, LogisticRegression, SoftmaxRegression, RidgeRegression, LassoRegression, ElasticNet, PolynomialRegression, ExponentialRegression
from sealion.nearest_neighbors import KNearestNeighbors

from cython_ensemble_learning import CythonEnsembleClassifier
from sealion.cython_knn import CythonKNN
from sealion.cython_naive_bayes import cy_MultinomialNaiveBayes, cy_GaussianNaiveBayes
from sealion.cython_tsne import cy_tSNE
from sealion.cython_unsupervised_clustering import CythonDBSCAN, CythonKMeans
from sealion.cython_decision_tree_functions import CythonTrainDecisionTraining as cython_training
from sealion.cython_decision_tree_functions import gini_impurity, probabilities, find_best_split, branch_data, tuplelize, anti_tupelize, chunk_predict, go_down_the_tree

