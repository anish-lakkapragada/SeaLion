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

from sealion import cython_ensemble_learning, cython_knn, cython_naive_bayes, cython_tsne, cython_unsupervised_clustering, cython_decision_tree_functions

from sealion.cython_ensemble_learning import CythonEnsembleClassifier
from sealion.cython_knn import CythonKNN
from sealion.cython_naive_bayes import cy_MultinomialNaiveBayes, cy_GaussianNaiveBayes
from sealion.cython_tsne import cy_tSNE
from sealion.cython_unsupervised_clustering import CythonDBSCAN, CythonKMeans

cython_training = cython_decision_tree_functions.CythonTrainDecisionTraining
gini_impurity = cython_decision_tree_functions.gini_impurity
probabilities = cython_decision_tree_functions.probabilities
find_best_split = cython_decision_tree_functions.find_best_split
branch_data = cython_decision_tree_functions.branch_data
tuplelize = cython_decision_tree_functions.tuplelize
anti_tupelize = cython_decision_tree_functions.anti_tupelize
chunk_predict = cython_decision_tree_functions.chunk_predict
go_down_the_tree = cython_decision_tree_functions.go_down_the_tree

from sealion import DimensionalityReduction, utils, decision_trees, ensemble_learning, unsupervised_clustering, regression, nearest_neighbors, neural_networks

from sealion.DimensionalityReduction import tSNE, PCA
from sealion.utils import one_hot, revert_one_hot, revert_softmax, confusion_matrix
from sealion.decision_trees import DecisionTree
from sealion.ensemble_learning import RandomForest, EnsembleClassifier
from sealion.unsupervised_clustering import KMeans, DBSCAN
from sealion.regression import LinearRegression, LogisticRegression, SoftmaxRegression, RidgeRegression, LassoRegression, ElasticNet, PolynomialRegression, ExponentialRegression
from sealion.nearest_neighbors import KNearestNeighbors

from sealion.neural_networks.optimizers import GD, Momentum, SGD, AdaGrad, RMSProp, Adam
from sealion.neural_networks.layers import Dense, Flatten, Dropout, ReLU, ELU, SELU, LeakyReLU, Sigmoid, Tanh, Swish, Softmax
from sealion.neural_networks.models import NeuralNetwork, NeuralNetwork_MapReduce
from sealion.neural_networks.loss import CrossEntropy, MSE, softmax