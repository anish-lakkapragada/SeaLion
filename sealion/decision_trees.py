"""
@author : Anish Lakkapragada
@date : 1-10-2021

The DecisionTree() class is so awesome it deserves to be in a file of its own. Powered by Cython and parallel processing
it's remarkably fast and it also has a good amount of unprecedented functionality.
"""

from collections import defaultdict, Counter
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
import pandas as pd
import numpy as np
import math
import warnings

from sealion.cython_models import cython_decision_tree_functions
gini_impurity = cython_decision_tree_functions.gini_impurity
probabilities = cython_decision_tree_functions.probabilities
find_best_split = cython_decision_tree_functions.find_best_split
branch_data = cython_decision_tree_functions.branch_data
tuplelize, anti_tupelize = cython_decision_tree_functions.tuplelize, cython_decision_tree_functions.anti_tupelize
chunk_predict = cython_decision_tree_functions.chunk_predict
go_down_the_tree = cython_decision_tree_functions.go_down_the_tree
cython_training = cython_decision_tree_functions.CythonTrainDecisionTraining


class DecisionTree() :
    """
    Decision Trees are powerful algorithms that create a tree by looking at the data and finding what questions are best
    to ask? For example if you are trying to predict whether somebody has cancer it may ask, "Do they have cells with
    a size >= 5?" or "Is there any history of smoking or drinking in the family?" These algorithms can easily fit most
    datasets, and are very well capable of overfitting (luckily we offer some parameters to help with that.)

    If you are going to be using categorical features with this Decision Tree, make sure to one hot encode it. You can
    do that with the one_hot() function in the utils module.

    This Decision Tree class doesn't support regression, or any labels that have continuous values. It is only for
    classification tasks. The reason this is such is because most curves and lines can be formed or modeled better with
    regression algorithms (Linear, Polynomial, Ridge, etc. are all available in the regression module) or neural networks.
    Decision Trees will typically overfit on such tasks.

    ----
    Methods :

    __init__(max_branches = math.inf, min_samples = 1) :
        Both these two parameters prevent overfitting, each pretty simple and easy to understand.

        ->> max_branches are the maximum number of branches the tree can have. It can very from 4 to 1000, depending on
        the number of features and the complexity of the task. You may be wondering why the default is math.inf - it's basically
        saying that there is no threshold for how many branches there can be. This prevents overfitting as you would much
        rather have a model with 200 branches that makes a few errors on the training set but generalizes well instead
        of a model with 1000 branches that gets everything in the training set but a lot less in the test.

        ->> min_samples is the minimum number of samples a branch needs to continue to split. If a decision tree is
        overfitting, you may notice it will create a bunch of trees just to fit one or two data points. By putting min_samples
        to 5 for example, what you are doing is making sure that the tree is not just trying to satisfy every point
        at the expense of overfitting but rather generalize to the whole. If there are less samples in a branch than the
        min_samples parameter, that branch will not split and will become a leaf (node) with the most common label.

    fit(x_train, y_train) :
        ->> x_train is the data, must be in a python list/numpy array that is 2D ([[]] not [])
            ->> remember to one_hot() any categorical features
        ->> y_train is the data, must be in a python list/numpy array that is 1D.

    predict(x_test) :
        ->> x_test is the data you want to be predicted. It must be 2D.
        ->> this method uses parallel predicting to make things go faster

    evaluate(x_test, y_test) :
        ->> x_test is your testing data (2D)
        ->> y_test is your labels (1D)

    average_branches() :
        ->> we understand that finding a good value for the max_branches parameter can be tricky, so this method
        gives the average number of branches that your tree has with the current parameters. It is not **exact** but it
        is a good starting point for you to start tuning the model.

    return_tree() :
        ->> this method just returns the tree created.

    give_tree(tree) :
        ->> Here you are giving in a tree to this class for it to use. If you understand the general structure of how our
        tree is created you can try to manually try to create a tree (that's not really machine learning though) - but
        most likely you will use this to give the best tree you got from a random forest. There is a give_best_tree method
        in the Random Forest class that will give the best tree created and you can feed that tree you get into this
        model. The Decision Tree will use the tree given by you (unless you call .fit() again)

    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is the predictions (1D)
        ->> y_test is the labels (1D)
        ->> here you get a plot of the predictions and the labels plotted. It can let you know where the model is
        struggling (e.g. predicting a lot of 3s instead of 2s, etc.).
    """
    def __init__(self, max_branches = math.inf, min_samples = 1) :
        """
        :param max_branches: maximum number of branches for the decision tree
        :param min_samples: minimum number of samples in a branch for the branch to split
        """
        self.max_branches = max_branches
        self.min_samples = min_samples

    def _build_tree(self, data, labels, max_branches = math.inf):

        training = cython_training()
        tree = training.build_tree(data, labels, self.min_samples, max_branches)
        self.splits_to_most_common_labels = training.get_splits_to_MCLs()
        self.data_points_to_count = training.get_data_points_to_count()
        return tree


    def average_branches(self):
        """
        :return: an estimate of how many branches the decision tree has right now with its
        current parameters.
        """
        all_counts = list({count : data_point for data_point, count in self.data_points_to_count.items()}.keys())
        return sum(all_counts)/len(all_counts)

    def fit(self, x_train, y_train) :
        """
        :param x_train: training data - 2D (make sure to one_hot categorical features)
        :param y_train: training labels
        :return:
        """
        min_samples, max_branches = self.min_samples, self.max_branches

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")

        x_train, y_train = x_train.tolist(), y_train.tolist()
        self.tree  = DecisionTree._build_tree(self, x_train, y_train, max_branches=max_branches)

        try : #give a warning to user
            if isinstance(float(self.tree), float) :
                warnings.warn("Training failed (no tree created). \n Experiment with max_branches or min_samples "
                              "parameters.")
        except Exception : pass

    def return_tree(self):
        """
        :return: the tree inside (if you want to look at it)
        """
        return self.tree

    def give_tree(self, tree) :
        """
        :param tree: a tree made by you or given by give_best_tree in the RandomForest module
        :return: None, just that the tree you give is now the tree used by the decision tree.
        """
        self.tree = tree

    def predict(self, x_test) :
        """
        :param x_test: 2D prediction data.
        :return: predictions in 1D vector/list.
        """
        prediction_data = np.array(x_test)
        if len(prediction_data.shape) != 2 : raise ValueError("x_test must be 2D (even if only one sample.)")
        def parallel_predict(self, prediction_data):
            '''split to individual data frames, make it into python lists, run that through chunk_predict() method'''

            predictions = []
            with parallel_backend("threading", n_jobs = -1) :
                predictions.append(Parallel()(delayed(go_down_the_tree)(list(data_point), self.tree)  for data_point in
                                              prediction_data))
            return np.array(predictions)

        return parallel_predict(self, prediction_data).flatten()

    def evaluate(self, x_test, y_test):
        """
        :param x_test: 2D testing data.
        :param y_test: 1D testing labels.
        :return: accuracy score.
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D")
        x_test, y_test = x_test.tolist(), y_test.tolist()

        y_pred = DecisionTree.predict(self, x_test)
        amount_correct = sum([1 if pred == label else 0 for pred, label in zip(y_pred, y_test)])
        return (amount_correct)/len(y_pred)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions given by model
        :param y_test: actual labels
        :return: an image of the predictions and the labels.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        plt.scatter([_ for _ in range(len(y_pred))], y_pred, color="blue",
                 label="predictions/y_pred")  # plot all predictions in blue
        plt.scatter([_ for _ in range(len(y_test))], y_test, color="green",
                 label="labels/y_test")  # plot all labels in green
        plt.legend()
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.show()

