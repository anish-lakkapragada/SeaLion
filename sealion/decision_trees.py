"""
@author : Anish Lakkapragada
@date : 1-10-2021

The DecisionTree() class is so awesome it deserves to be in a file of its own. Powered by Cython and parallel processing
it's remarkably fast and it also has a good amount of unprecedented functionality.
"""

from joblib import Parallel, delayed, parallel_backend
import numpy as np
import math
import warnings


class DecisionTree:
    """
    Decision Trees are powerful algorithms that create a tree by looking at the data and finding what questions are best
    to ask? For example if you are trying to predict whether somebody has cancer it may ask, "Do they have cells with
    a ``size >= 5``?" or "Is there any history of smoking or drinking in the family?" These algorithms can easily fit most
    datasets, and are very well capable of overfitting (luckily we offer some parameters to help with that.)

    If you are going to be using categorical features with this Decision Tree, make sure to one hot encode it. You can
    do that with the ``one_hot()`` function in the utils module.

    This Decision Tree class doesn't support regression, or any labels that have continuous values. It is only for
    classification tasks. The reason this is such is because most curves and lines can be formed or modeled better with
    regression algorithms (Linear, Polynomial, Ridge, etc. are all available in the regression module) or neural networks.
    Decision Trees will typically overfit on such tasks.
    """

    def __init__(self, max_branches=math.inf, min_samples=1):
        """
        :param max_branches: maximum number of branches for the decision tree
        :param min_samples: minimum number of samples in a branch for the branch to split
        """

        from .cython_decision_tree_functions import (
            go_down_the_tree,
            CythonTrainDecisionTraining,
        )

        self.go_down_the_tree = go_down_the_tree
        self.cython_training = CythonTrainDecisionTraining
        self.max_branches = max_branches
        self.min_samples = min_samples

    def _build_tree(self, data, labels, max_branches=math.inf):

        training = self.cython_training()
        tree = training.build_tree(data, labels, self.min_samples, max_branches)
        self.splits_to_most_common_labels = training.get_splits_to_MCLs()
        self.data_points_to_count = training.get_data_points_to_count()
        return tree

    def average_branches(self):
        """
        :return: an estimate of how many branches the decision tree has right now with its
            current parameters.
        """
        all_counts = list(
            {
                count: data_point
                for data_point, count in self.data_points_to_count.items()
            }.keys()
        )
        return sum(all_counts) / len(all_counts)

    def _preprocess_row(row):
        list_row = np.array(row).tolist()
        new_row = []
        for element in list_row:
            try:
                list_row = list_row.tolist()
                new_row.append(list_row)
            except Exception:
                new_row.append(element)
        return new_row

    def fit(self, x_train, y_train):
        """
        :param x_train: training data - 2D (make sure to one_hot categorical features)
        :param y_train: training labels
        :return:
        """
        min_samples, max_branches = self.min_samples, self.max_branches

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2:
            raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1:
            raise ValueError("y_train must be 1D.")

        x_train = [DecisionTree._preprocess_row(row) for row in x_train]
        y_train = y_train.tolist()

        self.tree = DecisionTree._build_tree(
            self, x_train, y_train, max_branches=max_branches
        )

        try:  # give a warning to user
            if isinstance(float(self.tree), float):
                warnings.warn(
                    "Training failed (no tree created). \n Experiment with max_branches or min_samples "
                    "parameters."
                )
        except Exception:
            pass

    def return_tree(self):
        """
        :return: the tree inside (if you want to look at it)
        """
        return self.tree

    def give_tree(self, tree):
        """
        :param tree: a tree made by you or given by give_best_tree in the RandomForest module
        :return: None, just that the tree you give is now the tree used by the decision tree.
        """
        self.tree = tree

    def predict(self, x_test):
        """
        :param x_test: 2D prediction data.
        :return: predictions in 1D vector/list.
        """
        prediction_data = np.array(x_test)
        if len(prediction_data.shape) != 2:
            raise ValueError("x_test must be 2D (even if only one sample.)")

        def parallel_predict(self, prediction_data):
            """split to individual data frames, make it into python lists, run that through chunk_predict() method"""

            predictions = []
            with parallel_backend("threading", n_jobs=-1):
                predictions.append(
                    Parallel()(
                        delayed(self.go_down_the_tree)(list(data_point), self.tree)
                        for data_point in prediction_data
                    )
                )
            return np.array(predictions)

        return parallel_predict(self, prediction_data).flatten()

    def evaluate(self, x_test, y_test):
        """
        :param x_test: 2D testing data.
        :param y_test: 1D testing labels.
        :return: accuracy score.
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2:
            raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1:
            raise ValueError("y_test must be 1D")
        x_test, y_test = x_test.tolist(), y_test.tolist()

        y_pred = DecisionTree.predict(self, x_test)
        amount_correct = sum(
            [1 if pred == label else 0 for pred, label in zip(y_pred, y_test)]
        )
        return (amount_correct) / len(y_pred)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions given by model
        :param y_test: actual labels
        :return: an image of the predictions and the labels.
        """
        import matplotlib.pyplot as plt

        plt.cla()
        plt.scatter(
            [_ for _ in range(len(y_pred))],
            y_pred,
            color="blue",
            label="predictions/y_pred",
        )  # plot all predictions in blue
        plt.scatter(
            [_ for _ in range(len(y_test))],
            y_test,
            color="green",
            label="labels/y_test",
        )  # plot all labels in green
        plt.legend()
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.show()
