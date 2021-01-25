"""
@author : Anish Lakkapragada
@date : 1-10-2021

Ensemble Learning is an often overshadowed and underestimated field of machine learning. Here we provide 2 algorithms
central to the game - random forests and ensemble/voting classifier. Random Forests are very especially fast
with parallel processing to fit multiple decision trees at the same time.
"""

import pandas as pd
import numpy as np
from multiprocessing import cpu_count
from joblib import parallel_backend, delayed, Parallel
import random
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class RandomForest():
    """
    Random Forests may seem intimidating but they are super simple. They are just a bunch of Decision Trees that are
    trained on different sets of the data. You give us the data, and we will create those different sets. You may choose
    for us to sample data with replacement or without, either way that's up to you. Keep in mind that because this is
    a bunch of Decision Trees, classification is only supported (avoid using decision trees for regression - it's
    range of predictions is limited.) The random forest will have each of its decision trees predict on data and just
    choose the most common prediction (not the average.)

    Enjoy this module - it's one of our best.

    ----
    Methods :

    __init__(num_classifiers = 20, max_branches = math.inf, min_samples = 1, replacement = True, min_data = None) :
        WOAH - that's a lot of parameters. Let's go over them - it's all simpler than you may think.
        ->> num classifiers : just the number of decision trees the random forest will have.
        ->> max_branches (check the DecisionTree documentation for this) : a parameter for each decision tree on the
         max number of branches it can have. math.inf just means there is no such boundary.
        ->> min_samples (check the DecisionTree docs) : a parameter on the minimum samples a branch in a decision tree
        can have.
        ->> replacement : whether or not you want each set of data to share data with any other sets. For every classifier/decision tree
        there will be a dataset created for it to train on. If replacement = True, then we will sample a random number of points
        from the data and be fine if each dataset shares some points. If False, then the data is divided into num_classifiers chunks,
        and each classifier/tree is trained on a dataset.
        ->> min_data is a parameter for the minimum number of data needed for any dataset. Because if replacement = True
        the datasets will have a random number of points - if you want to have a minimum number of points in each dataset
        you can set that here. It's default is 50% of the data given (the None is just because we don't know the amount of
        data yet.)

    fit(x_train, y_train) :
        Uses parallel processing to train multiple trees simultaneously.
        ->> x_train must be 2D ([[]] not []) and is your data
        ->> y_train is your labels and must be 1D

    predict(x_test) :
        ->> x_test is your prediction data (2D)

    evaluate(x_test, y_test) :
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)

    give_best_tree(x_test, y_test) :
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)
        ->> returns the tree that performed best on this data. You can then use that tree inside the DecisionTree class.
        Just do DecisionTree().give_tree(RandomForest.give_best_tree(x_test, y_test))

    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.

    """
    def __init__(self, num_classifiers= 20, max_branches= math.inf, min_samples=1, replacement=True, min_data=None):
        """
        :param num_classifiers: Number of decision trees you want created.
        :param max_branches: Maximum number of branches each Decision Tree can have.
        :param min_samples: Minimum number of samples for a branch in any decision tree (in the forest) to split.
        :param replacement: Whether or not any of the data points in different chunks/sets of data can overlap.
        :param min_data: Minimum number of data there can be in any given data chunk. Each classifier is trained on a
        chunk of data, and if you want to make sure each chunk has 3 points for example you can set min_data = 3. It's
        default is 50% of the amount of data, the None is just a placeholder.
        """
        self.trees = []
        self.num_classifiers = num_classifiers
        self.max_branches = max_branches
        self.min_samples = min_samples
        self.replacement = replacement
        self.min_data = min_data

    def fit(self, x_train, y_train):
        """
        :param x_train: 2D training data
        :param y_train: 1D training labels
        :return:
        """
        data, labels = np.array(x_train).tolist(), np.array(y_train).tolist()
        num_classifiers = self.num_classifiers
        max_branches = self.max_branches
        min_samples = self.min_samples
        replacement = self.replacement
        min_data = self.min_data

        # on default set min_data = 50% of your dataset
        if not min_data: min_data = round(0.5 * len(data))

        # merge data and labels together [(d1, l1) .. (dN, lN)]
        data_and_labels = [(data_point, label) for data_point, label in zip(data, labels)]


        self.chunk_data, self.chunk_labels = [], []
        if replacement:
            for classifier in range(num_classifiers):
                num_samples = min_data + random.randint(0, len(data) - min_data)
                data_and_labels_set = random.sample(data_and_labels, num_samples)
                self.chunk_data.append([data_point for data_point, _ in data_and_labels_set])
                self.chunk_labels.append([label for _, label in data_and_labels_set])
        else:
            '''no replacement just use up all of the data here'''
            data_and_labels_df = pd.DataFrame({"data": data, "labels": labels})
            data_and_labels_full_set = np.array_split(data_and_labels_df,
                                                      num_classifiers)  # splits into num_classifiers dataframes
            for df in data_and_labels_full_set:
                self.chunk_data.append(np.array(df)[:, 0].flatten())
                self.chunk_labels.append(np.array(df)[:, 1].flatten())

        self.trees = []
        from joblib import parallel_backend
        with parallel_backend('threading', n_jobs = -1):
            Parallel()(delayed(RandomForest._train_new_tree)(self, data_chunk, label_chunk) for data_chunk, label_chunk
                       in zip(self.chunk_data, self.chunk_labels))

        self.decision_trees = []  # stores each tree in a decision tree class
        for tree in self.trees:
            dt = DecisionTree()
            dt.give_tree(tree)
            assert dt.tree == tree
            self.decision_trees.append(dt)

    def _train_new_tree(self, data, labels):
        dt = DecisionTree(max_branches = self.max_branches, min_samples = self.min_samples)
        dt.fit(data, labels)
        self.trees.append(dt.tree)

    def predict(self, x_test):
        """
        :param x_test: testing data (2D)
        :return: Predictions in 1D vector/list.
        """
        predictions = np.zeros(len(x_test))
        for decision_tree in self.decision_trees:
            predictions += decision_tree.predict(x_test)
        return np.round_(predictions / len(self.decision_trees))

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: accuracy score
        """
        y_pred = RandomForest.predict(self, x_test)
        amount_correct = sum([1 if pred == label else 0 for pred, label in zip(y_pred, y_test)])
        return amount_correct / len(x_test)

    def give_best_tree(self, x_test, y_test):
        """
        You give it the data and the labels, and it will find the tree in the forest that does the best. Then it will
        return that tree. You can then take that tree and put it into the DecisionTree class using the give_tree method.
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: tree that performs the best (dictionary data type)
        """
        evaluations = {decision_tree.evaluate(x_test, y_test): decision_tree for decision_tree in self.decision_trees}
        return evaluations[max(evaluations)].tree  # tree with best score

    def _evaluate_decision_tree(data_labels_tree):
        '''give the decision tree'''
        x_test, y_test, decision_tree = data_labels_tree
        dt = DecisionTree()
        dt.give_tree(decision_tree)
        return dt.evaluate(x_test, y_test)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """

        import matplotlib.pyplot as plt
        plt.cla()
        y_pred, y_test = y_pred.flatten(), y_test.flatten()
        plt.scatter([_ for _ in range(len(y_pred))], y_pred, color="blue", label="predictions/y_pred")
        plt.scatter([_ for _ in range(len(y_test))], y_test, color="green", label="labels/y_test")
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()

def _train_new_predictor(prediction_dataset):
    predictor, name, x_train, y_train = prediction_dataset
    predictor.fit(x_train, y_train)
    return [name, predictor]

class EnsembleClassifier() :
    """
    Aside from random forests, voting/ensemble classifiers are also another popular way of ensemble learning. How it works
    is by training multiple different classifiers (you choose!) and predicting the most common class (or the average for
    regression - more on that later.) Pretty simple actually, and works quite effectively. This module also can
    tell you the best classifier in a group with its get_best_predictor(), so that could be useful. Similar to
    give_best_tree() in the random forest module, what it does is give the class of the algorithm that did the best on the
    data you gave it. This can also be used for rapid hypertuning on the exact same module (giving the same class but
    with different parameters in the init.)

    Example :
    >>> from sealion.regression import SoftmaxRegression
    >>> from sealion.naive_bayes import GaussianNaiveBayes
    >>> from sealion.nearest_neighbors import KNearestNeighbors
    >>> ec = EnsembleClassifier({'algo1' : SoftmaxRegression(num_classes = 3), 'algo2' : GaussianNaiveBayes(), 'algo3' : KNearestNeighbors()}, classification = True)
    >>> ec.fit(X_train, y_train)
    >>> y_pred = ec.predict(X_test) #predict
    >>> ec.evaluate_all_predictors(X_test, y_test)
    algo1 : 95%
    algo2 : 90%
    algo3 : 75%
    >>> best_predictor = ec.get_best_predictor(X_test, y_test) #get the best predictor
    >>> print(best_predictor) #is it Softmax Regression, Gaussian Naive Bayes, or KNearestNeighbors that did the best?
    <regression.SoftmaxRegression object at 0xsomethingsomething>
    >>> y_pred = best_predictor.predict(X_test) #looks like softmax regression, let's use it


    Here we first important all the algorithms we are going to be using from their respective modules. Then
    we create an ensemble classifier by passing in a dictionary where each key stores the name, and each value stores
    the algorithm. Classification = True by default, so we didn't need to put that (if you want regression put it to
    False. A good way to remember classification = True is the default is that this is an EnsembleCLASSIFIER.)

    We then fitted that and got it's predictions. We saw how well each predictor did (that's where the names come in)
    through the evaluate_all_predictors() method. We could then get the best predictor and use that class. Note that
    this class will ONLY use algorithms other than neural networks, which should be plenty. This is because neural networks
    have a different evaluate() method and typically will be more random in performance than other algorithms.

    I hope that example cleared anything up. The fit() method trains in parallel (thanks joblib!) so it's pretty
    fast. As usual, enjoy this algorithm!

    ----
    Methods

    __init__(predictors, classification = True)
        ->> predictors was the dictionary with the names of the algorithms and their classes that will be trained
        ->> classification is whether the task is classification or regression. If regression, set this to False.
        This is useful to figure out whether or not to round or average the predictions.

    fit(x_train, y_train) :
        Uses parallel processing to train multiple trees simultaneously.
        ->> x_train must be 2D ([[]] not []) and is your data
        ->> y_train is your labels and must be 1D

    predict(x_test) :
        ->> x_test is your prediction data (2D)

    evaluate(x_test, y_test) :
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)

    evaluate_all_predictors(x_test, y_test) :
        ->> print out the name of each classifier and its score on the x_test (data) and y_test (labels) given

    get_best_predictor(x_test, y_test) :
        ->> returns the class of the algorithm that does the best on the data you give here
        ->> check the above example for how that may work

    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.

    """
    def __init__(self, predictors, classification = True):
        """
        :param predictors: dict of {name (string) : algorithm (class)}. See example above.
        :param classification: is it a classification or regression task? default classification - if regression set this
        to False.
        """
        self.classification = classification
        self.cython_ensemble_classifier = cython_ensemble_learning.CythonEnsembleClassifier(predictors, self.classification)
    def fit(self, x_train, y_train) :
        """
        :param x_train: 2D training data
        :param y_train: 1D training labels
        :return:
        """
        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")
        self.predictors = self.cython_ensemble_classifier.get_predictors()

        with parallel_backend('threading', n_jobs=cpu_count()):
            for name, predictor in self.predictors.items() :
                self.predictors[name].fit(x_train, y_train)

        self.trained_predictors = self.predictors
        self.cython_ensemble_classifier.give_trained_predictors(self.trained_predictors)
    def predict(self, x_test):
        """
        :param x_test: testing data (2D)
        :return: Predictions in 1D vector/list.
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return self.cython_ensemble_classifier.predict(x_test)
    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: accuracy score
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")
        return self.cython_ensemble_classifier.evaluate(x_test, y_test)
    def evaluate_all_predictors(self, x_test, y_test) :
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: None, just prints out the name of each algorithm in the predictors dict fed to the __init__ and its
        score on the data given.
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")
        return self.cython_ensemble_classifier.evaluate_all_predictors(x_test, y_test)
    def get_best_predictor(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: the class of the algorithm that did best on the given data. look at the above example if this doesn't
        make sense.
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")
        return self.cython_ensemble_classifier.get_best_predictor(x_test, y_test)
    def visualize_evaluation(self, y_pred, y_test) :
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        y_pred, y_test = y_pred.flatten(), y_test.flatten()
        if self.classification :
            plt.scatter([_ for _ in range(len(y_pred))], y_pred, color="blue", label="predictions/y_pred")
            plt.scatter([_ for _ in range(len(y_test))], y_test, color="green", label="labels/y_test")
        else :
            plt.plot([_ for _ in range(len(y_pred))], y_pred, color="blue", label="predictions/y_pred")
            plt.plot([_ for _ in range(len(y_test))], y_test, color="green", label="labels/y_test")
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()







