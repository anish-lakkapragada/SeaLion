"""
@author : Anish Lakkapragada
@date : 1-12-2021

KNearestNeighbors is another powerful ML algorithm. Easy to understand, but remarkably useful. Pretty fast, too.

"""

import numpy as np


class KNearestNeighbors:
    """
    Arguably the easiest machine learning algorithm to make and understand. Simply looks at the k closest points (you give these points)
    for a data point you want to predict on, and if the majority of the closest k points are class X, it will predict
    back class X. The k number is decided by you, and should be odd (what happens if there's a tie?). If used for regression
    (we support that), it will just take the average of all of their values. If you are going to use regression, make
    sure to set regression = True - otherwise you will get a very low score in the evaluate() method as it will
    assume it's for classification (KNNs typically are.)

    A great introduction into ML - maybe consider using this and then seeing if you can beat it with your own version
    from scratch.If you can - please send it on GitHub!

    Other than that, enjoy!
    """

    def __init__(self, k=5, regression=False):
        """
        :param k: number of points for a prediction used in the algorithm
        :param regression: are you using regression or classification (if classification - do nothing)
        """

        from .cython_knn import CythonKNN

        self.cython_knn_model = CythonKNN()
        self.k = k
        self.regression = regression

    def fit(self, x_train, y_train):
        """
        :param x_train: 2D training data
        :param y_train: 1D training labels
        :return:
        """
        self.cython_knn_model.fit(
            x_train, y_train, k=self.k, regression=self.regression
        )

    def predict(self, x_test):
        """
        :param x_test: 2D prediction data
        :return: predictions in 1D vector/list
        """
        return np.array(self.cython_knn_model.predict(x_test))

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: accuracy score (r^2 score if regression = True)
        """
        return self.cython_knn_model.evaluate(x_test, y_test)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions given by model, 1D vector/list
        :param y_test: actual labels, 1D vector/list

        Visualize the predictions and labels to see where the model is doing well and struggling.
        """
        import matplotlib.pyplot as plt

        plt.cla()
        if not self.regression:
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
        else:
            plt.plot(
                [_ for _ in range(len(y_pred))],
                y_pred,
                color="blue",
                label="predictions/y_pred",
            )  # plot all predictions in blue
            plt.plot(
                [_ for _ in range(len(y_test))],
                y_test,
                color="green",
                label="labels/y_test",
            )  # plot all labels in green
        plt.title("Prediction & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()
