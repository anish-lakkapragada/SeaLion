import numpy as np
from sealion.cython_models import cython_naive_bayes


class MultinomialNaiveBayes() :
    """
    Multinomial Naive Bayes is the most popular form of Naive Bayes there is. (Multinomial) Naive Bayes is well known for
    its usage in spam classifiers, and this specific module is just for that. You give your data,
    tokenized into numeric form (just means each word has a specific index), and you will get a
    spam classifier! You can then put your data in and get a prediction. Works for 2 or more categories/unique labels.
    Pretty fast with Cython.


    For example if your data was :

    [['get free money!'], ['buy 2 get 3'], ['Towards Data Science - New Article!']]

    And you tokenized it to :

    >>> data = [[1, 2, 3], [4, 5, 1, 6], [7, 8,9, 10, 11]]

    Your labels maybe (based on how "spammy" they are) :

    >>> labels = [1, 1, 0] # where 1 is spam and 0 is not

    You can feed this directly into this class like :
    >>> mnb = MultinomialNaiveBayes()
    >>> mnb.fit(data, labels)
    >>> mnb.predict() ...

    ----
    Methods
    fit(x_train, y_train) :
        ->> performs gradient descent
        ->> x_train is your training data (2D)
        ->> y_train is your training labels (1D)
    predict(x_test) :
        ->> x_test is your prediction data (2D)
        ->> returns the predictions (0s and 1s) in a numpy array (1D)
    evaluate(x_test, y_test) :
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)
        ->> will the percent of predictions on x_test it got correct
    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.
    """

    def __init__(self):
        """Init, no args."""
        self.inner_cython_mnb = cython_naive_bayes.cy_MultinomialNaiveBayes()
    def fit(self, x_train, y_train):
        """
        :param x_train: data (2D)
        :param y_train: labels (1D)
        :return: None
        """
        self.inner_cython_mnb.fit(x_train, y_train)
    def predict(self, x_test):
        """
        :param x_test: testing data (2D)
        :return: predictions in a 1D vector/list
        """
        return np.array(self.inner_cython_mnb.predict(x_test))
    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: Just what percent of predictions on x_test are actually correct
        """
        return self.inner_cython_mnb.evaluate(x_test, y_test)

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

class GaussianNaiveBayes() :
    """
    Gaussian Naive Bayes is the lesser known version of Naive Bayes. It is used for the standard numeric data you may
    feed into a model like logistic regression. What it does is create a normal (Gaussian) probability distribution for each
    feature so it can measure the chance that a data point belongs to certain class given that it has this or that feature.
    It will do this for every data point, for every class, and for every feature (not as efficient as Softmax Regression.)
    Just another cool algorithm.

    ----
    Methods :

    __init__() :
        ->> no params here
    fit(x_train, y_train) :
        ->> x_train is your training data (2D)
        ->> y_train is your training labels (1D)
    predict(x_test) :
        ->> x_test is your prediction data (2D)
        ->> will return all predictions in a 1-D vector/python list.
    evaluate(x_test, y_test) :
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)
        ->> will the percent of predictions on x_test it got correct
    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.
    """
    def __init__(self):
        self.inner_cython_gnb = cython_naive_bayes.cy_GaussianNaiveBayes()
    def fit(self, x_train, y_train):
        """
        :param x_train: training data (2D)
        :param y_train: training labels (1D)
        :return:
        """
        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")
        self.inner_cython_gnb.fit(x_train, y_train)
    def predict(self, x_test):
        """
        :param x_test: prediction data (2D)
        :return: predictions in a 1D vector/list
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return np.array(self.inner_cython_gnb.predict(x_test))
    def evaluate(self, x_test, y_test):
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")
        return self.inner_cython_gnb.evaluate(x_test, y_test)
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
