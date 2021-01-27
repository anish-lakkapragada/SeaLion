"""
@author : Anish Lakkapragada
@date : 1 - 17 - 2021

This module contains all of the regression modules you will need. Yes all. Stating from the bases of Linear Regression
and Logistic Regression, it works its way up to more advanced ones like Elastic Net and Softmax Regression. Also
has polynomial and exponential regression, along with some regularizers.
"""
import numpy as np
from tqdm import tqdm
import warnings

from sealion.utils import one_hot
warnings.filterwarnings('ignore', category = np.ComplexWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)

def r2_score(y_pred, y_test):
    num = np.sum(np.power(y_test - y_pred, 2))
    denum = np.sum(np.power(y_test - np.mean(y_test), 2))
    return 1 - num / denum


class LinearRegression():
    """
    While KNNs (K-Nearest-Neighbors) may be the simplest ML algorithms out there, Linear Regression is probably the one
    you heard first. You may have used it on a TI-84 before - all it does is it fit a line to the data. It does this through
    the gradient descent algorithm or a neat normal equation. We will use the normal equation as usually it works just the same
    and is faster, but for much larger datasets you should explore neural networks or dimensionality reduction (check the algos there.)
    The equation in school taught is y = mx + b, but we'll denote it as : y_hat = m1x1 + m2x2 ... mNxN + b.
    The hat is meant to resemble the predictions, and the reason we do it from m1 ... mN is because our data can be in N
    dimensions, not necessarily one.

    Some other things to know is that your data for x_train should always be 2D. 2D means that it is [[]]. This doesn't
    mean the data is necessarily 2D (this could look like [[1, 2], [2, 3]]) - but just means that its lists inside
    lists. y_train is your labels, or the "correct answers" so it should be in a 1D list, which is just a list. This
    library assumes just a bit of knowledge about this - and it isn't too difficult - so feel free to search this up.

    Another thing to note here is that for our library you can enter in numpy arrays of python lists, but you will always
    get numpy arrays back (standard practice with other libs, too.)

    A lot of the methods here are consistent and same to a lot of the other classes of the library, so reading this
    will make it a lot easier down the line.

    The goal of this module is it for it to be a useful algorithm, but I also hope this is inspiring to your journey
    of machine learning. It isn't nearly as hard as it seems.

    ----
    Methods

    fit(x_train, y_train) :
        ->> x_train is your training data (2D)
        ->> y_train is your training labels (1D)

    predict(x_test) :
        ->> x_test is your prediction data (2D)
        ->> you will get back a 1D numpy array of a prediction for every data point in x_test

    evaluate(x_test, y_test) :
        ->> this method is to evaluate the model on data it hasn't seen/trained on (not in x_train in the fit() method.)
        ->> reason you do this is because you want to see whether the model can generalize what it learnt on x_train
        to x_test.
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)
        ->> returns the r^2 score

    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.
    """

    def fit(self, x_train, y_train):
        """
        :param x_train: 2D training data
        :param y_train: 1D training labels
        :return:
        """
        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")


        #Closed form solution below :
        #Weights = (XT X)^-1 XT y
        #Intercept_vector = y_train - weights * mean(x_train)


        self.weights = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T.dot(y_train))
        self.bias = np.mean(y_train, axis=0) - np.dot(np.mean(x_train, axis=0), self.weights)

    def predict(self, x_test):
        """
        :param x_test: 2D prediction data
        :return : predictions in a 1D numpy array
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return np.dot(x_test, self.weights) + self.bias

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: r^2 score
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        y_pred = np.dot(x_test, self.weights) + self.bias
        return r2_score(y_pred, y_test)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot([_ for _ in range(len(y_pred))], y_pred, color="blue",
                 label="predictions/y_pred")  # plot all predictions in blue
        plt.plot([_ for _ in range(len(y_test))], y_test, color="green",
                 label="labels/y_test")  # plot all labels in green
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def _perc_correct(y_pred, y_test) :
    return np.sum((y_pred == y_test).astype('int'))/len(y_pred)

class LogisticRegression():
    """
    Logistic Regression is in the sweet spot of being easy to understand and useful. Despite having "regression" in the
    name, what it does is binary classification. Say you had a dataset of people's heights and weights, along with
    other attributes (we call them features) and you wanted to predict whether they were healthy (0) or unhealthy (1).
    You may choose to use logistic regression as this task is binary classification (classifying 2 categories.) Make sure
    your labels are 0 and 1.

    Logistic Regression doesn't have a closed form solution, so we'll have to use the gradient descent algorithm. It may
    take longer but we've provided a progress bar for you to see how it's going.

    A few parameters of our model are related to gradient descent, so I'll explain them here :
    ->> accuracy desired : threshold to when you can stop training (when the accuracy is high enough)
    ->> learning rate : how fast gradient descent will move (too high - won't learn, too low - too long)
    ->> max_iters : max number of iterations of gradient descent
    ->> show_acc : whether or not to show the accuracy while training/gradient descent is performed

    A little bit of research you may want to look into is the sigmoid function, it's what really is at the core of
    distinguishing logistic and linear regression. It'll make more sense after you look at the differences in their output
    equations.

    With that in mind, we can get started!

    ----
    Methods

    __init__(accuracy_desired=0.95, learning_rate=0.01, max_iters=1000, show_acc=True) :
        ->> accuracy_desired : threshold to stop training (setting it to 1 may take too long and may even lead to a worse model)
        ->> learning_rate : how fast you want gradient descent to happen
        ->> max_iters : max number of iterations of gradient descent
        ->> whether or not to show the accuacy during training
    reset_params() :
        ->> only can be called after the fit() method
        ->> resets the weights and biases learnt by the model.
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
    def __init__(self, accuracy_desired=0.95, learning_rate=0.01, max_iters=1000, show_acc=True):
        """
        :param accuracy_desired: the accuracy at which the fit() method can stop
        :param learning_rate: how fast gradient descent should run
        :param max_iters: how many iterations of gradient descent are allowed
        :param show_acc: whether or not to show the accuracy in the fit() method
        """
        self.accuracy_desired = accuracy_desired
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.show_acc = show_acc

    def reset_params(self):
        """
        Run this after the fit() method has been run please.
        :return: Nothing, just redoes the weight init for all parameters.
        """

        self.weights = np.random.randn(self.num_features, 1)
        self.bias = np.random.uniform()

    def fit(self, x_train, y_train):
        """
        :param x_train: training data (2D)
        :param y_train: training labels (1D)
        :return:
        """

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")

        learning_rate = self.learning_rate
        max_iters = self.max_iters
        show_acc = self.show_acc
        accuracy_desired = self.accuracy_desired

        def param_init(x_train):
            '''initialize weights and biases'''
            num_features = len(x_train[0])  # number of features
            weights = np.random.randn(num_features, 1)
            bias = np.random.uniform()
            return weights, bias

        weights, bias = param_init(x_train)
        self.num_features = len(x_train[0])
        m = len(y_train)
        iterations = tqdm(range(max_iters), position = 0)
        for iteration in iterations:
            y_hat = sigmoid(np.dot(x_train, weights) + bias)  # forward pass, y_hat = sigmoid(wx + b)

            # gradient descent
            y_hat, y_train = y_hat.reshape(m, 1), y_train.reshape(m, 1)
            dLdZ = (y_hat - y_train) / m  # Log Loss derivative
            weights -= learning_rate * np.dot(x_train.T, dLdZ)
            bias -= learning_rate * sum(dLdZ)

            perc_correct = _perc_correct(np.round_(y_hat), y_train)
            accuracy_perc = perc_correct

            if show_acc:
                iterations.set_description("accuracy : " + str(round(accuracy_perc * 100, 2)) + "% ")

            if accuracy_perc >= accuracy_desired:
                iterations.close()
                print("Accuracy threshold met!")
                break

            self.weights = weights
            self.bias = bias

    def predict(self, x_test):
        """
        :param x_test: prediction data (2D)
        :return: predictions in a 1D vector/list
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return np.round_(sigmoid(np.dot(x_test, self.weights) + self.bias)).flatten()

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: what % of its predictions on x_test were correct
        """

        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        y_pred = LogisticRegression.predict(self, x_test)
        y_pred = np.array(y_pred).flatten()
        y_test = np.array(y_test).flatten()  # make sure same shape
        perc_correct = _perc_correct(np.round_(y_pred), y_test)
        return perc_correct

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


def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores))


class SoftmaxRegression():
    """
        Once you know logistic regression, softmax regression is a breeze. Logistic regression is really a specific
        type of softmax regression, where the number of classes is equal to 2. Whereas logistic regression only can predict
        for 2 classes (0 and 1), softmax regression can predict for N number of classes. This can be 5, 3, or even a thousand!
        To define the number of classes in this model, insert the num_classes parameter in the init. ALL parameters in
        this class are the same as in logistic regression except for that argument.

        Another note, if you use softmax regression with 2 classes - you just end up using logistic regression. In general
        you should use logistic regression if there are only 2 classes as it is faster and optimized as such.

        If you're interested in the theory, the primary change is from the sigmoid function to the softmax function.
        Also look into the log loss and crossentropy loss, both of them are at the heart of softmax and logistic regression.
        Maybe interesting to read up on that.

        ----
        Methods

        __init__(num_classes, accuracy_desired=0.95, learning_rate=0.01, max_iters=1000, show_acc=True) :
            ->> num_classes : number of classes, or unique labels, your dataset has
            ->> accuracy_desired : threshold to stop training (setting it to 1 may take too long and may even lead to a worse model)
            ->> learning_rate : how fast you want gradient descent to happen
            ->> max_iters : max number of iterations of gradient descent
            ->> whether or not to show the accuacy during training
        reset_params() :
            ->> only can be called after the fit() method
            ->> resets the weights and biases learnt by the model.
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

    def __init__(self, num_classes, accuracy_desired=0.95, learning_rate=0.01, max_iters=1000,
                 show_acc=True):
        """
        :param num_classes : number of classes your dataset has
        :param accuracy_desired: the accuracy at which the fit() method can stop
        :param learning_rate: how fast gradient descent should run
        :param max_iters: how many iterations of gradient descent are allowed
        :param show_acc: whether or not to show the accuracy in the fit() method
        """
        self.accuracy_desired = accuracy_desired
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.show_acc = show_acc
        self.num_classes = num_classes

    def reset_params(self):
        """
        Run this after the fit() method has been run please.
        :return: Nothing, just redoes the weight init for all parameters.
        """
        self.weights = np.random.randn(self.num_features, self.num_classes)
        self.bias = np.random.uniform()

    def fit(self, x_train, y_train):
        """
        :param x_train: training data (2D)
        :param y_train: training labels (1D)
        :return:
        """

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")

        learning_rate = self.learning_rate
        max_iters = self.max_iters
        show_acc = self.show_acc
        accuracy_desired = self.accuracy_desired
        num_classes = self.num_classes

        y_train = one_hot(y_train, depth=num_classes)  # one_hot_encode the labels
        x_train, y_train = np.array(x_train), np.array(y_train)  # convert to numpy arrays

        def param_init(x_train):
            num_features = len(x_train[0])
            weights = np.random.randn(num_features, num_classes)
            bias = np.random.randn(1, num_classes)
            return weights, bias

        weights, bias = param_init(x_train)
        self.num_features = len(x_train[0])
        self.num_classes = num_classes  # important for parameter intializations
        m = len(y_train)

        iterations = tqdm(range(max_iters))
        for iteration in iterations:

            Z = np.dot(x_train, weights) + bias  # forward pass start
            y_hat = np.apply_along_axis(softmax, 1, Z)# softmax for each output

            # gradient descent
            dLdZ = (y_hat - y_train) / m  # crossentropy loss derivative
            weights -= learning_rate * np.dot(x_train.T, dLdZ)
            bias -= learning_rate * np.sum(dLdZ)  # change bias -> dLdZ is a matrix now due to one_hot_labels
            num_correct = sum([1 if np.argmax(pred) == np.argmax(label) else 0 for pred, label in
                               zip(np.round_(y_hat), y_train)])  # what percent are labeled correctly - super simple!

            accuracy_perc = num_correct / m
            if show_acc:
                iterations.set_description("accuracy : " + str(round(accuracy_perc * 100, 2)) + "% ")

            if accuracy_perc >= accuracy_desired:
                iterations.close()
                print("Accuracy threshold met!")
                break

            self.weights = weights
            self.bias = bias

    def predict(self, x_test):
        """
        :param x_test: prediction data (2D)
        :return: predictions in a 1D vector/list
        """

        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")

        x_test = np.array(x_test)
        if len(x_test.shape) < 2:
            x_test = np.array([np.array([x_i]) for x_i in x_test])  # make it 2D

        Z = np.dot(x_test, self.weights) + self.bias
        y_hat = np.apply_along_axis(softmax, 1, Z)  # softmax for each output)
        return np.apply_along_axis(np.argmax, 1, y_hat)  # go through each output and give the index of where

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: what % of its predictions on x_test were correct
        """

        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        Z = np.dot(x_test, self.weights) + self.bias
        y_pred = np.apply_along_axis(softmax, 1, Z)
        y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
        perc_correct = _perc_correct(y_pred, y_test)
        return perc_correct

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


class RidgeRegression():
    """
    Imagine you have a dataset with 1000 features. Most of these features will usually be irrelevant to the task your solving;
    only a few of them will really matter. If you don't want to use Dimensionality Reduction (see the algorithms there),
    you may want to consider using this. What ridge regression does is try to keep the weights as small as possible,
    a.k.a. regularization. This is because if a weight of a feature is not needed you want it to be 0 -  you don't want it
    to be 0.01 because of overfitting or the particular instances of the training data. Therefore it will work well
    with many features as it reduces the weights, hence making the model overfit less and generalize more (do we really need
    those 0.1s and 0.2s in the weights?) As StatQuest said, it "desensitizes" the training data (highly recommend you
    watch that video - regularization can be tough.)

    You'll be glad to know that this uses a closed form solution (generally much faster than iterative gradient descent.)
    There are other algorithms for regularization in regression like Lasso and Elastic Net, a combination of Lasso and Ridge,
    that are available in this module.

    There's only one parameter you need to worry about, which is alpha. It is simply how much to punish the model for its
    weights (especially unnecessary ones). It's typically set between 0 and 1.

    ----
    Methods

    __init__(alpha = 0.5) :
        ->> alpha is the value for how much to punish the model. Remember the higher it is, the smaller the weights.
        ->> 0.5 default
    fit(x_train, y_train) :
        ->> x_train is your training data (2D)
        ->> y_train is your training labels (1D)

    predict(x_test) :
        ->> x_test is your prediction data (2D)
        ->> you will get back a 1D numpy array of a prediction for every data point in x_test

    evaluate(x_test, y_test) :
        ->> this method is to evaluate the model on data it hasn't seen/trained on (not in x_train in the fit() method.)
        ->> reason you do this is because you want to see whether the model can generalize what it learnt on x_train
        to x_test.
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)
        ->> returns the r^2 score

    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.
    """


    def __init__(self, alpha=0.5):
        """
        Set the alpha parameter for the model.
        :param alpha: default 0.5, ranges from 0 - 1
        """
        self.alpha = alpha

    def fit(self, x_train, y_train):
        """
        :param x_train: 2D training data
        :param y_train: 1D training labels
        :return:
        """
        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")


        #Closed form solution below :
        #Weights = (XT X)^-1 XT y
        #Intercept_vector = y_train - weights * mean(x_train)

        self.weights = np.linalg.inv(x_train.T.dot(x_train) + self.alpha + np.identity(x_train.shape[1])).dot(
            x_train.T.dot(y_train))
        self.bias = np.mean(y_train, axis=0) - np.dot(np.mean(x_train, axis=0), self.weights)

    def predict(self, x_test):
        """
        :param x_test: 2D prediction data
        :return : predictions in a 1D numpy array
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return np.dot(x_test, self.weights) + self.bias

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: r^2 score
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        y_pred = np.dot(x_test, self.weights) + self.bias
        return r2_score(y_pred, y_test)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot([_ for _ in range(len(y_pred))], y_pred, color="blue",
                 label="predictions/y_pred")  # plot all predictions in blue
        plt.plot([_ for _ in range(len(y_test))], y_test, color="green",
                 label="labels/y_test")  # plot all labels in green
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()


def _lasso_sign_weight(weight):
    if weight > 0: return 1
    if weight < 0: return -1
    if weight == 0: return 0


class LassoRegression():
    """
    Regularizer for regression models just like Ridge Regression. A few notable differences, but for the most part
    will do the same thing. Lasso regression tries to minimize the weights just like ridge regression, but one of
    its big differences is its tendency to make the weights of the regression model 0. This greatly decreases
    overfitting by making sure unnecessary features aren't considered in the model.

    Another difference is the use of gradient descent instead of a closed form solution like Ridge Regression. It shares
    the same alpha parameter to determine how much you want to "punish" (i.e. reduce) the weights, especially those not needed.

    ----
    Methods :

    __init__(alpha=0.5, accuracy_desired=0.95, learning_rate=0.01, max_iters=1000, show_acc=True) :
        ->> alpha is the penalty for the weights
        ->> check above documentation for all other parameters (recommended your familiar with gradient descent before
        using Lasso Regression.)
    reset_params() :
        ->> resets all weights and biases
    fit(x_train, y_train) :
        ->> same rule as any other regression model
        ->> x_train must be 2D and is your data
        ->> y_train is your labels and must be 1D
    predict(x_test) :
        ->> this method takes in your prediction data (must be 2D) in x_test and returns the predictions in a 1D list/vector
    evaluate(x_test, y_test) :
        ->> give your validation set here and you will get back the r^2 score
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels(1D)
    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.

    """
    def __init__(self, alpha=0.5, accuracy_desired=0.95, learning_rate=0.01, max_iters=10000, show_acc=True):
        """
        :param alpha: penalty for the weights, applied to both lasso and ridge components
        Check above documentation for all other parameters.
        """

        self.alpha = alpha
        self.accuracy_desired = accuracy_desired
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.show_acc = show_acc

    def reset_params(self):
        """
        Resets all weights and biases of the model.
        :return: none
        """
        self.weights = np.random.randn(self.num_features, 1)
        self.bias = np.random.uniform()

    def fit(self, x_train, y_train):
        """
        :param x_train: training data (must be 2D)
        :param y_train: training labels (must be 1D)
        :return: None, just the model has the weights and biases stored
        """

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")

        self.num_features = len(x_train[0])

        def param_init(x_train):
            '''initialize weights and biases'''
            num_features = len(x_train[0])  # number of features
            weights = np.random.randn(num_features, 1)
            bias = np.random.uniform()
            return weights, bias

        weights, bias = param_init(x_train)

        iterations = tqdm(range(self.max_iters))
        m = len(y_train)
        for iteration in iterations:

            if weights.sum() == np.nan or weights.sum() == np.inf or np.isnan(weights.sum()) or np.isinf(weights).any() :
                warnings.warn("Failed convergence, please change hyperparameters.")
                break

            y_hat = np.dot(x_train, weights) + bias

            dLdYh = (y_hat.flatten() - y_train.flatten()) / m
            lasso_signs = self.alpha * np.apply_along_axis(_lasso_sign_weight, 1, weights)
            mse_grad = np.dot(x_train.T, dLdYh)
            gradient = mse_grad + lasso_signs
            weights -= self.learning_rate * np.expand_dims(gradient, 1)
            bias -= self.learning_rate * np.sum(dLdYh)

            r2_score_current = r2_score(y_hat.flatten(), y_train.flatten())

            if self.show_acc :
                iterations.set_description("r^2 : " + str(round(r2_score_current * 100, 2)) + "%")

            if r2_score_current >= self.accuracy_desired:
                iterations.close()
                print("Accuracy threshold met!")
                break

            self.weights, self.bias = weights, bias

    def predict(self, x_test):
        """
        :param x_test: testing data (must be 2D)
        :return: predictions (1D vector/list)
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return (np.dot(x_test, self.weights) + self.bias).flatten()

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: r^2 score for the predictions of x_test and y_test
         """

        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        y_pred = (np.dot(x_test, self.weights) + self.bias).flatten()
        return r2_score(y_pred, y_test)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot([_ for _ in range(len(y_pred))], y_pred, color="blue",
                 label="predictions/y_pred")  # plot all predictions in blue
        plt.plot([_ for _ in range(len(y_test))], y_test, color="green",
                 label="labels/y_test")  # plot all labels in green
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()

class ElasticNet():
    """
    Elastic Net is a combination of Ridge and Lasso Regression. It implements both penalties, and you just decide how
    much weight each should have. The parameter l1_r in the __init__ of this class is the amount of "importance" lasso regression
    has (specifically the regularization term), on a scale from 0 - 1. If lasso regression gets an "importance" of 0.7,
    then we give the ridge regression part of this model an "importance" 0.3. Uses gradient descent, as there is no
    closed form solution.

    ----
    Methods :

    __init__(l1_r = 0.5,  alpha=0.5, accuracy_desired=0.95, learning_rate=0.01, max_iters=1000, show_acc=True) :
        ->> l1_r is how important lasso regression will be in this mix of lasso and ridge regression
        ->> alpha is the penalty for the weights, and is applied to both the lasso and ridge penalty
        ->> check lasso documentation for alpha, and above documentation for the other parameters
    reset_params() :
        ->> resets all weights and biases
    fit(x_train, y_train) :
        ->> same rule as any other regression model
        ->> x_train must be 2D and is your data
        ->> y_train is your labels and must be 1D
    predict(x_test) :
        ->> this method takes in your prediction data (must be 2D) in x_test and returns the predictions in a 1D list/vector
    evaluate(x_test, y_test) :
        ->> give your validation set here and you will get back the r^2 score
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels(1D)
    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.
    """
    def __init__(self, l1_r = 0.5,  alpha=0.5, accuracy_desired=0.95, learning_rate=0.01, max_iters=1000, show_acc=True):
        """
        :param l1_r: The weight that lasso regression gets in this model. Default 0.5, but setting it higher tips the
        scale. Setting it to 0 or 1 makes it just ridge or lasso regression.
        :param alpha: penalty for the weights, applied to both lasso and ridge components
        Check above documentation for all other parameters.
        """
        self.l1_r  = l1_r
        self.alpha = alpha
        self.accuracy_desired = accuracy_desired
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.show_acc = show_acc

    def reset_params(self):
        """
        Resets all weights and biases of the model.
        :return: none
        """
        self.weights = np.random.randn(self.num_features, 1)
        self.bias = np.random.uniform()

    def fit(self, x_train, y_train):
        """
        :param x_train: training data (must be 2D)
        :param y_train: training labels (must be 1D)
        :return: None, just the model has the weights and biases stored
        """
        learning_rate = self.learning_rate
        show_acc = self.show_acc
        accuracy_desired = self.accuracy_desired

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")

        self.num_features = len(x_train[0])

        def param_init(x_train):
            '''initialize weights and biases'''
            num_features = len(x_train[0])  # number of features
            weights = np.random.randn(num_features, 1)
            bias = np.random.uniform()
            return weights, bias

        weights, bias = param_init(x_train)

        iterations = tqdm(range(self.max_iters))
        m = len(y_train)
        for iteration in iterations:

            if weights.sum() == np.nan or weights.sum() == np.inf or np.isnan(weights.sum()) or np.isinf(weights).any() :
                warnings.warn("Failed convergence, please change hyperparameters.")
                break

            y_hat = np.dot(x_train, weights) + bias

            dLdYh = (y_hat.flatten() - y_train.flatten()) / m
            lasso_signs = self.alpha * np.apply_along_axis(_lasso_sign_weight, 1, weights)
            mse_grad = np.dot(x_train.T, dLdYh)
            gradient = mse_grad + lasso_signs * self.l1_r + (1 - self.l1_r) * self.alpha * weights.reshape(weights.shape[0])
            weights -= learning_rate * np.expand_dims(gradient, 1)
            #weights -= np.expand_dims(learning_rate * (np.dot(x_train.T, dLdYh) + lasso_signs), 0)
            bias -= learning_rate * np.sum(dLdYh)

            r2_score_current = r2_score(y_hat.flatten(), y_train.flatten())

            if show_acc :
                iterations.set_description("r^2 : " + str(round(r2_score_current * 100, 2)) + "%")

            if r2_score_current >= accuracy_desired:
                iterations.close()
                print("Accuracy threshold met!")
                break

            self.weights, self.bias = weights, bias

    def predict(self, x_test):
        """
        :param x_test: testing data (must be 2D)
        :return: predictions (1D vector/list)
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return (np.dot(x_test, self.weights) + self.bias).flatten()

    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: r^2 score for the predictions of x_test and y_test
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        y_pred = (np.dot(x_test, self.weights) + self.bias).flatten()
        return r2_score(y_pred, y_test)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot([_ for _ in range(len(y_pred))], y_pred, color="blue",
                 label="predictions/y_pred")  # plot all predictions in blue
        plt.plot([_ for _ in range(len(y_test))], y_test, color="green",
                 label="labels/y_test")  # plot all labels in green
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()


def predict_single_exp_reg(x, w, l, a) :
    #print("l * np.power(w, x) : ", l * np.power(w, x))
    w, l = np.array(w), np.array(l)
    return np.sum(l * np.power(w.astype('complex'), x.astype('complex')).astype('float')) + a


class ExponentialRegression() :
    """

    Say you've got some curved data. How can a line possibly fit that data? Glad you ask - that's why this class exists.
    ExponentialRegression is a special type of regression that finds a base and coefficient for every
    feature (and a bias for everything) to fit your data. All of these parameters are learnt through the standard use of
    gradient descent. It can learn an exponentially curved line by essentially turning it into a line and finding the
    parameters to go through that. Make sure to normalize your data either by the ol' (X - mu) divided by sigma or
    just by dividing it by the biggest number. Gradient Descent can struggle with high outputs, so this is crucial
    (very similar problem here to exploding gradients in a neural network.)

    This module was created as a what-if kinda situation (not as famous as the other models), so please go crazy with it
    and let us know if you find anything interesting.

    ----
    Methods
    __init__( accuracy_desired = 0.95, learning_rate = 0.01, max_iters = 1000, show_acc = True) :
        ->> accuracy_desired : simply how well you want this model to do on a scale of 0 -1. Setting to 1 may mean that
        it will have to run through all iters in max_iters.
        ->> learning_rate : just how gradient descent performs.
        ->> max_iters : max number of iterations of gradient descent allowed for the model
        ->> show_acc : whether or not you want the accuracy, or evaluation, of the model to be shown
        when the fit() method is being run.
    reset_params() :
        ->> resets all parameters
        -> must be done after the fit() method has been run at least 1x
    fit(x_train,y_train) :
        ->> x_train is your training data, which is 2D in a python list/numpy array
        ->> y_train is your training labels, and should be 1D
        ->> please experiment with the learning_rate parameter in the __init__, as that could make training faster or slower!
    predict(x_test) :
        ->> x_test is your 2D prediction data
        ->> returns the predictions in a 1D vector/list
    evaluate(x_test, y_test) :
        ->> x_test is your testing data (2D)
        ->> y_test is your testing labels (1D)
        ->> returns an accuracy score
    visualize_evaluation(y_pred, y_test) :
        ->> y_pred is your predictions (1D) - they must come from the predict() method
        ->> y_test are the labels (1D)
        ->> visualize the predictions and labels. This will help you know if your model is predicting too high,
        too low, etc.
    """
    def __init__(self, accuracy_desired = 0.95, learning_rate = 0.01, max_iters = 1000, show_acc = True):
        """
        :param accuracy_desired: the accuracy at which the fit() method can stop
        :param learning_rate: how fast gradient descent should run
        :param max_iters: how many iterations of gradient descent are allowed
        :param show_acc: whether or not to show the accuracy in the fit() method
        """
        self.accuracy_desired = accuracy_desired
        self.learning_rate = learning_rate
        self.show_acc = show_acc
        self.max_iters = max_iters

    def reset_params(self):
        """
        Run this after the fit() method has been run please.
        :return: Nothing, just redoes the weight init for all parameters.
        """
        self.weights = np.random.randn(self.n_features) * np.sqrt(1 / self.n_features)
        self.lambda_matrix = np.random.randn(self.n_features) * np.sqrt(1 / self.n_features)
        self.alpha = np.random.randn()

    def fit(self, x_train, y_train):
        """
        :param x_train: training data (2D)
        :param y_train: training labels (1D)
        :return:
        """

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")

        n_features = len(x_train[0])
        weights = np.random.randn(n_features) * np.sqrt(1 /n_features)
        lambda_matrix = np.random.randn(n_features) * np.sqrt(1 /n_features)
        alpha = np.random.randn()
        self.n_features = n_features

        losses = []
        m = len(y_train)
        lr = self.learning_rate

        iterations = tqdm(range(self.max_iters), position=0)
        for iteration in iterations:
            y_hat = np.apply_along_axis(predict_single_exp_reg, 1, x_train, w=weights, l=lambda_matrix, a = alpha)
            dJdY_hat = (y_hat - y_train) / m
            if np.isnan(np.sum(dJdY_hat)) :
                warnings.warn("Failed convergence/training. Please try again with a lower learning rate, and tune the max_iters"
                              "parameter if needed. Also make sure your data is scaled from 0 to 1 (easier convergence.)")
                break
            losses.append(np.sum(np.abs(dJdY_hat)))
            dYhdTheta = []  # matrix needs to be transposed
            dYhdLambda = []
            for feature in range(n_features):
                X_feature = x_train[:, feature]
                theta_clone_vector = (np.zeros(len(X_feature)) + weights[feature]).astype('complex')
                dYhdThetaj = lambda_matrix[feature] * X_feature * np.power(theta_clone_vector,
                                                                           (X_feature - 1).astype('complex')).astype('float')
                dYhdTheta.append(dYhdThetaj)
                dYhdLambdaj = np.power(theta_clone_vector, (X_feature).astype('complex')).astype('float')
                dYhdLambda.append(dYhdLambdaj)

            dYhdTheta = np.array(dYhdTheta).T
            weights -= lr * np.dot(dJdY_hat.T, dYhdTheta).T

            dYhdLambda = np.array(dYhdLambda).T
            lambda_matrix -= lr * np.dot(dJdY_hat.T, dYhdLambda).T

            alpha -= lr * np.sum(dJdY_hat)

            self.weights = weights
            self.alpha = alpha
            self.lambda_matrix = lambda_matrix
            score = ExponentialRegression.evaluate(self, x_train, y_train)

            if self.show_acc:
                iterations.set_description("r^2 : " + str(round(score * 100, 2)) + "%")

            if score >= self.accuracy_desired:
                iterations.close()
                print("Accuracy threshold met!")
                break

        self.weights = weights
        self.lambda_matrix = lambda_matrix
        self.alpha = alpha
        self.losses = losses
    def predict(self, x_test) :
        """
        :param x_test: prediction data (2D)
        :return: predictions in a 1D vector/list
        """
        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return np.apply_along_axis(predict_single_exp_reg, 1, x_test, w = self.weights, l = self.lambda_matrix, a = self.alpha)
    def evaluate(self, x_test, y_test):
        """
        :param x_test: testing data (2D)
        :param y_test: testing labels (1D)
        :return: r^2 score
        """
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        y_pred = ExponentialRegression.predict(self, x_test)
        avg_error = np.mean(np.abs(y_pred - y_test))
        return 1 - avg_error/np.mean(np.abs(y_test))
    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot([_ for _ in range(len(y_pred))], y_pred, color="blue",
                 label="predictions/y_pred")  # plot all predictions in blue
        plt.plot([_ for _ in range(len(y_test))], y_test, color="green",
                 label="labels/y_test")  # plot all labels in green
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()

def _poly_transform(data) :
    new_data = []
    for observation in data :
        new_d = np.power(observation, np.linspace(1, len(observation), len(observation)))
        new_data.append(new_d)
    return new_data

class PolynomialRegression():
    """
    Polynomial Regression is like an extended version of Linear Regression (sort of like Exponential
    Regression.) All it does is turn the equation from y = m1x1 + m2x2 ... mNxN + bias
    to y = m1x1^1 + m2x2^2 ... mNxN^N + bias. It just adds those power combinations
    to make the regression module model the data better. Neural networks can also model
    those functions similarily.

    Please normalize your data with this module with the (X - mu) / sigma or just by
    dividing by the maximum value. This will help with faster convergence.

    Not as famous as some other regression algorithms, so may you need a bit of experimentation.

    ----
    Methods
    reset_params() :
        ->> only can be called after the fit() method
        ->> resets the weights and biases learnt by the model.
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


    def reset_params(self):
        """
        Run this after the fit() method has been run please.
        :return: Nothing, just redoes the weight init for all parameters.
        """
        self.inner_linear_model.reset_params()  # calls the LinearRegression.reset_params() method

    def fit(self, x_train, y_train):
        """
        :param x_train: training data (2D)
        :param y_train: training labels (1D)
        :return:
        """

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(x_train.shape) != 2: raise ValueError("x_train must be 2D (even if only one sample.)")
        if len(y_train.shape) != 1: raise ValueError("y_train must be 1D.")

        x_train = _poly_transform(x_train)  # transform data

        self.inner_linear_model = LinearRegression()  # build the linear regression model
        self.inner_linear_model.fit(x_train, y_train)  # fit it (using all given parameters)

    def predict(self, x_test):
        """
        :param x_test: points to be predicted on. Has to be stored in 2D array, even if just one data point. Ex. : [[1, 1]] or [[1, 1], [2, 2], [3, 3]]
        :return: flattened numpy array of your data going through the forward pass (that's rounded as it's either 0 or 1)
        """

        x_test = np.array(x_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        return self.inner_linear_model.predict(x_test)  # just use the prediction method

    def evaluate(self, x_test, y_test):
        """
        :param x_test: data to be evaluated on
        :param y_test: labels
        :return: r^2 score
        """

        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 1: raise ValueError("y_test must be 1D.")

        x_test = _poly_transform(x_test)
        return self.inner_linear_model.evaluate(x_test, y_test)

    def visualize_evaluation(self, y_pred, y_test):
        """
        :param y_pred: predictions from the predict() method
        :param y_test: labels for the data
        :return: a matplotlib image of the predictions and the labels ("correct answers") for you to see how well the model did.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        y_pred, y_test = y_pred.flatten(), y_test.flatten()
        plt.plot([_ for _ in range(len(y_pred))], y_pred, color="blue", label="predictions/y_pred")
        plt.plot([_ for _ in range(len(y_test))], y_test, color="green", label="labels/y_test")
        plt.title("Predictions & Labels Plot")
        plt.xlabel("Data number")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()




