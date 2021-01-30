"""
@author : Anish Lakkapragada
@date : 1 - 23 - 2021

The loss functions are really simple. You just need to understand whether it is a classification or regression task.
All losses will be set in the model.finalize() model.
"""
import numpy as np
import warnings
from scipy.special import softmax as sfmx_indiv

warnings.filterwarnings('ignore', category = RuntimeWarning)

class Loss :
    def __init__(self):
        self.SGD = False
    def loss(self, y, y_pred) :
        pass
    def grad(self, y, y_pred):
        pass

class MSE(Loss) :
    """
    MSE stands for mean-squared error, and its the loss you'll want to use for regression. To set it in the model.finalize()
    method just do :
    >>> from sealion import neural_networks as nn
    >>> model = nn.models.NeuralNetwork(layers_list)
    >>> model.finalize(loss = nn.loss.MSE(), optimizer = ...)

    and you're all set!

    """
    def __init__(self):
        super().__init__()
        self.type_regression = True
    def loss(self, y, y_pred):
        error = np.sum(np.power(y_pred - y, 2)) / (2 * len(y))
        return error
    def grad(self, y, y_pred):
        return (y_pred - y) / len(y)


def softmax(x) :
    softmax_output =  np.apply_along_axis(sfmx_indiv, 1, x)
    return softmax_output

class CrossEntropy(Loss) :
    """
    This loss function is for classification problems. I know there's a binary log loss and then a multi-category cross entropy
    loss function for classification, but they're essentially the same thing so I thought using one class would make it easier.
    Remember to use one-hot encoded data for this to work (check out utils.)

    If you are using this loss function, make sure your last layer is Softmax and vice versa. Otherwise, annoying error
    messages will occur.

    To set this in the model.finalize() method do :
    >>> from sealion import neural_networks as nn
    >>> model = nn.models.NeuralNetwork()
    ... add the layers ...
    >>> model.add(nn.layers.Softmax()) #last layer has to be softmax
    >>> model.finalize(loss = nn.loss.CrossEntropy(), optimizer = ...)

    and that's all there is to it.
    """
    def __init__(self):
        super().__init__()
        self.type_regression = False
    def loss(self, y, y_pred) :
        return np.sum(y * np.log(y_pred + 1e-20)) / len(y) #now give the crossentropy loss
    def grad(self, y, y_pred):
        y_pred = softmax(y_pred)
        return (y_pred - y) / len(y) #give the sexy partial derivative

