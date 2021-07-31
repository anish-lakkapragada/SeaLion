"""
@author : Anish Lakkapragada
@date : 1 - 23 - 2021

This file contains all of the layer classes you'll need to build your models. You won't need the forward() and
backward() methods to deal with, just really the init. Examples are embedded in the docs because this can get a little complex,
so bear with me here.
"""
from sealion.neural_networks.optimizers import Momentum
import numpy as np


class Layer:
    """Base Layer class. All layers inherit from this."""
    def __init__(self):
        self.parameters = {}
        self.gradients = {}
        self.activation = None
        self.nesterov = False
        self.indices = ...
        pass

    def forward(self, inputs):
        """This method saves the inputs, and returns its outputs"""
        pass

    def backward(self, grad):
        """This method takes in a gradient (e.x ``∂l/∂Z2``) and returns its grad (e.x. ``dL/dA1``)"""
        pass


class Flatten(Layer):
    """
    This would be better explained as Image Data Flattener. Let's say your dealing with MNIST (check out the examples
    on github) - you will have data that is 60000 by 28 by 28. Neural networks can't work with data like that - it has
    to be a matrix. What you could do is take that ``60000 * 28 * 28`` data and apply this layer to make the data
    ``60000 * 784``. 784 because ``28 * 28 = 784`` and we just "squished" this matrix to one layer. If you have data that has colors,
    e.g. ``60000 * 28 * 28 * 3``, applying this layer would turn it into a ``60000 * 2352`` matrix, ``2352 = 28 * 28 * 3``.

    An example for how this would work with something like MNIST (the grayscale ``60k * 28 * 28`` dataset) is shown below.

    >>> from sealion import neural_networks as nn
    >>> from sealion.neural_networks.models import NeuralNetwork
    >>> model = NeuralNetwork()
    >>> model.add(nn.layers.Flatten()) # always, always add that on the first layer
    >>> model.add(nn.layers.Dense(784, whatever_output_size, more_args)) # now put that data to ``28*28 = 784``.
    >>> # Do more cool stuff...
    """

    def forward(self, inputs):
        self.inputs = inputs
        return inputs.reshape(inputs.shape[0], np.product(inputs.shape[1:]))

    def backward(self, grad):
        return self.inputs


class Dropout(Layer):
    """
    Dropout is one of the most well-known regularization techniques there is. Imagine you were working on a coding
    project with about 200 people. If we just relied on one person to know how to compile, another person to know how to debug,
    then what happens when those special people leave for a day, or worse leave forever?

    Now I know that seems to have no connection to dropout, but here's how it does. In dropout this sort of scenario is prevented.
    "Dropping out", or setting to 0, some of the outputs of some of the neurons means that the model will have to learn
    that it can't just depend on one neuron to learn the most important features. This means has each neuron learn some features,
    some other features, etc. and can't just depend on one node. The model will become more robust and generalize
    better with Dropout as every neuron now has a better set of weights. Normally due to dropout, it will be applied
    in training, but then "reverse-applied" in testing. Dropout will make the training accuracy go down a bit, but remember
    in the end it's testing on real-world data that matters.

    There's a parameter dropout_rate on what percent (from 0 to 1 here) you want each neuron in the layer you are at to become 0.
    This is essentially the chance of dropping out any given neuron, or usually what percent of neurons will be dropped out. Typical
    values range from 0.1 to 0.5. Example below.

    Let's say you've gotten your models up so far:

    >>> model.add(nn.layers.Flatten())
    >>> model.add(nn.layers.Dense(128, 64, activation = nn.layers.ReLU()))

    And now you want to add dropout. Well just add that layer like such:

    >>> dropout_probability = 0.2
    >>> model.add(nn.layers.Dropout(dropout_probability))

    This will just mean that about 20% of the neurons coming from this first input layer will be dropped out. A higher dropout rate
    may not always lead to better generalization, but usually will decrease training accuracy.

    In dropout remember 2 things, not just one matter. The probability, and the layers its applied at. **Experimentation is key**.
    """

    def __init__(self, dropout_rate):
        super().__init__()
        self.dr = dropout_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.dropped_inputs = self.inputs * np.random.binomial(
            1, (1 - self.dr), inputs.shape
        )
        return self.dropped_inputs

    def backward(self, grad):
        grad[np.where(self.dropped_inputs[self.indices] == 0)] = 0 # set droppped out outputs to 0 
        grad *= 1/(1-self.dr) # everything else scale by * 1/(1-p)
        return grad


class Dense(Layer):
    """

    This is the class that you will depend on the most. This class is for creating the fully-connected layers that
    make up Deep Neural Networks - where each neuron in a previous layer is connected to each layer in the next. Feel free
    to watch a couple of youtube tutorials from 3Blue1Brown (if you like calculus :) or others to get a better understanding
    of these parameters, or just look at the examples on my github.

    The main method of course is the init. You will have to define the input_size, the output_size, and the activation and
    weight initialization are optional parameters.

    In a neural network the number of nodes starting in a layer are known as the input_size here (fan_in in papers), and
    the number of nodes in the output of a layer is the output_size here (fan_out in papers.) The activation parameter
    is for the activation/non-linearity function you want your layers to go through. If you don't understand what I just meant,
    sorry - but another way to think about it is that its a way to change the outputs of your network a bit for it to
    fit you dataset a little better (looking at a graph will help.) The default is no activation (some people call that Linear),
    so you'll need to add that yourself. I'll get to weight init in a bit. Examples below.

    To add a layer, here's how it's done:

    >>> from sealion import neural_networks as nn
    >>> model = nn.models.NeuralNetwork()
    >>> model.add(nn.layers.Dense(128, 64)) # input_size = 128, output_size = 64

    This sets up a neural network with 128 incoming nodes (that means we have 128 numeric features), and then 64 output nodes.

    Let's say we wanted an activation function, like a relu or sigmoid (there are a bunch to choose from this API.)
    You could add that layer here like such:

    >>> model = nn.models.NeuralNetwork() # clear all existing layers
    >>> model.add(nn.layers.Dense(128, 64, activation=nn.layers.Sigmoid())) # all outputs go through the sigmoid function

    Onto weight initalization! The jargon starts .... now. What weight init does is make the weights come from
    a special distribution (typically Gaussian) with a given standard deviation based on the input and output size.
    The reason this is done is because you don't want the weights to be initalized too big or else the gradients in
    backpropagation may cause the model to go way off and become NaNs or infinity (this is known as exploding gradients.)
    For neural networks that are small that solve datasets like XOR or even breast cancer this isn't a problem, but for
    deep neural networks for problems like MNIST this is a huge concern. The weight_init you will want to use is also
    dependent on what activation you are using. Most activation functions will do well with Xavier Glorot, so that is
    set as the default. You can choose to also use He for ReLU, LeakyReLU, ELU, or other variants of the ReLU activation.
    For SELU, you may choose to use LeCun weight initalization.

    The possible choices are:

    >>> "xavier" # no activation, logistic/sigmoid, softmax, and tanh
    >>> "he"     # relu + similar variants
    >>> "lecun"  # selu
    >>> "none"   # if you want to do this

    To set this you can just do:

    >>> model = nn.models.NeuralNetwork() # clear all existing layers
    >>> model.add(nn.layers.Dense(128, 64, activation=nn.layers.ReLU(), weight_init="he"))

    Sorry for so much documentation, but this really is the class you will call the most.
    """

    def __init__(self, input_size: int, output_size: int, activation=None, weight_init="xavier") -> None:
        super().__init__()

        weight_init_addition = 1  # no weight init
        if weight_init.lower() == "xavier":
            weight_init_addition = np.sqrt(2 / (output_size + input_size))
        elif weight_init.lower() == "he":
            weight_init_addition = np.sqrt(2 / input_size)
        elif weight_init.lower() == "lecun":
            weight_init_addition = np.sqrt(1 / input_size)
        elif not weight_init.lower() == "none":
            print(f"No known weight init : {weight_init}")

        self.parameters["weights"] = (
            np.random.randn(input_size, output_size) * weight_init_addition
        )
        self.parameters["bias"] = np.random.randn(output_size)
        self.gradients["weights"] = np.random.randn(*self.parameters["weights"].shape)
        self.gradients["bias"] = np.random.randn(*self.parameters["bias"].shape)
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.parameters["weights"]) + self.parameters["bias"]

    def backward(self, grad):
        # here we can update our weights
        """if gradient ∂L/∂ZN, store grads ∂l/∂w1 and return dL/dA1"""

        self.gradients["weights"] = self.inputs[self.indices].T.dot(grad)
        self.gradients["bias"] = np.sum(grad, axis=0)
        return np.dot(grad, (self.parameters["weights"]).T)  # dLdZ2 * dZ2/dA1

class BatchNormalization(Layer): 
    """
    Batch Normalization is a frequently used technique in today's models (especially in CNNs). 

    Often times you are told to normalize your data (make it bell-curved shaped) before 
    you feed it to your model - that's because normalization makes it easier for the model to learn 
    because the loss curve is smoother (thus getting to the minima is easier and faster.) So, 
    why don't we apply normalization to the inputs of all of the other layers (aside from the input layer)? 
    That's what batch normalization does. 

    I got this explanation from `CodeEmporium <https://www.youtube.com/watch?v=DtEq44FTPM4>`__

    In order for a given hidden layer to normalize its inputs it needs to know the mean 
    and variance (just standard deviation squared) of the inputs it usually gets. Note that 
    this does change (because the parameters of prior layers change), so the mean and variance
    is not necessarily the same throughout training. To do this we need to use a moving average 
    to approximate the mean and variance of the given inputs fed to a B.N. layer we would like to normalize. 

    While the B.N. first uses this mean and variance to normalize its inputs to a standard normal distribution, 
    where the mean is 0 and the standard deviation is 1, it is not guaranteed that this distribution is optimal 
    as inputs for the succeeding layer. So the B.N. layer also learns a mean and variance, called beta and gamma respectively, 
    to adjust the standard normal distribution (it got by normalizing its inputs) before feeding it to the next
    layer. 

    Onto the parameters: 

    **input_size**: number of features the B.N. layer needs to normalize (just output_size of the Dense layer above)

    **momentum**: how slowly to change the mean and variance that the B.N. layer uses for normalization. 
    A higher momentum means that the mean and the variance the B.N. layer uses will change very slowly, 
    whereas a lower momentum means that the mean and the variance the B.N. layer uses will change very quickly 
    in response to changes in the B.N.'s inputs (and their mean and variance). Default 0.9. 

    **epsilon**: tiny value needed in normalization if the variance ever becomes 0 to protect against 0 division errors. Default 0.001. 

    **lr**: learning rate for the B.N. layer's learning of the normalized mean and variance. Default 0.01. 


    To add Batch Normalization to your model simply do:

    >>> import sealion as sl 
    >>> model = sl.neural_networks.models.NeuralNetwork()
    >>> model.add(...) # add whatever Dense Layers 
    >>> model.add(sl.neural_networks.layers.BatchNormalization(input_size = 5, momentum = 0.9, lr = 0.01))

    Note you may want to experiment on whether to place a Batch Normalization layer after an activation or before, etc. I don't know 
    too much about this, but handsonml said this so I might as well too. 
    """
    def __init__(self, input_size: int, momentum=0.9, epsilon=0.001, lr=0.01) -> None: 
        super().__init__() 
        self.momentum = momentum 
        self.epsilon = epsilon
        self.lr = lr
        self.first = False 
        self.training = True # by default we are assuming it is training

        # create mu and sigma 
        self.mu, self.var = np.zeros(input_size), np.zeros(input_size)

        # create the learnable parameters gamma and beta 
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
    
    def forward(self, inputs): 
        # first calculate mu and the variance
        current_mu = np.mean(inputs, axis=0)
        current_var = np.var(inputs, axis = 0)

        # update mu and variance 
        if self.first and self.training: 
            self.mu = current_mu 
            self.var = current_var
            self.first = False 

        if not self.first and self.training: 
            self.mu = self.momentum * self.mu + (1 - self.momentum) * current_mu
            self.var = self.momentum * self.var + (1 - self.momentum) * current_var

        # only update self.mu and self.var based on inputs in training 

        self.X_hat = (inputs - self.mu) / np.sqrt(self.var + self.epsilon)
        
        self.y = self.gamma * self.X_hat + self.beta
        return self.y 
    
    def backward(self, grad):
        # first find dL/dgamma and dL/dbeta and update params   

        dLdBeta = np.sum(grad, axis = 0)
        dLdGamma = np.sum(grad * self.X_hat, axis = 0)


        # we have to pass back dL/dX
        N = grad.shape[0] 
        dLdX_hat = grad * self.gamma
        
        dLdX = (N * dLdX_hat - np.sum(dLdX_hat, axis = 0) - self.X_hat * np.sum(dLdX_hat * self.X_hat, axis = 0)) / (N * np.sqrt(self.var + self.epsilon))

        # update last
        self.beta -= self.lr * dLdBeta
        self.gamma -= self.lr * dLdGamma

        return dLdX


class Activation(Layer):
    """
    To add Activation Functions, check the Dense class for a tutorial, or the examples.
    """

    def __init__(self, activ_func, activ_func_prime):
        super().__init__()
        self.activ_func = activ_func
        self.activ_func_prime = activ_func_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.activ_func(inputs)

    def backward(self, grad):
        return self.activ_func_prime(self.inputs[self.indices]) * grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(x):
    return np.tanh(x)


def tanh_prime(z):
    return 1 - np.power(tanh(z), 2)


def relu(x):
    return np.maximum(x, 0)


def relu_prime(z):
    return (z > 0).astype(int)


def leaky_relu(x, leak=0.01):
    return np.maximum(leak * x, x)


def leaky_relu_prime(z, leak=0.01):
    grad = np.ones_like(z)
    grad[z < 0] *= leak
    return grad


def elu(z, alpha=1):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))


def elu_prime(z, alpha=1):
    grad = np.where(z > 0, np.ones_like(z), alpha * (np.exp(z)))
    return grad


def swish(x):
    return x * sigmoid(x)


def swish_prime(z):
    return swish(z) + sigmoid(z) * (1 - swish(z))


def selu(x):
    alpha = 1.6732
    lamb = 1.0507
    return lamb * np.maximum((alpha * np.exp(x) - alpha), x)


def selu_prime(x):
    alpha = 1.6732
    lamb = 1.0507
    grad = np.where(x <= 0, np.ones_like(x), lamb * alpha * np.exp(x))
    return grad


class Tanh(Activation):
    """Uses the tanh activation, which squishes values from -1 to 1."""

    def __init__(self):
        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    """Uses the sigmoid activation, which squishes values from 0 to 1."""

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class Swish(Activation):
    """
    Uses the swish activation, which is sort of like ReLU and sigmoid combined. It's really just ``f(x) = x * sigmoid(x)``.
    Not as used as other activation functions, but give it a try!
    """

    def __init__(self):
        super().__init__(swish, swish_prime)


class ReLU(Activation):
    """
    The most well known activation function and the pseudo-default almost. All it does is turn negative values to 0 and
    keep the rest. Basically ``f(x) = max(x, 0)``
    """

    def __init__(self):
        super().__init__(relu, relu_prime)


class LeakyReLU(Activation):
    """
    Variant of the ReLU activation, just allows negatives values to be something more like 0.01 instead of 0, which means
    the neuron is "dead" as ``0 * anything`` is 0.

    The leak is the slope of how low negative values you are willing to tolerate. Usually set from 0.001 to 0.2, but the
    default of 0.01 works usually quite well.
    """

    def __init__(self, leak=0.01):
        super().__init__(leaky_relu, leaky_relu_prime)
        self.leak = leak

    def forward(self, inputs):
        self.inputs = inputs
        return self.activ_func(inputs, leak=self.leak)

    def backward(self, grad):
        return self.activ_func_prime(self.inputs[self.indices], leak=self.leak) * grad

class ELU(Activation):
    """
    Solves the similar dying activation problem. The default of 1 for alpha works quite well in practice, so you won't need
    to change it much.
    """

    def __init__(self, alpha=1):
        super().__init__(elu, elu_prime)
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        return self.activ_func(inputs, alpha=self.alpha)

    def backward(self, grad):
        return self.activ_func_prime(self.inputs[self.indices], alpha=self.alpha) * grad


class SELU(Activation):
    """
    Special type of activation function, that will "self-normalize" (have a mean of 0, and a standard deviation of 1) its outputs.
    This self-normalization typically leads to faster convergence.

    If you are using this activation function make sure weight_init is set = "lecun" in whichever layer applied.
    It also need its inputs (in the beginning and all throughout) to be standardized (mu = 0, sigma = 1) for it to work,
    so make sure to get that taken care of. You can do that by standardizing your inputs and then always using
    SELU activation function (remember the "lecun" part!).
    """

    def __init__(self):
        super().__init__(selu, selu_prime)


class PReLU(Layer):
    """
    The PReLU activation function is essentially the same thing as the LeakyReLU activation, except that the "leak"
    parameter is learnt during training. To learn this parameter, gradient descent is used - so you have a learning rate and momentum parameter.
    Both are on a scale of 0 to 1. We initialize this "leak" parameter to 0.25 at the first iteration.
    """

    def __init__(self, lr = 0.0001, momentum = 0.9) :
        super().__init__()
        self.a = 0.25
        self.delta_a = 0 
        self.momentum = momentum
        self.lr = lr
    def forward(self, inputs) :
        self.inputs = inputs
        self.outputs = np.maximum(inputs, 0) + self.a * np.minimum(inputs, 0)
        return self.outputs
    def backward(self, grad) :
        # gradient is the ∂J/∂Output of this PReLU function
        
        # first find ∂J/∂a and update a
        dOutputda = np.zeros(self.outputs.shape)
        dOutputda[np.where(self.outputs < 0)] = self.inputs[np.where(self.outputs < 0)]

        # now return back dJdInput
        dOutputdInput = np.ones(self.inputs.shape)
        dOutputdInput[np.where(self.outputs <= 0)] = self.a
        
        self.delta_a = self.momentum * self.delta_a + self.lr * (grad * dOutputda).sum()
        self.a += self.delta_a  
 
        return grad * dOutputdInput


class Softmax(Layer):
    """
    Softmax activation function, used for multi-class (2+) classification problems.
    Make sure to use crossentropy with softmax, and it is only meant for the last layer!
    """

    def forward(self, inputs):
        self.inputs = inputs
