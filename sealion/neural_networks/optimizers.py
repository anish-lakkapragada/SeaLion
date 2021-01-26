"""
@author : Anish Lakkapragada
@date : 1 - 23 - 2021

The optimizer APIs really go from the simple basic Gradient Descent optimization algorithms to newer, and more nuanced
algorithms like Adam and RMSprop. The optimizers will build up in sequential order, so please read the documentation in
that order too if you are unfamiliar with these algorithms.

You may want to check out this blog post to understand these algorithms in more detail :
https://ruder.io/optimizing-gradient-descent/

"""

import numpy as np

class Optimizer :
    def __init__(self):
        self.nesterov = False
    def setup(self, nn):
        pass
    def update(self, nn):
        pass

class GD(Optimizer) :
    '''

    The simplest optimizer -  you will quickly outgrow it. All you need to understand here is that the learning rate
    is just how fast you want the model to learn (default 0.001) set typically from 1e-6 to 0.1. A higher learning rate may mean
    the model will struggle to learn, but a slower learning rate may mean the model will but will also have to take more
    time. It is probably the most important hyperparameter in all of today's machine learning.

    The clip_threshold is simply a value that states if the gradients is higher than this value, set it to this.
    This is to prevent too big gradients, which makes training harder. The default for all these optimizers is infinity,
    which just means no clipping - but feel free to change that. You'll have to experiment quite a bit to find a good value.
    This method is known as gradient clipping.

    In the model.finalize() method :
    >>> model.finalize(loss = ..., optimizer = nn.optimizers.GD(lr = 0.5, clip_threshold = 5)) # here the learning
    # learning rate is 0.5 and the threshold for clipping is 5.
    '''
    def __init__(self, lr = 0.001, clip_threshold = np.inf) :
        super().__init__()
        self.lr = lr
        self.clip_threshold = clip_threshold
    def update(self, nn):
        for layer in nn :
            for param_name, _ in layer.parameters.items() :
                gradients = layer.gradients[param_name]
                gradients[np.where(np.abs(gradients) > self.clip_threshold)] = self.clip_threshold
                layer.parameters[param_name] -= self.lr * gradients

class Momentum(Optimizer) :
    """
    If you are unfamiliar with gradient descent, please read the docs on that in GD class for this to hopefully make more sense.

    Momentum optimization is the exact same thing except with a little changes. All it does is accumulate the past gradients,
    and go in that direction. This means that as it makes updates it gains momentum and the gradient updates become bigger and bigger
    (hopefully in the right direction.) Of course this will be uncontrolled on its own, so a momentum parameter (default 0.9)
    exists so the previous gradients sum don't become too big. There is also a nesterov parameter (default false, but
    set that true!) which sees how the loss landscape will be in the future, and makes its decisions based off of that.

    An example :
    >>> momentum = nn.optimizers.Momentum(lr = 0.02, momentum = 0.3, nesterov = True) #learning rate is 0.2, momentum at 0.3, and we have nesterov!
    >>> model.finalize(loss = ..., optimizer = momentum)

    There's also a clip_threshold argument which you implements gradient clipping, an explanation you can find in the GD()
    class's documentation.

    Usually though this works really good with SGD...
    """
    def __init__(self, lr = 0.001, momentum = 0.9, nesterov = False, clip_threshold = np.inf) :
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.clip_threshold = clip_threshold
    def setup(self, nn):
        self.momentum_params = {}
        for layer_num in range(len(nn)):
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items():
                self.momentum_params[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)

        if self.nesterov :
            for layer_num in range(len(nn)) :
                layer = nn[layer_num]
                layer.nesterov = True
                nn[layer_num] = layer

    def update(self, nn):
        '''nn is self.layers'''
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items() :
                self.momentum_params[param_name + str(layer_num)] = self.momentum * self.momentum_params[param_name + str(layer_num)] \
                 - self.lr * layer.gradients[param_name]
                final_grad = self.momentum_params[param_name + str(layer_num)]
                final_grad[np.where(np.abs(final_grad) > self.clip_threshold)] = self.clip_threshold
                layer.parameters[param_name] +=final_grad


class SGD(Optimizer) :
    """
    SGD stands for Stochastic gradient descent, which means that it calculates its gradients on random (stochastic) picked samples
    and their predictions. The reason it does this is because calculating the gradients on the whole dataset can take a
    really long time. However ML is a tradeoff and the one here is that calculating gradients on just a few samples
    means that if those samples are all outliers it can respond poorly, so SGD will train faster but not get as high an
    accuracy as Gradient Descent on its own.

    Fortunately though, there are work arounds. Implementing momentum and nesterov with SGD means you get faster training
    and also the convergence is great as now the model can go in the right direction and generalize instead of overreact
    to hyperspecific training outliers. By default nesterov is set to False and there is no momentum (set to 0.0), so please
    change that as you please.

    To use this optimizer, just do :
    >>> model.finalize(loss = ..., optimizer = nn.optimizers.SGD(lr = 0.2, momentum = 0.5, nesterov = True, clip_threshold = 50))

    Here we implemented SGD optimization with a learning rate of 0.2, a momentum of 0.5 with nesterov's accelerated gradient, and
    also gradient clipping at 50.

    """
    def __init__(self, lr = 0.001, momentum = 0.0, nesterov = False, clip_threshold = np.inf) :
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.clip_threshold = clip_threshold
    def setup(self, nn):
        '''nn is self.layers'''
        self.momentum_params = {}
        for layer_num in range(len(nn)):
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items():
                self.momentum_params[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)

        if self.nesterov :
            for layer_num in range(len(nn)) :
                layer = nn[layer_num]
                layer.nesterov = True
                nn[layer_num] = layer

    def update(self, nn):
        '''nn is self.layers'''
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items() :
                self.momentum_params[param_name + str(layer_num)] = self.momentum * self.momentum_params[param_name + str(layer_num)] \
                 - self.lr * layer.gradients[param_name]
                final_grad = self.momentum_params[param_name + str(layer_num)]
                final_grad[np.where(np.abs(final_grad) > self.clip_threshold)] = self.clip_threshold
                layer.parameters[param_name] += final_grad

class AdaGrad(Optimizer) :
    """
    Slightly more advanced optimizer, an understanding of momentum will be invaluable here. AdaGrad and a whole plethora
    of optimizers use adaptive gradients, or an adaptive learning rate. This just means that it will assess the landscape
    of the cost function, and if it is steep it will slow it down, and if it is flatter it will accelerate. This is a huge
    deal for avoiding gradient descent from just going into a steep slope that leads to a local minima and being stuck,
    or gradient descent being stuck on a saddle point.

    The only new parameter is e, or this incredibly small value that is meant to prevent division by zero. It's set to
    1e-10 by default, and you probably won't ever need to think about it.

    As an example :
    >>> model.finalize(loss = ..., optimizer = nn.optimizers.AdaGrad(lr = 0.5, nesterov = True, clip_threshold = 5))

    AdaGrad is not used in practice much as often times it stops before reaching the global minima due to the gradients
    being too small to make a difference, but we have it anyways for your enjoyment. Better optimizers await!

    """
    def __init__(self, lr = 0.001, nesterov = False, clip_threshold = np.inf, e = 1e-10):
        super().__init__()
        self.lr = lr
        self.nesterov = nesterov
        self.clip_threshold = clip_threshold
        self.e = e
    def setup(self, nn):
        self.momentum_params = {} #the s in page 354 of handsonmlv2
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items() :
                self.momentum_params[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)
        if self.nesterov :
            for layer_num in range(len(nn)) :
                layer = nn[layer_num]
                layer.nesterov = True
                nn[layer_num] = layer
    def update(self, nn):
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name ,_ in layer.parameters.items() :
                self.momentum_params[param_name + str(layer_num)] += np.power(layer.gradients[param_name], 2)
                final_grad = self.lr * layer.gradients[param_name] / (np.sqrt(self.momentum_params[param_name + str(layer_num)]) + self.e)
                final_grad[np.where(np.abs(final_grad) > self.clip_threshold)] = self.clip_threshold
                layer.parameters[param_name] -= final_grad

class RMSProp(Optimizer) :
    """
    RMSprop is a widely known and used algorithm for deep neural network. All it does is solve the problem AdaGrad has
    of stopping too early by not scaling down the gradients so much. It does through a beta parameter, which is set to 0.9
    (does quite well in practice.) A higher beta means that past gradients are more important, and a lower one means current
    gradients are to be valued more.

    An example :
    >>> model.finalize(loss = ..., optimizer = nn.optimizers.RMSprop(nesterov = True, beta = 0.9))

    Of course there is the nesterov, clipping threshold, and e parameter all for you to tune.
    """
    def __init__(self, lr = 0.001, beta = 0.9, nesterov = False, clip_threshold = np.inf, e = 1e-10):
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.nesterov = nesterov
        self.clip_threshold = clip_threshold
        self.e = e
    def setup(self, nn):
        self.momentum_params = {} #the s in page 354 of handsonmlv2
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items() :
                self.momentum_params[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)
        if self.nesterov :
            for layer_num in range(len(nn)) :
                layer = nn[layer_num]
                layer.nesterov = True
                nn[layer_num] = layer
    def update(self, nn):
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name ,_ in layer.parameters.items() :
                self.momentum_params[param_name + str(layer_num)] = self.beta * self.momentum_params[param_name + str(layer_num)] + \
                                                                    (1 - self.beta) * np.power(layer.gradients[param_name], 2)
                final_grad = self.lr * layer.gradients[param_name] / (np.sqrt(self.momentum_params[param_name + str(layer_num)]) + self.e)
                final_grad[np.where(np.abs(final_grad) > self.clip_threshold)] = self.clip_threshold
                layer.parameters[param_name] -= final_grad


class Adam(Optimizer) :
    """
    Most popularly used optimizer, typically considered the default just like ReLU for activation functions.
    Combines the ideas of RMSprop and momentum together, meaning that it will adapt to different landscapes but move
    in a faster direction. The beta1 parameter (default 0.9) controls the momentum optimization,
    and the beta2 parameter (default 0.999) controls the adaptive learning rate parameter here. Once again higher betas
    mean the past gradients are more important.

    Often times you won't know what works best - so hyperparameter tune.

    As an example :
    >>> model.finalize(loss = ..., optimizer = nn.optimizers.Adam(lr = 0.1, beta1 = 0.5, beta2 = 0.5))

    Adaptive Gradients may not always work as good as SGD or Nesterov + Momentum optimization. For MNIST, I have tried
    both and there's barely a difference. If you are using Adam optimization and it isn't working maybe try using
    nesterov with momentum instead.
    """
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, nesterov = False, clip_threshold = np.inf, e = 1e-10):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.nesterov = nesterov
        self.clip_threshold = clip_threshold
        self.e = e
        self.t = 1
    def setup(self, nn):
        self.momentum_M, self.momentum_S = {}, {}
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items() :
                self.momentum_M[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)
                self.momentum_S[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)
        if self.nesterov :
            for layer_num in range(len(nn)) :
                layer = nn[layer_num]
                layer.nesterov = True
                nn[layer_num] = layer

    def update(self, nn):


        for layer_num in range(len(nn)):
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items():
                self.momentum_M[param_name + str(layer_num)] = self.beta1 * self.momentum_M[param_name + str(layer_num)] \
                    - (1 - self.beta1) * layer.gradients[param_name]
                self.momentum_S[param_name + str(layer_num)] = self.beta2 * self.momentum_S[param_name + str(layer_num)] \
                    + (1 - self.beta2)  * np.power(layer.gradients[param_name], 2)
                m_hat = self.momentum_M[param_name + str(layer_num)] / (1 - np.power(self.beta1, self.t))
                s_hat = self.momentum_S[param_name + str(layer_num)] / (1 - np.power(self.beta2, self.t))
                final_grad = self.lr * m_hat / (self.e + np.sqrt(s_hat))
                final_grad[np.where(np.abs(final_grad) > self.clip_threshold)] = self.clip_threshold
                layer.parameters[param_name] += final_grad



class Nadam(Optimizer) :
    """
    Nadam optimization is the same thing as Adam, except there's nesterov updating. Basically this class is the same as
    the Adam class, except there is no nesterov parameter (default true.)

    As an example :
    >>> model.finalize(loss = ..., optimizer = nn.optimizers.Nadam(lr = 0.1, beta1 = 0.5, beta2 = 0.5))
    """
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, clip_threshold = np.inf, e = 1e-10):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.nesterov = True
        self.clip_threshold = clip_threshold
        self.e = e
        self.t = 1
    def setup(self, nn):
        self.momentum_M, self.momentum_S = {}, {}
        for layer_num in range(len(nn)) :
            layer = nn[layer_num]
            for param_name, _ in layer.parameters.items() :
                self.momentum_M[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)
                self.momentum_S[param_name + str(layer_num)] = np.zeros(layer.gradients[param_name].shape)
        if self.nesterov :
            for layer_num in range(len(nn)) :
                layer = nn[layer_num]
                layer.nesterov = True
                nn[layer_num] = layer

    def update(self, nn):
        for layer_num in range(len(nn)) :
            layer =nn[layer_num]
            for param_name, _ in layer.parameters.items() :
                self.momentum_M[param_name + str(layer_num)] = self.beta1 * self.momentum_M[param_name + str(layer_num)] \
                        - (1 - self.beta1) * layer.gradients[param_name]
                self.momentum_S[param_name + str(layer_num)] = self.beta2 * self.momentum_S[param_name + str(layer_num)] \
                         + (1 - self.beta2) * np.power(layer.gradients[param_name], 2)
                m_hat = self.momentum_M[param_name + str(layer_num)] / (1 - np.power(self.beta1, self.t))
                s_hat = self.momentum_S[param_name + str(layer_num)] / (1 - np.power(self.beta2, self.t))
                final_grad = self.lr * m_hat / (np.sqrt(s_hat) + self.e)
                final_grad[np.where(np.abs(final_grad) > self.clip_threshold)] = self.clip_threshold
                layer.gradients[param_name] += final_grad

