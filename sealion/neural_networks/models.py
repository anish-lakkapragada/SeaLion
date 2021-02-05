"""
@author : Anish Lakkapragada
@date : 1 - 23 - 2021

The classes that glue the layers, losses, and optimizers all together. Here you will find the neural network class for
you to build deep neural networks and we also have a map-reduce implementation of neural networks for fun. Highly
recommend you see the examples on GitHub as they will help you more than anything else.
"""

import numpy as np
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import cpu_count
import random

def r2_score(y_pred, y_test):
    num = np.sum(np.power(y_test - y_pred, 2))
    denum = np.sum(np.power(y_test - np.mean(y_test), 2))
    return 1 - num / denum

def _perc_correct(y_pred, y_test) :
    return np.sum((y_pred == y_test).astype('int'))/len(y_pred)

class NeuralNetwork() :
    """
    This class is very rich and packed with methods, so I figured it would be best to have this tutorial guide you
    on a tutorial on the "hello world" of machine learning.

    There are two ways to initialize this model with its layers, through the init() or through the .add() function.

    The first way :
    >>> import neural_networks as nn
    >>> layers = [nn.layers.Flatten(), nn.layers.Dense(784, 64, activation = nn.layers.ReLU()), nn.layers.Dense(64, 32, activation = nn.layers.ReLU()),
    >>> nn.layers.Dense(32, 10, activation = nn.layers.Softmax())]
    >>> model = nn.models.NeuralNetwork(layers)

    Or you can go through the second way :
    >>> import neural_networks as nn
    >>> model = nn.models.NeuralNetwork()
    >>> model.add(nn.layers.Flatten())
    >>> model.add(nn.layers.Dense(784, 64, activation = nn.layers.ReLU()))
    >>> model.add(nn.layers.Dense(64, 32, activation = nn.layers.ReLU()))
    >>> model.add(nn.layers.Dense(32, 10, activation = nn.layers.ReLU()))

    Either way works just fine.

    Next, you will want to perhaps see how complex the model is (having too many parameters means a model can easily
    overfit), so for that you can just do :
    >>> num_params = model.num_parameters() #returns an integer
    >>> assert num_params == 52650

    Looks like our model will be pretty complex. Next up is finalizing and training.
    >>> model.finalize(loss = nn.loss.CrossEntropy(), optimizer = nn.optimizers.Adam())

    Here we use cross-entropy loss for this classification problem and the Adam optimizer.

    Onto training. Assuming our data is 60k * 28 * 28 (sounds a lot like MNIST) in the variable X_train and we have y_train is
    a one-hot encoded matrix size 60k * 10 (10 classes) we can do :

    >>> model.train(X_train, y_train, epochs = 20) #train for 20 epochs

    which will work fine. Here we have a batch_size of 32 (default), and the way we make this run fast is by making
    batch_size datasets, and training them in parallel via multithreading.

    If you want to change batch_size to 18 you could do:
    >>> model.train(X_train, y_train, epochs = 20, batch_size=18) #train for 20 epochs, batch_size 18

    If you want the gradients to be calculated over the entire dataset (this will be much longer) you can do :
    >>> model.full_batch_train(X_train, y_train, epochs = 20)

    Lastly there's also something known as mini-batch gradient descent, which just does gradient descent but randomly chooses
    a percent of the dataset for gradients to be calculated upon. This cannot be parallelized, but still runs fast :
    >>> model.mini_batch_train(X_train, y_train, N = 0.1) #here we take 10% of X_train or 6000 data-points randomly selected for calculating gradients at a time

    All of these methods have a show_loop parameter on whether you want to see the tqdm loop, which is set true by default.

    Now that we have trained our model, time to test and use it.
    To evaluate it, given X_test (shape 10k * 28 * 28) and y_test (shape 10k * 10), we can feed this into
    our evaluate() function :

    >>> model.evaluate(X_test, y_test)

    This gives us the loss, which can be not so interpretable - so we have some other options.

    If you are doing a classification problem as we are here you may do instead :
    >>> model.categorical_evaluate(X_test, y_test)

    which just gives what percent of X_test was classified correctly.

    For regression just do :
    >>> model.regression_evaluate(X_test, y_test)

    which gives the r^2 value of the predictions on X_test. If you are doing this make sure that y_test is not one-hot encoded.

    To predict on data we can just do :
    >>> predictions = model.predict(X_test)

    and if we want this in reverted one-hot-encoded form all we need to do is this :
    >>> from utils import revert_softmax
    >>> predictions = revert_softmax(predictions)

    Storing around 53,000 parameters in matrices isn't much, but for good practice let's store it.
    Using the give_parameters() method we can save our weights in a list :
    >>> parameters = model.give_parameters()

    Then we may want to pickle this using the given method, into a file called "MNIST_pickled_params" :
    >>> file_name = "MNIST_pickled_params"
    >>> model.pickle_params(FILE_NAME=file_name)

    Now this is the beauty - let's say a few weeks from now we come back and realize we probably should train for 20 epochs.
    BUT - we don't want to have to restart training (imagine this was a really big network.)

    So we can just load the weights using the pickle module, build the same model architecture, insert the params, and hit train()!!

    >>> import pickle
    >>> with open('MNIST_pickled_params.pickle', 'rb') as f :
    >>>  params = pickle.load(f)

    Now we can create the model structure (must be the EXACT same) :
    >>> model = nn.models.NeuralNetwork()
    >>> model.add(nn.layers.Flatten())
    >>> model.add(nn.layers.Dense(784, 64, activation = nn.layers.ReLU()))
    >>> model.add(nn.layers.Dense(64, 32, activation = nn.layers.ReLU()))
    >>> model.add(nn.layers.Dense(32, 10, activation = nn.layers.ReLU()))

    Obviously we want to finalize our model for good practice :
    >>> model.finalize(loss = nn.loss.CrossEntropy(), optimizer = nn.optimizers.Adam())

    and we can enter in the weights & biases using the enter_parameters() :
    >>> model.enter_parameters(params) #must be params given from the give_parameters() method

    and train! :
    >>> model.train(X_train, y_train, epochs = 100) #let's try 100 epochs this time

    The reason this is such a big deal is because we don't have to start training from scratch all over again. The model
    will not have to start from 0% accuracy, but may start with 80% given the params belonging to our partially-trained model
    we loaded from the pickle file. This is of course not a big deal for something like MNIST but will be for bigger model architectures, or datasets.

    Of course there's a lot more things we could've changed, but I think that's pretty good for now!
    """

    def __init__(self, layers = None):
        from .layers import Dense, Flatten, Softmax, Dropout
        from .loss import CrossEntropy, softmax
        from .optimizers import Adam, SGD
        from .utils_nn import revert_one_hot_nn, revert_softmax_nn

        self.Dense = Dense
        self.Flatten = Flatten
        self.Softmax = Softmax
        self.Dropout = Dropout
        self.CrossEntropy = CrossEntropy
        self.softmax = softmax
        self.Adam = Adam
        self.SGD = SGD
        self.revert_one_hot = revert_one_hot_nn
        self.revert_softmax = revert_softmax_nn

        self.layers = []
        if layers : #self.layers = layers
            for layer in layers :
                self.layers.append(layer)
                if layer.activation :
                    self.layers.append(layer.activation)
        self.finalized = False
        self.sgd_indices = ...
        self.adam_t = 0

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=np.ComplexWarning)

    def num_parameters(self):
        num_params = 0
        for layer in self.layers :
            if isinstance(layer, self.Dense) :
                num_params += np.product(layer.parameters['weights'].shape) + layer.parameters['bias'].shape
        return num_params
    def enter_parameters(self, parameters) :
        for layer_num in range(len(self.layers)) :
            try :
                if parameters[layer_num]['weights'] == None and parameters[layer_num]['bias'] == None: continue
            except Exception :
                self.layers[layer_num].parameters['weights'] = parameters[layer_num]['weights']
                self.layers[layer_num].parameters['bias'] = parameters[layer_num]['bias']

    def give_parameters(self):
        parameters = []
        for layer in self.layers :
            try :
                parameters.append({'weights' : layer.parameters['weights'], 'bias' : layer.parameters['bias']})
            except Exception :
                parameters.append({'weights' : None, 'bias' : None})
        return parameters

    def pickle_params(self, FILE_NAME = "NeuralNetwork_learnt_parameters"):
        parameters = NeuralNetwork.give_parameters(self)
        import pickle

        with open(FILE_NAME + '.pickle', 'wb') as f:
            pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def add(self, layer):
        self.layers.append(layer)
        if layer.activation :
            self.layers.append(layer.activation)

    def forward(self, inputs):
        for layer in self.layers :
            if isinstance(layer, self.Softmax) : continue
            inputs = layer.forward(inputs)
        return inputs
    def backward(self, grad):
        for layer in reversed(self.layers) :
            if isinstance(layer, self.Softmax) : continue
            grad = layer.backward(grad)
        return grad

    def finalize(self, loss, optimizer) :
        '''Both of these have to be the classes for the loss and optimizations'''
        self.loss, self.optimizer = loss, optimizer
        self.finalized = True

    def _sgd_updating_indices(self):
        '''only call if the loss is SGD'''

        for layer_num in range(len(self.layers)) :
            layer = self.layers[layer_num]
            layer.indices = self.sgd_indices
            self.layers[layer_num]= layer

    def _chunk_train(self, data_chunk, label_chunk):
        '''the actual training code!'''
        predicted = NeuralNetwork.forward(self, data_chunk)
        for layer_num in range(len(self.layers)):
            if self.layers[layer_num].nesterov:
                '''nesterov accel'''
                try:
                    self.layers[layer_num].parameters['weights'] += self.optimizer.momentum * \
                                                                    self.optimizer.momentum_params[
                                                                        'weights' + str(layer_num)]
                    self.layers[layer_num].parameters['bias'] += self.optimizer.momentum * \
                                                                 self.optimizer.momentum_params['bias' + str(layer_num)]
                except Exception:
                    pass

        if isinstance(self.optimizer, self.Adam) :
            self.adam_t += 1
            self.optimizer.t = self.adam_t

        if isinstance(self.optimizer, self.SGD) :
            '''select indices for the gradients and update all layers'''
            start_index = np.random.randint(len(predicted) - 3)
            self.sgd_indices = slice(start_index, len(predicted) - 1)
            NeuralNetwork._sgd_updating_indices(self)


        grad = self.loss.grad(label_chunk[self.sgd_indices], predicted[self.sgd_indices])  # calculate the dL/dY
        NeuralNetwork.backward(self, grad)
        self.optimizer.update(self.layers)

    def train(self, x_train, y_train, epochs=1, batch_size=32, show_loop = True):
        if not self.finalized : raise ValueError("This model isn't finalized. Call it through model.finalize().")

        if isinstance(self.layers[-1], self.Softmax) and not isinstance(self.loss, self.CrossEntropy) :
            raise ValueError("If the last layer activation is softmax, you must be using CrossEntropy loss.")

        if isinstance(self.loss, self.CrossEntropy) and not isinstance(self.layers[-1], self.Softmax) :
            raise ValueError("If the loss function is crossentropy, you must be using Softmax as your last layer.")

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(y_train.shape) != 2: raise ValueError("y_train must be 2D.")
        if not isinstance(self.layers[0], self.Flatten) and len(x_train.shape) > 2:
            raise ValueError("There must be a Flatten layer in the beginning if you are working with data that is not "
                             "a matrix (e.g. something that is 60k * 28 * 28 not 60k * 784)")

        # perform batch operations
        if batch_size > x_train.shape[0]:
            warnings.warn("Batch size is more than the number of samples, so we have set batch_size = 1 and are just performing full batch gradient descent.")
            batch_size = 1

        x_train = np.array_split(x_train, batch_size)
        y_train = np.array_split(y_train, batch_size)
        self.optimizer.setup(self.layers)

        epochs = tqdm(range(epochs), position=0, ncols=100) if show_loop else range(epochs)
        evaluation_batch = random.randint(0, len(x_train) - 1)
        evaluation_method = NeuralNetwork.regression_evaluate if self.loss.type_regression else NeuralNetwork.categorical_evaluate

        for epoch in epochs :
            Parallel(prefer = "threads")(delayed(NeuralNetwork._chunk_train)(self, data_chunk, label_chunk) for data_chunk, label_chunk in
                                         zip(x_train, y_train))

            if show_loop :
             epochs.set_description("Acc : " + str(round(evaluation_method(self, x_train[evaluation_batch], y_train[evaluation_batch])* 100, 2)) + "%")

        # adjust dropout
        for layer_num in range(len(self.layers)) :
            '''adjust the weights and biases if there is dropout'''
            try :
                layer = self.layers[layer_num]
                if isinstance(layer, self.Dropout) :
                    receiving_layer = self.layers[layer_num + 1] #this is the layer that receives the dropout
                    receiving_layer.parameters['weights'] *= (1 - layer.dr)
                    receiving_layer.parameters['bias'] *= (1 - layer.dr)
                    self.layers[layer_num + 1] = receiving_layer
            except Exception :
                pass

    def full_batch_train(self, x_train, y_train, epochs = 1, show_loop = True):
        if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

        if isinstance(self.layers[-1], self.Softmax) and not isinstance(self.loss, self.CrossEntropy):
            raise ValueError("If the last layer activation is softmax, you must be using CrossEntropy loss.")

        if isinstance(self.loss, self.CrossEntropy) and not isinstance(self.layers[-1], self.Softmax):
            raise ValueError("If the loss function is crossentropy, you must be using Softmax as your last layer.")

        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(y_train.shape) != 2: raise ValueError("y_train must be 2D.")
        if not isinstance(self.layers[0], self.Flatten) and len(x_train.shape) > 2:
            raise ValueError("There must be a Flatten layer in the beginning if you are working with data that is not "
                             "a matrix (e.g. something that is 60k * 28 * 28 not 60k * 784)")

        epochs = tqdm(range(epochs), position = 0, ncols = 100) if show_loop else range(epochs)
        evaluation_method = NeuralNetwork.regression_evaluate if self.loss.type_regression else NeuralNetwork.categorical_evaluate
        self.optimizer.setup(self.layers) #setup the optimizer

        for epoch in epochs :
            NeuralNetwork._chunk_train(self, x_train, y_train)

            if show_loop == True :
                epochs.set_description("Acc : " + str(round(evaluation_method(self, x_train, y_train) * 100, 2)) + "%")


        # adjust dropout
        for layer_num in range(len(self.layers)):
            '''adjust the weights and biases if there is dropout'''
            try:
                layer = self.layers[layer_num]
                if isinstance(layer, self.Dropout):
                    receiving_layer = self.layers[layer_num + 1]  # this is the layer that receives the dropout
                    receiving_layer.parameters['weights'] *= (1 - layer.dr)
                    receiving_layer.parameters['bias'] *= (1 - layer.dr)
                    self.layers[layer_num + 1] = receiving_layer
            except Exception:
                pass

    def mini_batch_train(self, x_train, y_train, epochs = 1, N = 0.2, show_loop = True) :
        if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

        if isinstance(self.layers[-1], self.Softmax) and not isinstance(self.loss, self.CrossEntropy):
            raise ValueError("If the last layer activation is softmax, you must be using CrossEntropy loss.")

        if isinstance(self.loss, self.CrossEntropy) and not isinstance(self.layers[-1], self.Softmax):
            raise ValueError("If the loss function is crossentropy, you must be using Softmax as your last layer.")


        x_train, y_train = np.array(x_train), np.array(y_train)
        if len(y_train.shape) != 2 : raise ValueError("y_train must be 2D.")
        if not isinstance(self.layers[0], self.Flatten) and len(x_train.shape) > 2:
            raise ValueError("There must be a Flatten layer in the beginning if you are working with data that is not "
                             "a matrix (e.g. something that is 60k * 28 * 28 not 60k * 784)")

        if N == 1 : raise ValueError("N cannot be equal to 1, has to be between 0 and 1.")

        N *= len(x_train)
        N = round(N)

        epochs = tqdm(range(epochs), position = 0, ncols = 100) if show_loop else range(epochs)
        evaluation_method = NeuralNetwork.regression_evaluate if self.loss.type_regression else NeuralNetwork.categorical_evaluate
        self.optimizer.setup(self.layers)

        for epoch in epochs:

            #set up the mini-batch
            start_index = random.randint(0, len(x_train) - N - 1)
            end_index = start_index + N
            data_chunk, label_chunk = x_train[start_index : end_index], y_train[start_index : end_index]

            NeuralNetwork._chunk_train(self, data_chunk, label_chunk)

            if show_loop == True:
                epochs.set_description("Acc : " + str(round(evaluation_method(self, x_train, y_train) * 100, 2)) + "%")

        #adjust dropout
        for layer_num in range(len(self.layers)) :
            '''adjust the weights and biases if there is dropout'''
            try :
                layer = self.layers[layer_num]
                if isinstance(layer, self.Dropout) :
                    receiving_layer = self.layers[layer_num + 1] #this is the layer that receives the dropout
                    receiving_layer.parameters['weights'] *= (1 - layer.dr)
                    receiving_layer.parameters['bias'] *= (1 - layer.dr)
                    self.layers[layer_num + 1] = receiving_layer
            except Exception :
                pass


    def predict(self, x_test):

        if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")


        x_test = np.array(x_test)
        y_pred = NeuralNetwork.forward(self, x_test)
        if isinstance(self.layers[-1], self.Softmax) : y_pred = self.softmax(y_pred)

        if self.loss.type_regression :
            return y_pred
        else :
            return np.round_(y_pred)


    def evaluate(self, x_test, y_test):
        if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

        y_pred = NeuralNetwork.predict(self, x_test)

        return self.loss.loss(y_test, y_pred)

    def regression_evaluate(self, x_test, y_test):
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

        y_pred = NeuralNetwork.predict(self, x_test)

        return np.mean([r2_score(y_pred[:, col].flatten(), y_test[:, col].flatten()) for col in range(y_pred.shape[1])]) #mean of each columns r^2

    def categorical_evaluate(self, x_test, y_test) :
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

        y_pred = NeuralNetwork.predict(self, x_test)

        return _perc_correct(self.revert_softmax(y_pred).flatten(), self.revert_one_hot(y_test).flatten())

class NeuralNetwork_MapReduce():
    """
    Map Reduce is a key algorithm to training neural networks these days. Maybe not as complicated as some other strategies,
    but as usual I couldn't resist the urge to try to see how this could work.

    Map Reduce is used to train datasets a lot faster - utilizing the multiple cores your computer might have. What it does
    is take a dataset and chop into however many cores you want to use (default is all of them), and then have each core train
    a model given that data. So if you use 16 cores, you will essentially have 64 models built on datasets that each are 1/16th
    the size of the given data.

    Then at test time, predictions will be ran through each model and will be averaged/rounded depending on whether its a
    regression or classification task. All the other methods of add(), finalize(), are mostly the same as with the
    NeuralNetwork() class.

    Let's say we are working with MNIST in variables X_train, and y_train. Well to first declare and finalize our model
    we could do :

    >>> import neural_networks as nn
    >>> model = nn.models.NeuralNetwork_MapReduce()
    >>> model.add(nn.layers.Flatten())
    >>> model.add(nn.layers.Dense(784, 64, activation = nn.layers.ReLU()))
    >>> model.add(nn.layers.Dense(64, 32, activation = nn.layers.ReLU()))
    >>> model.add(nn.layers.Dense(32, 10, activation = nn.layers.ReLU()))
    >>> model.finalize(loss = nn.loss.CrossEntropy(), optimizer = nn.optimizers.Adam())

    to train() it we could then do :
    >>> if model.train(X_train, y_train, epochs=20, num_cores=-1)

    which will work. I feel like I need to explain what num_cores is. It is simply the number of cores you want
    to be utilized. By default it is -1 which means all available cores. -2 would mean using all but 1, and -3 would
    mean using all but 2, -4 would mean using all but 3, etc. You could also set it to a positive number like 16, 100 -
    which is just the amount of cores utilized by the model. Another note is that this class doesn't have the show_loop
    argument.

    to predict() and evaluate() you could just do :
    >>> pred = model.predict(X_test)
    >>> categorical_evaluation = model.categorical_evaluate(X_test, y_test)

    to save the weights in a pickle file you could do :
    >>> file_name = "MNIST_MapReduce"
    >>> model.pickle_best_parameters(X_test, y_test, WEIGHT_FILE_NAME=file_name)

    Here what is going on is that the parameters you are getting pickled are the params of the best model.
    This makes MapReduce also useful as often times random initialization of parameters can effect the performance
    of the model, so running it a couple of times on many cores may help you find a good weight init.

    There's another method of give_best_parameters() which is also very similar to the pickle_best_parameters().

    This class was created to see what could happen, and I'm adding it here because hopefully you will find use in training
    many models to find the best model or using this to utilize all of your CPU cores. If you find a bug or an error,
    as usual please notify on Github or email me at anish.lakkapragada@gmail.com

    """
    def __init__(self, layers=None):
        from .utils_nn import revert_one_hot_nn, revert_softmax_nn
        self.revert_softmax = revert_softmax_nn
        self.revert_one_hot = revert_one_hot_nn
        self.layers = []

        if layers:  # self.layers = layers
            for layer in layers:
                self.layers.append(layer)
                if layer.activation:
                    self.layers.append(layer.activation)
        self.finalized = False

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=np.ComplexWarning)
    def add(self, layer):
        self.layers.append(layer)
        if layer.activation :
            self.layers.append(layer.activation)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def finalize(self, loss, optimizer):
        '''Both of these have to be the classes for the loss and optimizations'''
        self.loss, self.optimizer = loss, optimizer
        self.finalized = True

    def num_parameters(self):
        return self.models[0].num_parameters()

    def _train_individual_model(self, model_num) :
        '''here we train the individual model for the MapReduce algo'''
        model = self.models[model_num]
        data_chunk, label_chunk = self.x_train[model_num], self.y_train[model_num]
        model.full_batch_train(data_chunk, label_chunk, epochs=self.epochs, show_loop=False)
        self.models[model_num] = model #update it to the train algorithm

    def give_best_parameters(self, x_test, y_test) :
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")
        models = {} #evaluation_score : class
        for model in self.models  :
            score = model.evaluate(x_test, y_test)
            models[score] = model
        best_model = models[min(models)]
        return best_model.give_parameters()

    def pickle_best_parameters(self, x_test, y_test, FILE_NAME =  "NeuralNetwork_learnt_parameters"):
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")
        best_params = NeuralNetwork_MapReduce.give_best_parameters(self, x_test, y_test)
        """
        @author : Anish Lakkapragada
        @date : 1 - 23 - 2021

        The classes that glue the layers, losses, and optimizers all together. Here you will find the neural network class for
        you to build deep neural networks and we also have a map-reduce implementation of neural networks for fun. Highly
        recommend you see the examples on GitHub as they will help you more than anything else.
        """

        import numpy as np
        import warnings
        from joblib import Parallel, delayed
        from tqdm import tqdm
        from multiprocessing import cpu_count
        import random

        def r2_score(y_pred, y_test):
            num = np.sum(np.power(y_test - y_pred, 2))
            denum = np.sum(np.power(y_test - np.mean(y_test), 2))
            return 1 - num / denum

        def _perc_correct(y_pred, y_test):
            return np.sum((y_pred == y_test).astype('int')) / len(y_pred)

        class NeuralNetwork():
            """
            This class is very rich and packed with methods, so I figured it would be best to have this tutorial guide you
            on a tutorial on the "hello world" of machine learning.

            There are two ways to initialize this model with its layers, through the init() or through the .add() function.

            The first way :
            >>> import neural_networks as nn
            >>> layers = [nn.layers.Flatten(), nn.layers.Dense(784, 64, activation = nn.layers.ReLU()), nn.layers.Dense(64, 32, activation = nn.layers.ReLU()),
            >>> nn.layers.Dense(32, 10, activation = nn.layers.Softmax())]
            >>> model = nn.models.NeuralNetwork(layers)

            Or you can go through the second way :
            >>> import neural_networks as nn
            >>> model = nn.models.NeuralNetwork()
            >>> model.add(nn.layers.Flatten())
            >>> model.add(nn.layers.Dense(784, 64, activation = nn.layers.ReLU()))
            >>> model.add(nn.layers.Dense(64, 32, activation = nn.layers.ReLU()))
            >>> model.add(nn.layers.Dense(32, 10, activation = nn.layers.ReLU()))

            Either way works just fine.

            Next, you will want to perhaps see how complex the model is (having too many parameters means a model can easily
            overfit), so for that you can just do :
            >>> num_params = model.num_parameters() #returns an integer
            >>> assert num_params == 52650

            Looks like our model will be pretty complex. Next up is finalizing and training.
            >>> model.finalize(loss = nn.loss.CrossEntropy(), optimizer = nn.optimizers.Adam())

            Here we use cross-entropy loss for this classification problem and the Adam optimizer.

            Onto training. Assuming our data is 60k * 28 * 28 (sounds a lot like MNIST) in the variable X_train and we have y_train is
            a one-hot encoded matrix size 60k * 10 (10 classes) we can do :

            >>> model.train(X_train, y_train, epochs = 20) #train for 20 epochs

            which will work fine. Here we have a batch_size of 32 (default), and the way we make this run fast is by making
            batch_size datasets, and training them in parallel via multithreading.

            If you want to change batch_size to 18 you could do:
            >>> model.train(X_train, y_train, epochs = 20, batch_size=18) #train for 20 epochs, batch_size 18

            If you want the gradients to be calculated over the entire dataset (this will be much longer) you can do :
            >>> model.full_batch_train(X_train, y_train, epochs = 20)

            Lastly there's also something known as mini-batch gradient descent, which just does gradient descent but randomly chooses
            a percent of the dataset for gradients to be calculated upon. This cannot be parallelized, but still runs fast :
            >>> model.mini_batch_train(X_train, y_train, N = 0.1) #here we take 10% of X_train or 6000 data-points randomly selected for calculating gradients at a time

            All of these methods have a show_loop parameter on whether you want to see the tqdm loop, which is set true by default.

            Now that we have trained our model, time to test and use it.
            To evaluate it, given X_test (shape 10k * 28 * 28) and y_test (shape 10k * 10), we can feed this into
            our evaluate() function :

            >>> model.evaluate(X_test, y_test)

            This gives us the loss, which can be not so interpretable - so we have some other options.

            If you are doing a classification problem as we are here you may do instead :
            >>> model.categorical_evaluate(X_test, y_test)

            which just gives what percent of X_test was classified correctly.

            For regression just do :
            >>> model.regression_evaluate(X_test, y_test)

            which gives the r^2 value of the predictions on X_test. If you are doing this make sure that y_test is not one-hot encoded.

            To predict on data we can just do :
            >>> predictions = model.predict(X_test)

            and if we want this in reverted one-hot-encoded form all we need to do is this :
            >>> from utils import revert_softmax
            >>> predictions = revert_softmax(predictions)

            Storing around 53,000 parameters in matrices isn't much, but for good practice let's store it.
            Using the give_parameters() method we can save our weights in a list :
            >>> parameters = model.give_parameters()

            Then we may want to pickle this using the given method, into a file called "MNIST_pickled_params" :
            >>> file_name = "MNIST_pickled_params"
            >>> model.pickle_params(FILE_NAME=file_name)

            Now this is the beauty - let's say a few weeks from now we come back and realize we probably should train for 20 epochs.
            BUT - we don't want to have to restart training (imagine this was a really big network.)

            So we can just load the weights using the pickle module, build the same model architecture, insert the params, and hit train()!!

            >>> import pickle
            >>> with open('MNIST_pickled_params.pickle', 'rb') as f :
            >>>  params = pickle.load(f)

            Now we can create the model structure (must be the EXACT same) :
            >>> model = nn.models.NeuralNetwork()
            >>> model.add(nn.layers.Flatten())
            >>> model.add(nn.layers.Dense(784, 64, activation = nn.layers.ReLU()))
            >>> model.add(nn.layers.Dense(64, 32, activation = nn.layers.ReLU()))
            >>> model.add(nn.layers.Dense(32, 10, activation = nn.layers.ReLU()))

            Obviously we want to finalize our model for good practice :
            >>> model.finalize(loss = nn.loss.CrossEntropy(), optimizer = nn.optimizers.Adam())

            and we can enter in the weights & biases using the enter_parameters() :
            >>> model.enter_parameters(params) #must be params given from the give_parameters() method

            and train! :
            >>> model.train(X_train, y_train, epochs = 100) #let's try 100 epochs this time

            The reason this is such a big deal is because we don't have to start training from scratch all over again. The model
            will not have to start from 0% accuracy, but may start with 80% given the params belonging to our partially-trained model
            we loaded from the pickle file. This is of course not a big deal for something like MNIST but will be for bigger model architectures, or datasets.

            Of course there's a lot more things we could've changed, but I think that's pretty good for now!
            """

            def __init__(self, layers=None):
                from .layers import Dense, Flatten, Softmax, Dropout
                from .loss import CrossEntropy, softmax
                from .optimizers import Adam, SGD
                from .utils_nn import revert_one_hot_nn, revert_softmax_nn

                self.Dense = Dense
                self.Flatten = Flatten
                self.Softmax = Softmax
                self.Dropout = Dropout
                self.CrossEntropy = CrossEntropy
                self.softmax = softmax
                self.Adam = Adam
                self.SGD = SGD
                self.revert_one_hot = revert_one_hot_nn
                self.revert_softmax = revert_softmax_nn

                self.layers = []
                if layers:  # self.layers = layers
                    for layer in layers:
                        self.layers.append(layer)
                        if layer.activation:
                            self.layers.append(layer.activation)
                self.finalized = False
                self.sgd_indices = ...
                self.adam_t = 0

                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=np.ComplexWarning)

            def num_parameters(self):
                num_params = 0
                for layer in self.layers:
                    if isinstance(layer, self.Dense):
                        num_params += np.product(layer.parameters['weights'].shape) + layer.parameters['bias'].shape
                return num_params

            def enter_parameters(self, parameters):
                for layer_num in range(len(self.layers)):
                    try:
                        if parameters[layer_num]['weights'] == None and parameters[layer_num]['bias'] == None: continue
                    except Exception:
                        self.layers[layer_num].parameters['weights'] = parameters[layer_num]['weights']
                        self.layers[layer_num].parameters['bias'] = parameters[layer_num]['bias']

            def give_parameters(self):
                parameters = []
                for layer in self.layers:
                    try:
                        parameters.append({'weights': layer.parameters['weights'], 'bias': layer.parameters['bias']})
                    except Exception:
                        parameters.append({'weights': None, 'bias': None})
                return parameters

            def pickle_params(self, FILE_NAME="NeuralNetwork_learnt_parameters"):
                parameters = NeuralNetwork.give_parameters(self)
                import pickle

                with open(FILE_NAME + '.pickle', 'wb') as f:
                    pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

            def add(self, layer):
                self.layers.append(layer)
                if layer.activation:
                    self.layers.append(layer.activation)

            def forward(self, inputs):
                for layer in self.layers:
                    if isinstance(layer, self.Softmax): continue
                    inputs = layer.forward(inputs)
                return inputs

            def backward(self, grad):
                for layer in reversed(self.layers):
                    if isinstance(layer, self.Softmax): continue
                    grad = layer.backward(grad)
                return grad

            def finalize(self, loss, optimizer):
                '''Both of these have to be the classes for the loss and optimizations'''
                self.loss, self.optimizer = loss, optimizer
                self.finalized = True

            def _sgd_updating_indices(self):
                '''only call if the loss is SGD'''

                for layer_num in range(len(self.layers)):
                    layer = self.layers[layer_num]
                    layer.indices = self.sgd_indices
                    self.layers[layer_num] = layer

            def _chunk_train(self, data_chunk, label_chunk):
                '''the actual training code!'''
                predicted = NeuralNetwork.forward(self, data_chunk)
                for layer_num in range(len(self.layers)):
                    if self.layers[layer_num].nesterov:
                        '''nesterov accel'''
                        try:
                            self.layers[layer_num].parameters['weights'] += self.optimizer.momentum * \
                                                                            self.optimizer.momentum_params[
                                                                                'weights' + str(layer_num)]
                            self.layers[layer_num].parameters['bias'] += self.optimizer.momentum * \
                                                                         self.optimizer.momentum_params[
                                                                             'bias' + str(layer_num)]
                        except Exception:
                            pass

                if isinstance(self.optimizer, self.Adam):
                    self.adam_t += 1
                    self.optimizer.t = self.adam_t

                if isinstance(self.optimizer, self.SGD):
                    '''select indices for the gradients and update all layers'''
                    start_index = np.random.randint(len(predicted) - 3)
                    self.sgd_indices = slice(start_index, len(predicted) - 1)
                    NeuralNetwork._sgd_updating_indices(self)

                grad = self.loss.grad(label_chunk[self.sgd_indices], predicted[self.sgd_indices])  # calculate the dL/dY
                NeuralNetwork.backward(self, grad)
                self.optimizer.update(self.layers)

            def train(self, x_train, y_train, epochs=1, batch_size=32, show_loop=True):
                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                if isinstance(self.layers[-1], self.Softmax) and not isinstance(self.loss, self.CrossEntropy):
                    raise ValueError("If the last layer activation is softmax, you must be using CrossEntropy loss.")

                if isinstance(self.loss, self.CrossEntropy) and not isinstance(self.layers[-1], self.Softmax):
                    raise ValueError(
                        "If the loss function is crossentropy, you must be using Softmax as your last layer.")

                x_train, y_train = np.array(x_train), np.array(y_train)
                if len(y_train.shape) != 2: raise ValueError("y_train must be 2D.")
                if not isinstance(self.layers[0], self.Flatten) and len(x_train.shape) > 2:
                    raise ValueError(
                        "There must be a Flatten layer in the beginning if you are working with data that is not "
                        "a matrix (e.g. something that is 60k * 28 * 28 not 60k * 784)")

                # perform batch operations
                if batch_size > x_train.shape[0]:
                    warnings.warn(
                        "Batch size is more than the number of samples, so we have set batch_size = 1 and are just performing full batch gradient descent.")
                    batch_size = 1

                x_train = np.array_split(x_train, batch_size)
                y_train = np.array_split(y_train, batch_size)
                self.optimizer.setup(self.layers)

                epochs = tqdm(range(epochs), position=0, ncols=100) if show_loop else range(epochs)
                evaluation_batch = random.randint(0, len(x_train) - 1)
                evaluation_method = NeuralNetwork.regression_evaluate if self.loss.type_regression else NeuralNetwork.categorical_evaluate

                for epoch in epochs:
                    Parallel(prefer="threads")(
                        delayed(NeuralNetwork._chunk_train)(self, data_chunk, label_chunk) for data_chunk, label_chunk
                        in
                        zip(x_train, y_train))

                    if show_loop:
                        epochs.set_description("Acc : " + str(
                            round(evaluation_method(self, x_train[evaluation_batch], y_train[evaluation_batch]) * 100,
                                  2)) + "%")

                # adjust dropout
                for layer_num in range(len(self.layers)):
                    '''adjust the weights and biases if there is dropout'''
                    try:
                        layer = self.layers[layer_num]
                        if isinstance(layer, self.Dropout):
                            receiving_layer = self.layers[layer_num + 1]  # this is the layer that receives the dropout
                            receiving_layer.parameters['weights'] *= (1 - layer.dr)
                            receiving_layer.parameters['bias'] *= (1 - layer.dr)
                            self.layers[layer_num + 1] = receiving_layer
                    except Exception:
                        pass

            def full_batch_train(self, x_train, y_train, epochs=1, show_loop=True):
                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                if isinstance(self.layers[-1], self.Softmax) and not isinstance(self.loss, self.CrossEntropy):
                    raise ValueError("If the last layer activation is softmax, you must be using CrossEntropy loss.")

                if isinstance(self.loss, self.CrossEntropy) and not isinstance(self.layers[-1], self.Softmax):
                    raise ValueError(
                        "If the loss function is crossentropy, you must be using Softmax as your last layer.")

                x_train, y_train = np.array(x_train), np.array(y_train)
                if len(y_train.shape) != 2: raise ValueError("y_train must be 2D.")
                if not isinstance(self.layers[0], self.Flatten) and len(x_train.shape) > 2:
                    raise ValueError(
                        "There must be a Flatten layer in the beginning if you are working with data that is not "
                        "a matrix (e.g. something that is 60k * 28 * 28 not 60k * 784)")

                epochs = tqdm(range(epochs), position=0, ncols=100) if show_loop else range(epochs)
                evaluation_method = NeuralNetwork.regression_evaluate if self.loss.type_regression else NeuralNetwork.categorical_evaluate
                self.optimizer.setup(self.layers)  # setup the optimizer

                for epoch in epochs:
                    NeuralNetwork._chunk_train(self, x_train, y_train)

                    if show_loop == True:
                        epochs.set_description(
                            "Acc : " + str(round(evaluation_method(self, x_train, y_train) * 100, 2)) + "%")

                # adjust dropout
                for layer_num in range(len(self.layers)):
                    '''adjust the weights and biases if there is dropout'''
                    try:
                        layer = self.layers[layer_num]
                        if isinstance(layer, self.Dropout):
                            receiving_layer = self.layers[layer_num + 1]  # this is the layer that receives the dropout
                            receiving_layer.parameters['weights'] *= (1 - layer.dr)
                            receiving_layer.parameters['bias'] *= (1 - layer.dr)
                            self.layers[layer_num + 1] = receiving_layer
                    except Exception:
                        pass

            def mini_batch_train(self, x_train, y_train, epochs=1, N=0.2, show_loop=True):
                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                if isinstance(self.layers[-1], self.Softmax) and not isinstance(self.loss, self.CrossEntropy):
                    raise ValueError("If the last layer activation is softmax, you must be using CrossEntropy loss.")

                if isinstance(self.loss, self.CrossEntropy) and not isinstance(self.layers[-1], self.Softmax):
                    raise ValueError(
                        "If the loss function is crossentropy, you must be using Softmax as your last layer.")

                x_train, y_train = np.array(x_train), np.array(y_train)
                if len(y_train.shape) != 2: raise ValueError("y_train must be 2D.")
                if not isinstance(self.layers[0], self.Flatten) and len(x_train.shape) > 2:
                    raise ValueError(
                        "There must be a Flatten layer in the beginning if you are working with data that is not "
                        "a matrix (e.g. something that is 60k * 28 * 28 not 60k * 784)")

                if N == 1: raise ValueError("N cannot be equal to 1, has to be between 0 and 1.")

                N *= len(x_train)
                N = round(N)

                epochs = tqdm(range(epochs), position=0, ncols=100) if show_loop else range(epochs)
                evaluation_method = NeuralNetwork.regression_evaluate if self.loss.type_regression else NeuralNetwork.categorical_evaluate
                self.optimizer.setup(self.layers)

                for epoch in epochs:

                    # set up the mini-batch
                    start_index = random.randint(0, len(x_train) - N - 1)
                    end_index = start_index + N
                    data_chunk, label_chunk = x_train[start_index: end_index], y_train[start_index: end_index]

                    NeuralNetwork._chunk_train(self, data_chunk, label_chunk)

                    if show_loop == True:
                        epochs.set_description(
                            "Acc : " + str(round(evaluation_method(self, x_train, y_train) * 100, 2)) + "%")

                # adjust dropout
                for layer_num in range(len(self.layers)):
                    '''adjust the weights and biases if there is dropout'''
                    try:
                        layer = self.layers[layer_num]
                        if isinstance(layer, self.Dropout):
                            receiving_layer = self.layers[layer_num + 1]  # this is the layer that receives the dropout
                            receiving_layer.parameters['weights'] *= (1 - layer.dr)
                            receiving_layer.parameters['bias'] *= (1 - layer.dr)
                            self.layers[layer_num + 1] = receiving_layer
                    except Exception:
                        pass

            def predict(self, x_test):

                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                x_test = np.array(x_test)
                y_pred = NeuralNetwork.forward(self, x_test)
                if isinstance(self.layers[-1], self.Softmax): y_pred = self.softmax(y_pred)

                if self.loss.type_regression:
                    return y_pred
                else:
                    return np.round_(y_pred)

            def evaluate(self, x_test, y_test):
                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                x_test, y_test = np.array(x_test), np.array(y_test)
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

                y_pred = NeuralNetwork.predict(self, x_test)

                return self.loss.loss(y_test, y_pred)

            def regression_evaluate(self, x_test, y_test):
                x_test, y_test = np.array(x_test), np.array(y_test)
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

                y_pred = NeuralNetwork.predict(self, x_test)

                return np.mean([r2_score(y_pred[:, col].flatten(), y_test[:, col].flatten()) for col in
                                range(y_pred.shape[1])])  # mean of each columns r^2

            def categorical_evaluate(self, x_test, y_test):
                x_test, y_test = np.array(x_test), np.array(y_test)
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

                y_pred = NeuralNetwork.predict(self, x_test)

                return _perc_correct(self.revert_softmax(y_pred).flatten(), self.revert_one_hot(y_test).flatten())

        class NeuralNetwork_MapReduce():
            """
            Map Reduce is a key algorithm to training neural networks these days. Maybe not as complicated as some other strategies,
            but as usual I couldn't resist the urge to try to see how this could work.

            Map Reduce is used to train datasets a lot faster - utilizing the multiple cores your computer might have. What it does
            is take a dataset and chop into however many cores you want to use (default is all of them), and then have each core train
            a model given that data. So if you use 16 cores, you will essentially have 64 models built on datasets that each are 1/16th
            the size of the given data.

            Then at test time, predictions will be ran through each model and will be averaged/rounded depending on whether its a
            regression or classification task. All the other methods of add(), finalize(), are mostly the same as with the
            NeuralNetwork() class.

            Let's say we are working with MNIST in variables X_train, and y_train. Well to first declare and finalize our model
            we could do :

            >>> import neural_networks as nn
            >>> model = nn.models.NeuralNetwork_MapReduce()
            >>> model.add(nn.layers.Flatten())
            >>> model.add(nn.layers.Dense(784, 64, activation = nn.layers.ReLU()))
            >>> model.add(nn.layers.Dense(64, 32, activation = nn.layers.ReLU()))
            >>> model.add(nn.layers.Dense(32, 10, activation = nn.layers.ReLU()))
            >>> model.finalize(loss = nn.loss.CrossEntropy(), optimizer = nn.optimizers.Adam())

            to train() it we could then do :
            >>> if model.train(X_train, y_train, epochs=20, num_cores=-1)

            which will work. I feel like I need to explain what num_cores is. It is simply the number of cores you want
            to be utilized. By default it is -1 which means all available cores. -2 would mean using all but 1, and -3 would
            mean using all but 2, -4 would mean using all but 3, etc. You could also set it to a positive number like 16, 100 -
            which is just the amount of cores utilized by the model. Another note is that this class doesn't have the show_loop
            argument.

            to predict() and evaluate() you could just do :
            >>> pred = model.predict(X_test)
            >>> categorical_evaluation = model.categorical_evaluate(X_test, y_test)

            to save the weights in a pickle file you could do :
            >>> file_name = "MNIST_MapReduce"
            >>> model.pickle_best_parameters(X_test, y_test, WEIGHT_FILE_NAME=file_name)

            Here what is going on is that the parameters you are getting pickled are the params of the best model.
            This makes MapReduce also useful as often times random initialization of parameters can effect the performance
            of the model, so running it a couple of times on many cores may help you find a good weight init.

            There's another method of give_best_parameters() which is also very similar to the pickle_best_parameters().

            This class was created to see what could happen, and I'm adding it here because hopefully you will find use in training
            many models to find the best model or using this to utilize all of your CPU cores. If you find a bug or an error,
            as usual please notify on Github or email me at anish.lakkapragada@gmail.com

            """

            def __init__(self, layers=None):
                from .utils_nn import revert_one_hot_nn, revert_softmax_nn
                self.revert_softmax = revert_softmax_nn
                self.revert_one_hot = revert_one_hot_nn
                self.layers = []

                if layers:  # self.layers = layers
                    for layer in layers:
                        self.layers.append(layer)
                        if layer.activation:
                            self.layers.append(layer.activation)
                self.finalized = False

                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=np.ComplexWarning)

            def add(self, layer):
                self.layers.append(layer)
                if layer.activation:
                    self.layers.append(layer.activation)

            def forward(self, inputs):
                for layer in self.layers:
                    inputs = layer.forward(inputs)
                return inputs

            def backward(self, grad):
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                return grad

            def finalize(self, loss, optimizer):
                '''Both of these have to be the classes for the loss and optimizations'''
                self.loss, self.optimizer = loss, optimizer
                self.finalized = True

            def num_parameters(self):
                return self.models[0].num_parameters()

            def _train_individual_model(self, model_num):
                '''here we train the individual model for the MapReduce algo'''
                model = self.models[model_num]
                data_chunk, label_chunk = self.x_train[model_num], self.y_train[model_num]
                model.full_batch_train(data_chunk, label_chunk, epochs=self.epochs, show_loop=False)
                self.models[model_num] = model  # update it to the train algorithm

            def give_best_parameters(self, x_test, y_test):
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")
                models = {}  # evaluation_score : class
                for model in self.models:
                    score = model.evaluate(x_test, y_test)
                    models[score] = model
                best_model = models[min(models)]
                return best_model.give_parameters()

            def pickle_best_parameters(self, x_test, y_test, FILE_NAME="NeuralNetwork_learnt_parameters"):
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")
                best_params = NeuralNetwork_MapReduce.give_best_parameters(self, x_test, y_test)

                import pickle

                with open(FILE_NAME + '.pickle', 'wb') as f:
                    pickle.dump(best_params, f, protocol=pickle.HIGHEST_PROTOCOL)

            def train(self, x_train, y_train, epochs=1, batch_size=32, num_cores=-1):
                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                # store these for the above method
                self.epochs = epochs
                self.batch_size = batch_size

                x_train, y_train = np.array(x_train), np.array(y_train)

                if len(y_train.shape) != 2: raise ValueError("y_train must be 2D.")
                if not num_cores: raise ValueError("num_cores cannot be 0!")
                num_cores_used = cpu_count() + 1 + num_cores if num_cores < 0 else num_cores
                self.num_cores_used = num_cores_used

                # split the x_train and y_train into the number of cores

                x_train = np.array_split(x_train, num_cores_used)
                y_train = np.array_split(y_train, num_cores_used)
                self.x_train, self.y_train = x_train, y_train

                # train using each dataset

                self.models = []

                for core in range(num_cores_used):
                    model = NeuralNetwork(self.layers)
                    model.finalize(self.loss, self.optimizer)
                    self.models.append(model)

                self.model_nums = range(len(self.models))
                Parallel()(delayed(NeuralNetwork_MapReduce._train_individual_model)(self, model_num) for
                           model_num in self.model_nums)

                self.num_cores_used = num_cores_used  # save this for the predict method

            def predict(self, x_test):
                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                x_test = np.array(x_test)
                predictions_across_models = np.array([model.predict(x_test) for model in self.models])
                if self.loss.type_regression:  # todo fix this
                    return np.sum(predictions_across_models, axis=0) / self.num_cores_used
                else:
                    return np.round_(np.sum(predictions_across_models, axis=0) / self.num_cores_used)

            def evaluate(self, x_test, y_test):
                if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

                x_test, y_test = np.array(x_test), np.array(y_test)
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

                y_pred = NeuralNetwork_MapReduce.predict(self, x_test)

                return self.loss.loss(y_test, y_pred)

            def regression_evaluate(self, x_test, y_test):
                x_test, y_test = np.array(x_test), np.array(y_test)
                # if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

                y_pred = NeuralNetwork_MapReduce.predict(self, x_test)

                return np.mean([r2_score(y_pred[:, col].flatten(), y_test[:, col].flatten()) for col in
                                range(y_pred.shape[1])])  # mean of each columns r^2

            def categorical_evaluate(self, x_test, y_test):
                x_test, y_test = np.array(x_test), np.array(y_test)
                if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

                y_pred = np.round_(NeuralNetwork_MapReduce.predict(self, x_test))

                return _perc_correct(self.revert_softmax(y_pred), self.revert_one_hot(y_test))

        import pickle

        with open(FILE_NAME + '.pickle', 'wb') as f:
            pickle.dump(best_params, f, protocol=pickle.HIGHEST_PROTOCOL)


    def train(self, x_train, y_train, epochs=1, batch_size=32, num_cores=-1) :
        if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

        #store these for the above method
        self.epochs = epochs
        self.batch_size = batch_size

        x_train, y_train = np.array(x_train), np.array(y_train)

        if len(y_train.shape) != 2: raise ValueError("y_train must be 2D.")
        if not num_cores: raise ValueError("num_cores cannot be 0!")
        num_cores_used = cpu_count() + 1 + num_cores if num_cores < 0 else num_cores
        self.num_cores_used = num_cores_used

        # split the x_train and y_train into the number of cores

        x_train = np.array_split(x_train, num_cores_used)
        y_train = np.array_split(y_train, num_cores_used)
        self.x_train, self.y_train = x_train, y_train

        # train using each dataset

        self.models = []

        for core in range(num_cores_used) :
            model = NeuralNetwork(self.layers)
            model.finalize(self.loss, self.optimizer)
            self.models.append(model)

        self.model_nums = range(len(self.models))
        Parallel()(delayed(NeuralNetwork_MapReduce._train_individual_model)(self, model_num) for
                   model_num in self.model_nums)


        self.num_cores_used = num_cores_used  # save this for the predict method

    def predict(self, x_test):
        if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

        x_test = np.array(x_test)
        predictions_across_models = np.array([model.predict(x_test) for model in self.models])
        if self.loss.type_regression : # todo fix this
            return np.sum(predictions_across_models, axis=0) / self.num_cores_used
        else :
            return np.round_(np.sum(predictions_across_models, axis=0) / self.num_cores_used)

    def evaluate(self, x_test, y_test):
        if not self.finalized: raise ValueError("This model isn't finalized. Call it through model.finalize().")

        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

        y_pred = NeuralNetwork_MapReduce.predict(self, x_test)

        return self.loss.loss(y_test, y_pred)

    def regression_evaluate(self, x_test, y_test):
        x_test, y_test = np.array(x_test), np.array(y_test)
        #if len(x_test.shape) != 2: raise ValueError("x_test must be 2D (even if only one sample.)")
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

        y_pred = NeuralNetwork_MapReduce.predict(self, x_test)

        return np.mean([r2_score(y_pred[:, col].flatten(), y_test[:, col].flatten()) for col in range(y_pred.shape[1])]) #mean of each columns r^2

    def categorical_evaluate(self, x_test, y_test) :
        x_test, y_test = np.array(x_test), np.array(y_test)
        if len(y_test.shape) != 2: raise ValueError("y_test must be 2D.")

        y_pred = np.round_(NeuralNetwork_MapReduce.predict(self, x_test))

        return _perc_correct(self.revert_softmax(y_pred), self.revert_one_hot(y_test))


