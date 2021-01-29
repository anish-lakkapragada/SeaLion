"""
Example of neural networks using the MNIST dataset, which is composed of 70000 28 * 28 grayscale images from 1 - 10. Here we will be
teaching a neural network to learn how to recognize digits.
"""
import sealion as sl  # first import this, under sl alias
from sealion.neural_networks.optimizers import SGD, Adam  # we'll use these 2, but feel free to try some more
from sealion.neural_networks.loss import CrossEntropy  # this is the loss function we'll use (classification problem)
import mnist  # library to load in images
from sealion.utils import one_hot  # one_hot function for our data

# Step 1 : Load and preprocess data!

X_train = mnist.train_images() / 255.0  # load 60k images in the training set, and divide by 255.0 to normalize from
# 0 -1. This helps with faster convergence.

y_train = one_hot(mnist.train_labels().flatten(),
                  depth=10)  # we one_hot here for the neural network labels. Depth 10 as there are 10 classes : 0,
# 1, 2 ... 9

X_test = mnist.test_images() / 255.0  # same thing here for the test data
y_test = one_hot(mnist.test_labels().flatten(), depth=10)  # and the test labels

# of course there's a lot we could do here, but lets' move on to Step 2

# Step 2 : Build the neural network

# the first part is to just build the architecture like the following :

model = sl.neural_networks.models.NeuralNetwork()
model.add(sl.neural_networks.layers.Flatten())
model.add(sl.neural_networks.layers.Dense(784, 64, activation=sl.neural_networks.layers.LeakyReLU(leak=0.2)))
model.add(sl.neural_networks.layers.Dense(64, 32, activation=sl.neural_networks.layers.LeakyReLU(leak=0.2)))
model.add(sl.neural_networks.layers.Dense(32, 10, activation=sl.neural_networks.layers.Softmax()))

# or you could build it like such :
model = sl.neural_networks.models.NeuralNetwork([
    sl.neural_networks.layers.Flatten(),
    sl.neural_networks.layers.Dense(784, 64, activation=sl.neural_networks.layers.LeakyReLU(leak=0.2)),
    sl.neural_networks.layers.Dense(64, 32, activation=sl.neural_networks.layers.LeakyReLU(leak=0.2)),
    sl.neural_networks.layers.Dense(32, 10, activation=sl.neural_networks.layers.Softmax())
])

# just a few notes. Here we created first a flatten() layer because we are taking the 28 * 28 input to just a 784 (
# *1) vector. So our input size for the dense layer is 784. everything after that with the 64 outputs in the first
# hidden layer, then 32 is arbitrary, and based on hyperparameter tuning. we used the leaky relu, instead of the
# typical relu, to avoid the dying neurons problems and Softmax() activation in the end to turn this into
# probabilities.

# Lastly we'll need to finalize the model, like such :

model.finalize(loss=CrossEntropy(), optimizer=SGD(lr=0.3, momentum=0.2, nesterov=True))  # "finalize" just
# means set the loss function and optimizer

# here we used CrossEntropy (must be used with softmax or vice versa) and the SGD optimizer. It has a learning rate
# of 0.3, momentum of 0.2, and utilizes nesterov accelerated gradient.

'''To see how complex our model is we can do : '''
num_parameters = model.num_parameters()
print("Number of parameters for the 1st : ", num_parameters)

# Step 3 : Training!

model.train(X_train, y_train, epochs=50)  # we can then train this model for 50 epochs....

# Step 4 : Evaluate!

'''Here we will evaluate our model and see how well it did.'''
print("Loss : ", model.evaluate(X_test, y_test))  # this is just the loss
print("Regression accuracy : ", model.regression_evaluate(X_test, y_test))  # regression way of calculating (horrible here,
# because this is classification)
print("Validation accuracy : ", model.categorical_evaluate(X_test, y_test))  # classification accuracy, this will
# percent the percent that was classified correctly


# It looks like we can do better. Let's build a new neural network, with a different architecture.

model = sl.neural_networks.models.NeuralNetwork()
model.add(sl.neural_networks.layers.Flatten())
model.add(sl.neural_networks.layers.Dense(784 , 128, activation = sl.neural_networks.layers.ELU()))
model.add(sl.neural_networks.layers.Dense(128, 64, activation = sl.neural_networks.layers.ELU()))
model.add(sl.neural_networks.layers.Dense(64, 10, activation = sl.neural_networks.layers.Softmax()))

model.finalize(loss = sl.neural_networks.loss.CrossEntropy(), optimizer = sl.neural_networks.optimizers.Adam(lr = 0.01)) #use adam

model.train(X_train, y_train, epochs=20) # train this model

print("Loss : ", model.evaluate(X_test, y_test))
print("Regression accuracy : ", model.regression_evaluate(X_test, y_test))
print("Validation accuracy : ", model.categorical_evaluate(X_test, y_test))

# It looks like we like this model.

'''
Often times training a neural network will take some time. Who wants to go through the process from the start? 
Instead, we can save our weights and biases so we can reuse them and plug them back into a neural network as we please. 
'''

parameters = model.give_parameters() # get the parameters (weights + biases)

# we can also just directly store this into a pickle file
FILE_NAME = "MNIST_weights"
model.pickle_params(FILE_NAME) # now the parameters will be stored in a file known as MNIST_weights.pickle

"""
Let's say we want to train the model again, this time with SGD but with a slower learning rate. But we don't want to 
start training from scratch (with the random parameter initialization), but rather with some pre-trained weights. Remember
that neural network we made a month ago (just pretend it was), why don't we just load its weights into our new model and train it
more? This is known as transfer learning, a key concept in machine learning today. 
"""

# first we have to create the exact same model architecture
model = sl.neural_networks.models.NeuralNetwork()
model.add(sl.neural_networks.layers.Flatten())
model.add(sl.neural_networks.layers.Dense(784 , 128, activation = sl.neural_networks.layers.ELU()))
model.add(sl.neural_networks.layers.Dense(128, 64, activation = sl.neural_networks.layers.ELU()))
model.add(sl.neural_networks.layers.Dense(64, 10, activation = sl.neural_networks.layers.Softmax()))

# but we can use a different optimizer

model.finalize(loss = sl.neural_networks.loss.CrossEntropy(), optimizer = sl.neural_networks.optimizers.SGD(lr = 0.01)) #use adam

# let's load in those weights using pickle
import pickle
with open('MNIST_weights.pickle', 'rb') as file :
    parameters = pickle.load(file) # load back the parameters in a variable

# now we can enter in the parameters into the weights of our current architecture (so we don't have to train from scratch)
model.enter_parameters(parameters)

# now let's train it some more!

model.train(X_train, y_train, epochs= 50) # train this model for 50 epochs

print("Loss : ", model.evaluate(X_test, y_test))
print("Regression accuracy : ", model.regression_evaluate(X_test, y_test))
print("Validation accuracy : ", model.categorical_evaluate(X_test, y_test))

# and if you so desire, we can save the weights again.
file_name = "MNIST_weights2"
model.pickle_params(file_name)

# this will generally do better and training will be much faster, so if you aren't seeing
# any noticeable results that's because this is MNIST. Either way we hope
# you enjoyed this tutorial and neural networks make a bit more sense
# with SeaLion.
