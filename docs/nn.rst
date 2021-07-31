Neural Networks
===============

Models
--------

.. autoclass:: sealion.neural_networks.models.NeuralNetwork
    :members: __init__, num_parameters, enter_parameters, give_parameters, pickle_params, add,
        forward, backward, finalize, train, full_batch_train, mini_batch_train, predict, evaluate,
        regression_evaluate, categorical_evaluate

Layers
------

.. autoclass:: sealion.neural_networks.layers.Layer
    :members: forward, backward

.. autoclass:: sealion.neural_networks.layers.Flatten

.. autoclass:: sealion.neural_networks.layers.Dropout

.. autoclass:: sealion.neural_networks.layers.Dense

.. autoclass:: sealion.neural_networks.layers.BatchNormalization

Activations
-----------

.. autoclass:: sealion.neural_networks.layers.Tanh

.. autoclass:: sealion.neural_networks.layers.Sigmoid

.. autoclass:: sealion.neural_networks.layers.Swish

.. autoclass:: sealion.neural_networks.layers.ReLU

.. autoclass:: sealion.neural_networks.layers.LeakyReLU

.. autoclass:: sealion.neural_networks.layers.ELU

.. autoclass:: sealion.neural_networks.layers.SELU

.. autoclass:: sealion.neural_networks.layers.PReLU

.. autoclass:: sealion.neural_networks.layers.Softmax

Losses
------

.. autoclass:: sealion.neural_networks.loss.Loss
    :members: loss, grad

.. autoclass:: sealion.neural_networks.loss.MSE

.. autoclass:: sealion.neural_networks.loss.CrossEntropy

Optimizers
----------

.. autoclass:: sealion.neural_networks.optimizers.Optimizer
    :members: setup, update

.. autoclass:: sealion.neural_networks.optimizers.GD

.. autoclass:: sealion.neural_networks.optimizers.Momentum

.. autoclass:: sealion.neural_networks.optimizers.SGD

.. autoclass:: sealion.neural_networks.optimizers.AdaGrad

.. autoclass:: sealion.neural_networks.optimizers.RMSProp

.. autoclass:: sealion.neural_networks.optimizers.Adam

.. autoclass:: sealion.neural_networks.optimizers.Nadam

.. autoclass:: sealion.neural_networks.optimizers.AdaBelief
