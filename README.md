# SeaLion

SeaLion was designed to teach programmers the popular machine learning concepts of today in a way that gives both intution and ways of application. 
We do this through documentation that explains the models, how they work, and how to use them with examples on familiar datasets like iris, breast cancer, MNIST, etc. 
Through this library we hope newer students will learn common algorithms and those more experienced will appreciate the extra morsels of functionality. 

## Installation
The package is available on PyPi. 
Install like such : 
```shell
pip install sealion
```

## Documentation
All documentation is available with the pydoc module. However useful they may be, I highly recommend you check the examples posted on GitHub here and seeing how the classes work. 

## Machine Learning Algorithms

The machine learning algorithms of SeaLion are listed below. Please note that the stucture of the listing isn't meant to resemble that of SeaLion's. Of course, 
new algorithms are being made right now. 

1. **Neural networks**
    * Optimizers
        - Gradient Descent (and mini-batch gradient descent)
        - Momentum Optimization w/ Nesterov Accelerated Gradient
        - Stochastic gradient descent (w/ momentum + nesterov)
        - AdaGrad 
        - RMSprop
        - Adam
        - Nadam
    * Layers
        - Flatten (turn 2D+ data to 2D matrices)
        - Dense (fully-connected layers) 
    * Regularization
        - Dropout
    * Activations
        - ReLU
        - Tanh
        - Sigmoid
        - Softmax
        - Leaky ReLU
        - ELU
        - SELU
        - Swish
    * Loss Functions
        - MSE (for regression)
        - CrossEntropy (for classification)
2. **Regression**
   - Linear Regression (Normal Equation, closed-form) 
   - Ridge Regression (closed-form solution)
   - Lasso Regression
   - Elastic-Net Regression
   - Logistic Regression
   - Softmax Regression
   - Exponential Regression 
   - Polynomial Regression
3. **Dimensionality Reduction**
    - Principal Component Analysis (PCA)
    - t-distributed Stochastic Neighbor Embedding (tSNE)
4. **Unsupervised Clustering**
    - KMeans
    - DBSCAN
5. **Naive Bayes**
    - Multinomial Naive Bayes
    - Gaussian Naive Bayes
6. **Trees**
    - Decision Tree (with max_branches, min_samples regularization)
7. **Ensemble Learning**
    - Random Forests
    - Ensemble/Voting Classifier
8. **Nearest Neighbors**
    - k-nearest neighbors
9. **Utils**
    - one_hot encoder function (one_hot())
    - plot confusion matrix function (confusion_matrix())
    - revert one hot encoding to 1D Array (revert_one_hot())
    - revert softmax predictions to 1D Array (revert_softmax())


 
    









