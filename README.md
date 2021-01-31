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

SeaLion can only support Python 3 (preferable 3.8), so please make sure you are on the newest version. 

## Documentation
All documentation is currently being put on a website. However useful it may be, I highly recommend you check the examples posted on GitHub here to see the usage of the APIs and how it works. 

## General Information
SeaLion was built by Anish Lakkapragada starting in Thanksgiving of 2020 and has continued onto early 2021. The library is meant for beginners to use when solving the standard libraries like iris, breast cancer, swiss roll, the moons dataset, MNIST, etc. The source code is not as much as most other ML libraries (only 4000 lines) and has much more relaxed security with the MIT license so feel free to read it or contribute. I am open to any suggestions. 

## What Tools We Used

## Machine Learning Algorithms

The machine learning algorithms of SeaLion are listed below. Please note that the stucture of the listing isn't meant to resemble that of SeaLion's APIs. Of course, 
new algorithms are being made right now. 

1. **Deep Neural Networks**
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
    * Transfer Learning
        - Save weights (in a pickle file) 
        - reload them and then enter them into the same neural network
        - this is so you don't have to start training from scratch
2. **Regression**
   - Linear Regression (Normal Equation, closed-form) 
   - Ridge Regression (L2 regularization, closed-form solution)
   - Lasso Regression (L1 regularization)
   - Elastic-Net Regression
   - Logistic Regression
   - Softmax Regression
   - Exponential Regression 
   - Polynomial Regression
3. **Dimensionality Reduction**
    - Principal Component Analysis (PCA)
    - t-distributed Stochastic Neighbor Embedding (tSNE)
4. **Unsupervised Clustering**
    - KMeans (w/ KMeans++)
    - DBSCAN
5. **Naive Bayes**
    - Multinomial Naive Bayes
    - Gaussian Naive Bayes
6. **Trees**
    - Decision Tree (with max_branches, min_samples regularization + CART training)
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
    
## Algorithms in progress
Some of the algorithms we are working on right now. 

1. **Batch Normalization**
2. **Binomial Naive Bayes**
3. **Gaussian Mixture Models**
4. **Barnes Hut t-SNE** (please, please contribute for this one)

## Contributing
If you feel you can do something better than how it is right now in SeaLion, please do! Believe me, you will find great joy in simplifying my code (probably using numpy) and speeding it up. The major problem right now is speed, some algorithms like PCA can handle 10000+ data points, whereas tSNE is unscalable with O(n^2) time complexity. We have solved this problem with Cython + parallel processing (thanks joblib), so algorithms (aside from neural networks) are working well with <1000 points. Getting to the next level will need some help. 

Most of the modules I use are numpy, pandas, joblib, and tqdm. I prefer using less dependencies in the code, so please keep it down to a minimum. 

Other than that, thanks for contributing!

## Acknowledgements
Plenty of articles and people helped me a long way. Some of the tougher questions I dealt with were Automatic Differentiation in neural networks, in which this [tutorial](https://www.youtube.com/watch?v=o64FV-ez6Gw) helped me. I also got some help on the O(n^2) time complexity problem of the denominator of t-SNE from this [article](https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/) and understood the mathematical derivation for the gradients (original paper didn't go over it) from [here](http://pages.di.unipi.it/errica/assets/files/sne_tsne.pdf). Lastly special thanks to Evan M. Kim and Peter Washington for helping make the normal equation and cauchy distribution in tSNE make sense. Also thanks to @KentoNishi for helping me understand open-source. 

## Feedback, comments, or questions
If you have any feedback or something you would like to tell me, please do not hesitate to share! Feel free to comment here on github or reach out to me through
anish.lakkapragada@gmail.com! 


©Anish Lakkapragada 2021
