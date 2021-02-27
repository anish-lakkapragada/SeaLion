<p align="center">
    <img src="https://github.com/anish-lakkapragada/SeaLion/blob/main/logo.png?raw=true" width = 300 height = 300 >
</p>

# SeaLion

![python](https://img.shields.io/pypi/pyversions/sealion?color=blueviolet&style=plastic)
![License](https://img.shields.io/pypi/l/sealion?color=informational&style=plastic)
![total lines](https://img.shields.io/tokei/lines/github/anish-lakkapragada/SeaLion?color=brightgreen)
![issues](https://img.shields.io/github/issues/anish-lakkapragada/SeaLion?color=yellow&style=plastic)
![pypi](https://img.shields.io/pypi/v/sealion?color=red&style=plastic)
![repo size](https://img.shields.io/github/repo-size/anish-lakkapragada/SeaLion?color=important)
![Deploy to PyPI](https://github.com/anish-lakkapragada/SeaLion/workflows/Deploy%20to%20PyPI/badge.svg)

SeaLion is designed to teach today's aspiring ml-engineers the popular
machine learning concepts of today in a way that gives both intuition and
ways of application. We do this through concise algorithms that do the job 
in the least jargon possible and examples to guide you through every step 
of the way. 

## Quick Demo

<p align="center">
    <img src="https://raw.githubusercontent.com/anish-lakkapragada/SeaLion/main/sealion_demo.gif" width = 580 height = 326>
    <br />
    <i>SeaLion in Action</i>
</p>

## General Usage

For most classifiers you can just do (we'll use Logistic Regression as an
example here) :

```python
from sealion.regression import LogisticRegression
log_reg = LogisticRegression()
```

to initialize, and then to train :

``` python
log_reg.fit(X_train, y_train) 
```

and for testing :

```python
y_pred = log_reg.predict(X_test) 
evaluation = log_reg.evaluate(X_test, y_test) 
```

For the unsupervised clustering algorithms you may do :

```python
from sealion.unsupervised_clustering import KMeans
kmeans = KMeans(k = 3)
```

and then to fit and predict :

```python
predictions = kmeans.fit_predict(X) 
```

Neural networks are a bit more complicated, so you may want to check an example
[here.](https://github.com/anish-lakkapragada/SeaLion/blob/main/examples/deep_learning_example.ipynb)

The syntax of the APIs was designed to be easy to use and familiar to most other ML libraries. This is to make sure both beginners and experts in the field
can comfortably use SeaLion. Of course, none of the source code uses other ML frameworks. 

## Testimonials and Reddit Posts

"Super Expansive Python ML Library"
   -   [@Peter Washington](https://twitter.com/peter\_washing/status/1356766327541616644), Stanford PHD candidate in Bio-Engineering

[Analytics Vidhya calls SeaLion's algorithms **beginner-friendly**, **efficient**, and **concise**.](https://www.analyticsvidhya.com/blog/2021/02/6-open-source-data-science-projects-that-provide-an-edge-to-your-portfolio/)

r/Python : [r/Python Post](https://www.reddit.com/r/Python/comments/lf59bw/machine_learning_library_by_14year_old_sealion/)

r/learnmachinelearning : [r/learningmachinelearning Post](https://www.reddit.com/r/learnmachinelearning/comments/lfv72l/a_set_of_jupyter_notebooks_to_help_you_understand/)

## Installation

The package is available on PyPI. Install like such :

``` shell
pip install sealion
```

SeaLion can only support Python 3, so please make sure you are on the
newest version.


## General Information

SeaLion was built by Anish Lakkapragada, a freshman in high school, starting in Thanksgiving of 2020
and has continued onto early 2021. The library is meant for beginners to
use when solving the standard libraries like iris, breast cancer, swiss
roll, the moons dataset, MNIST, etc. The source code is not as much as
most other ML libraries (only 4000 lines) so it's pretty easy to contribute to. He
hopes to spread machine learning to other high schoolers through this
library.


## Documentation

All documentation is currently being put on a website. However useful it
may be, I highly recommend you check the examples posted on GitHub here
to see the usage of the APIs and how it works.

### Updates for v4.1 and up!
First things first - thank you for all of the support. The two reddit posts did much better than I expected (1.6k upvotes, about 200 comments) and I got a lot
of feedback and advice. Thank you to anyone who participated in r/Python or r/learnmachinelearning.

SeaLion has also taken off with the posts. We currently have had 3 issues (1 closed) and have reached 195 stars and 20 forks. I wasn't expecting this and I am grateful for everyone who has shown their appreciation for this library. 

Also some issues have popped up. Most of them can be easily solved by just deleting sealion manually (going into the folder where the source is and just deleting it - not pip uninstall) and then reinstalling the usual way, but feel free to put an issue up anytime. 

In versions 4.1+ we are hoping to polish the library more. Currently 4.1 comes with Bernoulli Naive Bayes and we also have added precision, recall, and the f1 metric in the utils module. We are hoping to include Gaussian Mixture Models and Batch Normalization in the future. Code examples for these new algorithms will be created within a day or two after release. Thank you! 

### Updates for v3.0.0!

SeaLion v3.0 and up has had a lot of major milestones.

The first thing is that all the code examples (in jupyter notebooks) for
basically all of the modules in sealion are put into the examples
directory. Most of them go over using actual datasets like iris, breast
cancer, moons, blobs, MNIST, etc. These were all built using v3.0.8
-hopefully that clears up any confusion. I hope you enjoy them.

Perhaps the biggest change in v3.0 is how we have changed the Cython
compilation. A quick primer on Cython if you are unfamiliar - you take
your python code (in .py files), change it and add some return types and
type declarations, put that in a .pyx file, and compile it to a .so
file. The .so file is then imported in the python module which you use.

The main bug fixed was that the .so file is actually specific to the
architecture of the user. I use macOS and compiled all my files in .so,
so prior v3.0 I would just give those .so files to anybody else. However
other architectures and OSs like Ubuntu would not be able to recognize
those files. Instead what we do know is just store the .pyx files
(universal for all computers) in the source code, and the first time you
import sealion all of those .pyx files will get compiled into .so files
(so they will work for whatever you are using.) This means the first
import will take about 40 seconds, but after that it will be as quick as
any other import.

## Machine Learning Algorithms

The machine learning algorithms of SeaLion are listed below. Please note
that the stucture of the listing isn't meant to resemble that of
SeaLion's APIs. Of course, new algorithms are being made right now.

1.  **Deep Neural Networks**
    -   Optimizers
        -   Gradient Descent (and mini-batch gradient descent)
        -   Momentum Optimization w/ Nesterov Accelerated Gradient
        -   Stochastic gradient descent (w/ momentum + nesterov)
        -   AdaGrad
        -   RMSprop
        -   Adam
        -   Nadam
    -   Layers
        -   Flatten (turn 2D+ data to 2D matrices)
        -   Dense (fully-connected layers)
    -   Regularization
        -   Dropout
    -   Activations
        -   ReLU
        -   Tanh
        -   Sigmoid
        -   Softmax
        -   Leaky ReLU
        -   ELU
        -   SELU
        -   Swish
    -   Loss Functions
        -   MSE (for regression)
        -   CrossEntropy (for classification)
    -   Transfer Learning
        -   Save weights (in a pickle file)
        -   reload them and then enter them into the same neural network
        -   this is so you don't have to start training from scratch

2.  **Regression**

    -   Linear Regression (Normal Equation, closed-form)
    -   Ridge Regression (L2 regularization, closed-form solution)
    -   Lasso Regression (L1 regularization)
    -   Elastic-Net Regression
    -   Logistic Regression
    -   Softmax Regression
    -   Exponential Regression
    -   Polynomial Regression

3.  **Dimensionality Reduction**
    -   Principal Component Analysis (PCA)
    -   t-distributed Stochastic Neighbor Embedding (tSNE)

4.  **Unsupervised Clustering**
    -   KMeans (w/ KMeans++)
    -   DBSCAN

5.  **Naive Bayes**
    -   Multinomial Naive Bayes
    -   Gaussian Naive Bayes
    -   Bernoulli Naive Bayes

6.  **Trees**
    -   Decision Tree (with max\_branches, min\_samples regularization +
        CART training)

7.  **Ensemble Learning**
    -   Random Forests
    -   Ensemble/Voting Classifier

8.  **Nearest Neighbors**
    -   k-nearest neighbors

9.  **Utils**
    -   one\_hot encoder function (one\_hot())
    -   plot confusion matrix function (confusion\_matrix())
    -   revert one hot encoding to 1D Array (revert\_one\_hot())
    -   revert softmax predictions to 1D Array (revert\_softmax())

## Algorithms in progress

Some of the algorithms we are working on right now.

1.  **Batch Normalization**
3.  **Gaussian Mixture Models**
4.  **Barnes Hut t-SNE** (please, please contribute for this one)

## Contributing

First, install the required libraries:
```bash
pip install -r requirements.txt
```

If you feel you can do something better than how it is right now in
SeaLion, please do! Believe me, you will find great joy in simplifying
my code (probably using numpy) and speeding it up. The major problem
right now is speed, some algorithms like PCA can handle 10000+ data
points, whereas tSNE is unscalable with O(n\^2) time complexity. We have
solved this problem with Cython + parallel processing (thanks joblib),
so algorithms (aside from neural networks) are working well with \<1000
points. Getting to the next level will need some help.

Most of the modules I use are numpy, pandas, joblib, and tqdm. I prefer
using less dependencies in the code, so please keep it down to a
minimum.

Other than that, thanks for contributing!

## Acknowledgements

Plenty of articles and people helped me a long way. Some of the tougher
questions I dealt with were Automatic Differentiation in neural
networks, in which this
[tutorial](https://www.youtube.com/watch?v=o64FV-ez6Gw) helped me. I
also got some help on the O(n\^2) time complexity problem of the
denominator of t-SNE from this
[article](https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/) and
understood the mathematical derivation for the gradients (original paper
didn't go over it) from
[here](http://pages.di.unipi.it/errica/assets/files/sne_tsne.pdf). Also
I used the PCA method from handsonml so thanks for that too Aurélien
Géron. Lastly special thanks to Evan M. Kim and Peter Washington for
helping make the normal equation and cauchy distribution in tSNE make
sense. Also thanks to [@Kento Nishi](http://github.com/KentoNishi) for
helping me understand open-source.

## Feedback, comments, or questions

If you have any feedback or something you would like to tell me, please
do not hesitate to share! Feel free to comment here on github or reach
out to me through <anish.lakkapragada@gmail.com>!

©Anish Lakkapragada 2021
