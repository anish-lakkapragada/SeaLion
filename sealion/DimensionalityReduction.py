"""
@author : Anish Lakkapragada
@date : 1/9/2021

Contains PCA and tSNE, two great dimensionality reduction algorithms. PCA can deal with around 1000 - 10000 data points
(obviously depending on the number of dimensions.) tSNE is better for visualizations but struggles past five hundred points,
so PCA is usually better for big data and you can tSNE is for the toy datasets.
"""

import numpy as np

class PCA() :
    """
    PCA is an algorithm to project the data to however many dimensions you want (please not 0 - LOL!). It is the most
    well known dimensionality reduction algorithm. Can work with data (10000+) very fast. We even have an inverse_transform()
    method to do reverse PCA and visualize_variance() plot method for you to find the best dimensions for your data.

    ----
    Methods
    __init__(new_ndims) :
        ->> new_ndims : just give new_ndims, the number of new dimensions you want your data to have
    transform(X) :
        ->> give your data in X, which must be a 2D numpy array or python list. 2D meaning [[]] not [] (1D.)
        ->> we will return each data point to X in new_ndims (passed in __init__) space
        ->> all code (5 lines) for this method comes from Hands On Machine Learning (Edition 2) by Aurelien Geron
    inverse_transform(X) :
        ->> reverts each point in X to its original size
        ->> keep in mind that it is very hard to get the exact same X as you originally had as PCA naturally loses
        some of the variance. However, the structure and the shape will be preserved.
    visualize_variance(X, representation_dims) :
        ->> X is the data you want to be transformed in new_ndims space
        ->> representation_dims is a list of all the dimensions you want to try your data in. For example if you give
        [3, 4, 5, 6] your data will be tried in 3, 4, 5, and 6 dimensions. The data dimension (number) will be plotted on the
        x-axis and the variance (of that projection to 3, 4, 5, and 6 dimensions here) will be on the y-axis.
        This is to help you find which dimension you should turn your data into (probably the one which has a good
        variance and is the lowest dimension.)

    """
    def __init__(self, new_ndims):
        """
        :param new_ndims: The number of dimensions you want to reduce your data to.
        """
        self.new_ndims = new_ndims
    def transform(self, X):
        """
        :param X: the data you want transformed to new_ndims space (specified in __init___). Must be 2D (check above for
        help.)
        :return: all data points in new_ndims space
        """
        X = np.array(X)
        X_centered = X - X.mean(axis = 0)
        U, s, Vt = np.linalg.svd(X_centered)
        WD = Vt.T[:, :self.new_ndims]
        X_pca = X_centered.dot(WD)
        self.WD = WD
        return X_pca

    def inverse_transform(self, X):
        """
        :param X:  data that is in new_ndims space
        :return: all data points in the original dimensional space.
        """
        return X.dot(self.WD.T)

    def visualize_variance(self, X, representation_dims):
        """
        :param X: the data you want transformed (not transformed into new_ndims space)
        :param representation_dims: a range of the number of dimensions you want your data transformed in.
        If you want to try it out in 3 - 5 dimensions you can do a list of [3, 4, 5] for example or for 1 - 100 dimensions
        you can do range(1, 100 + 1)
        :return: show the image of the variance of X when it is reduced to each dimension in representation_dims.
        This is to help you find out which dimension to keep your data in (probably the lowest dimension with a
        desired amount of variance.)
        """
        X = np.array(X)
        self.original_variance = np.var(X)
        variances = []
        for dim in representation_dims :
            new_pca = PCA(dim)
            new_data = new_pca.transform(X)
            variances.append(np.sum(np.var(new_data, axis = 0)))
        #visualize!
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot([dim for dim in representation_dims], [variance for variance in variances], color = "blue", label = "Dimension Variance")
        plt.xlabel("# dimensions")
        plt.ylabel("Variance")
        plt.xticks(np.array([dim for dim in representation_dims], np.int))
        plt.yticks([])
        plt.title("Variance of PCA with # dimensions")
        plt.legend()

        plt.show()

class tSNE() :
    """
    Not as famous as PCA for Dimensionality Reduction, but very famous for Data Visualization of data in more than 3
    dimensions. The hardest part of it is the need for hyperparameter tuning, especially with the perplexity factor.
    Try as many values you get until you get visualizations you are satisfied with. The "t" stands for the
    t-distribution utilized in tSNE. A con to tSNE is its runtime - past 1000 points, tSNE can be very slow. Even with
    faster optimizations, the time complexity is O(nlogn).

    Another thing to note is the use of running it multiple times until you get the best visualization(s). You may want
    to consider running this multiple times in parallel on different cores using something like multiprocessing or
    multiprocess in python.

    ----
    Methods
    __init__(new_ndims, perplexity = 10, learning_rate = 200, momentum = 0.9, max_iters = 1000) :
        ->> new_ndims is the number of dimensions you want your data to go down to
        ->> perplexity is the parameter that sort of decides how many points your data has. It is affected by the scale
        of your data, so look at that as well. Can range from decimals to a hundred.
        ->> learning rate : this algorithm uses gradient descent so this parameter is really how fast you want it to learn.
        Don't be suprised by the default 200, the gradients are really small for this algo.
        ->> momentum : momentum optimization is used in gradient descent for faster convergence of tSNE in its visualization.
        The momentum parameter is really just a controlling factor on how fast to go.
        ->> max_iters : the number of iterations (or epochs) allowed for this algorithm.
    transform(X) :
        Phew! that was a lot of parameters for the __init__! Luckily it's just one here.
        ->> X is the data you want transformed. It has to be 2D ([[]]) not 1D ([]) and can be a numpy array or python list.
        Runtimes could vary from a few minutes to a few hours, but it will be worth it!
    """
    def __init__(self, new_ndims, perplexity = 10, learning_rate = 200, momentum = 0.9, max_iters = 1000) :
        """
        :param new_ndims: number of dimensions you want your data to go down to
        :param perplexity: how many points do you think are near the typical point in your data. Keep the scale in mind
        when setting this.
        :param learning_rate: learning rate used in gradient descent.
        :param momentum: momentum parameter for the gradient descent used.
        :param max_iters: max number of iterations we will allow gradient descent to run on tSNE
        """
        self.inner_cython_tsne = cython_tsne.cy_tSNE(new_ndims = new_ndims, perplexity = perplexity, learning_rate = learning_rate,
        momentum = momentum, max_iters = max_iters)
    def transform(self, X) :
        """
        :param X: the data you want transformed, must be 2D.
        :return: every data point in X transformed to new_ndims (set in __init__) space.
        """
        return self.inner_cython_tsne.transform(X)

