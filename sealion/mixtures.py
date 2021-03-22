"""
@author : Anish Lakkapragada
@date : 3 - 4 - 2021

Gaussian Mixtures are one of the best models today in the field of anomaly detection and unsupervised clustering.
But it's not just because of that that this is one of the best modules SeaLion has to offer.
"""

import numpy as np

class GaussianMixture() :
    """
    Gaussian Mixture Models are really just a fancier extension of KMeans. Make sure you know KMeans before you read this.
    If you are unfamiliar with KMeans, feel free to look at the examples on GitHub for unsupervised clustering or
    look at the unsupervised clustering documentation (which contains the KMeans docs.) You may also want to know
    what a gaussian (normal) distribution is.

    In KMeans, you make the assumption that your data is in spherical clusters, all of which are the same shape. What
    if your data is instead in such circles, but also maybe ovals that are thin, tall, etc. Basically what if your
    clusters are spherical-esque but of different shapes and sizes.

    This difference can be measured by the standard deviation, or variance, of each of the clusters. You can think of each
    cluster as a gaussian distribution, each with a different mean (similiar to the centroids in KMeans) and standard
    deviation (which effects how skinny/wide it is.) This model is called a mixture, because it essentially is just a
    mixture of gaussian functions. Some of the clusters maybe bigger than others (aside from shape), and this is
    also taken into account with a mixture weight - a coefficient that is multiplied to each gaussian distribution.

    With this gaussian distribution, you can then take any points and assign them to the cluster they have the
    highest change of being in. Because this is probabilities, you can say that a given data point has a 70% chance
    of being in this class, 20% this class, 5% this class, and 5% this other class - which is something you
    cannot do with KMeans. The parameters of the gaussian distribution are learnt using an algorithm known
    as Expectation Maximization, which you may want to learn about (super cool)!

    You can also do anomaly detection with Gaussian Mixture Models! You do this by looking at the probability each data
    point belonged to the cluster it had the highest chance of being in. We can call this list of a probability
    that a data point belonged to the cluster it was chosen to for all data points "confidences". If a given data points
    confidence is in the lowest _blank_ (you set this) percent of confidences, it's marked as an anomaly. This _blank_ is
    from 1 - 100, and is essentially what percent of your data you think are outliers.

    To find the best value for n_clusters, we also have a visualize_elbow_curve method to do that for you. Check the docs
    below if you're interested!

    That's a lot, so let's take a look at its methods!

    NOTE : X SHOULD BE 2D FOR ALL METHODS.

    ____
    Methods

    __init__(n_clusters = 5, retries = 3, max_iters = 200, kmeans_init = True)
        ->> n_clusters is the number of clusters you think are in the data (think of it as the k in KMeans)
        ->> retries is the number of times you want this model to retry. Just like KMeans may need to be tried multiple times,
            we run gaussian mixture models a few times and pick the best solution.
        ->> each retry works an iterative algorithm called Expectation-Maximization (EM). 200 is the maximum amount of iterations
        set by default for the maximum amount of iterations a single retry of EM can take.
        ->> the means of Gaussian Mixture models can be initialized using k-Means (if your data is more than 1D). You can
        set this parameter False if you don't want that.

    fit(X) :
        ->> given X, the model will find the means, standard deviation (or covariance matrices for multivariate data),
        and mixture weights - which are all of the parameters of GMMs
        ->> it will run through the amount of retries specified in the initialization
    predict(X) :
        ->> returns the predicted class for each data point in X
    soft_predict(X) :
        ->> returns the probability for each data point in X that it belonged to each of the n_clusters (defined in the init) clusters
    confidence_samples(X) :
        ->> returns for each data point its probability it belongs to the cluster it has the highest chance of being in.
        This is essentially the confidence of a data point belonging in the data.
    aic() :
        ->> AIC metric that measures how well the data is fit by the Gaussian Mixture
        ->> lower is better
    bic() :
        ->> BIC metric that measures how well the data is fit by the Gaussian Mixture
        ->> lower is better
    anomaly_detect(X, threshold) :
        ->> given this X data, return whether each sample is an anomaly based on whether its "confidence" to the cluster
        it joined is in the bottom threshold percent of all confidences. If its "confidence" is less than the threshold, it's
        marked as an outlier (True) else False. Data which has a low confidence is less likely to truly belong in the data,
        and more likely to be an outlier.
    visualize_clustering(color_dict) :
        ->> only work if your data is in 2 dimensions or more
        ->> visualizes the data and colors each of the data points according to which cluster it joined
        ->> if our color dict was : {0 : "green", 1 : "red", 2 : "orange"}
        ->> here a data point classified to the 0th cluster will be green, a data point to the 1st cluster will be red and
        a data point to the 2nd cluster will be orange.
    visualize_elbow_curve(min_n_clusters = 2, max_n_clusters = 5) :
        ->> to choose the best value of n_clusters, you want to find the elbow on this curve
        ->> this method will try out a bunch of values from n_clusters and then plot its BIC and AIC metrics for each
        value
        ->> the cluster values tried will be from min_n_clusters to max_n_clusters
    """
    def __init__(self, n_clusters = 5, retries = 3, max_iters = 200, kmeans_init = True):
        """
        :param n_clusters : number of clusters assumed in the data
        :param retries : number of times the algorithm will be tried, and then the best solution picked
        :max_iters : maximum number of iterations that the algorithm will be ran for each retry
        :kmeans_init : whether the centroids found by using KMeans should be used to initialize the means in this Gaussian Mixture.
        This will only work if your data is in more than one dimension. If you are using a lot of retries, this may
        not be a good option as it may lead to the same solution over and over again.
        """
        from .cython_mixtures import cy_GaussianMixture
        self.cython_gmm_class = cy_GaussianMixture
        self.cython_gaussian_mixture = cy_GaussianMixture(n_clusters, retries, max_iters, kmeans_init)
        self.k = n_clusters
        self.retries = retries
        self.max_iters = max_iters
        self.kmeans_init = kmeans_init
    def fit(self, X):
        """
        :param X : training data
        this method finds the parameters needed for Gaussian Mixtures to function
        """
        self.X = X
        self.cython_gaussian_mixture.fit(X)
    def predict(self, X):
        """
        :param X : prediction data
        this method returns the cluster each data point in X belongs to
        """
        return self.cython_gaussian_mixture.predict(X)
    def soft_predict(self, X):
        """
        :param X : prediction data
        this method returns the probability each data point belonged to each cluster. This is stored in a matrix
        with the length of X rows and the width of the amount of clusters.
        """
        return self.cython_gaussian_mixture.soft_predict(X)

    def confidence_samples(self, X):
        """
        :param X : prediction data

        This method essentially gives the highest probability in the rows of probabilities in the matrix that is the output
        of the soft_predict() method.

        Translation :

        It's telling you the probability each data point had in the cluster it had
        the largest probability in (and ultimately got assigned to), which is really telling you how confident it is
        that a given data point belongs to the data.
        """
        return self.cython_gaussian_mixture.confidence_samples(X)

    def aic(self):
        """
        Returns the AIC metric (lower is better.)
        """
        return self.cython_gaussian_mixture.aic()

    def bic(self):
        """
        Returns the BIC metric (lower is better.)
        """
        return self.cython_gaussian_mixture.bic()
    def anomaly_detect(self, X, threshold):
        """
        :param X : prediction data
        :param threshold : what percent of the data you believe is an outlier
        Returns whether each data point in X is an anomaly based on whether its confidence in the cluster it was assigned to
        is in the lowest **threshold** percent of all of the confidences.
        """
        return self.cython_gaussian_mixture.anomaly_detect(X, threshold)
    def visualize_clustering(self, color_dict):
        """
        :param color_dict : parameter of the label a cluster was assigned to to its color (must be matplotlib compatible)
        The color dict could be {0 : "green", 1 : "blue", 2 : "red"} for example.
        This method will not work for data that has only 1 dimension (univariate.) It will plot the data you just
        gave in the fit() method.
        """
        is_multivariate = self.cython_gaussian_mixture._is_multivariate()

        if is_multivariate :
            # then visualize
            y_pred = self.cython_gaussian_mixture.predict(self.X)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            for index, prediction in enumerate(y_pred):
                plt.scatter(self.X[index][0], self.X[index][1], color=color_dict[prediction])
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
            plt.title("Visualized Clustering with Gaussian Mixtures")
            plt.show()

        else :
            raise ValueError("MODEL HAS DATA WITH ONLY 1 DIMENSION. THIS METHOD CANNOT BE USED THEN (VISUALIZATION DIFFICULTIES.)")

    def visualize_elbow_curve(self, min_n_clusters = 2, max_n_clusters = 5):
        """
        :param min_n_clusters : the minimum value of n_clusters to be tried
        :param max_n_clusters : the max value of n_clusters to be tried

        This method tries different values for n_cluster, from min_n_cluster to max_n_cluster, and then plots
        their AIC and BIC metrics. Finding the n_cluster that leads to the "elbow" is probably the optimal n_cluster
        value.
        """
        n_clusters_to_BIC, n_clusters_to_AIC = [], []
        max_n_clusters += 1
        original_k = self.k
        for k in range(min_n_clusters, max_n_clusters) :
            self.k = k
            self.cython_gaussian_mixture = self.cython_gmm_class(k, self.retries, self.max_iters, self.kmeans_init)
            self.cython_gaussian_mixture.fit(self.X)
            n_clusters_to_BIC.append(self.cython_gaussian_mixture.bic())
            n_clusters_to_AIC.append(self.cython_gaussian_mixture.aic())

        self.k = original_k

        self.cython_gaussian_mixture = self.cython_gmm_class(original_k, self.retries, self.max_iters, self.kmeans_init)
        self.cython_gaussian_mixture.fit(self.X)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(np.array(list(range(min_n_clusters, max_n_clusters))), np.array(n_clusters_to_BIC), color = 'green', label = "BIC")
        plt.plot(np.array(list(range(min_n_clusters, max_n_clusters))), np.array(n_clusters_to_AIC), color = "blue", label = "AIC")

        plt.scatter(list(range(min_n_clusters, max_n_clusters)), n_clusters_to_BIC, color = "green")
        plt.scatter(list(range(min_n_clusters, max_n_clusters)), n_clusters_to_AIC, color = "blue")

        plt.xticks([k for k in range(min_n_clusters, max_n_clusters)])
        
        plt.legend()
        plt.xlabel("Number of Clusters")
        plt.ylabel("Information Criteria")
        plt.title("Elbow Curve in Gaussian Mixture Models")
        plt.show()


