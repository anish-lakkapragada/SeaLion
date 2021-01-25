"""
@author : Anish Lakkapragada
@date : 1/8/2021

Contains DBSCAN and KMeans, two unsupervised clustering algorithms. Both use Cython (compiled code in C) for faster
performance.
"""

import numpy as np

def indexOf(listy, element):
    indice = -1
    if element in listy :
        indice = listy.index(element)
    return indice

def _change_labels(predicted) :
    '''predicted is a numpy array'''
    list_predicted = predicted.tolist()
    list_predicted.sort()
    num_diff_categories = len(set(list_predicted))
    label_to_new_label = {}
    organized_diff_categories = list(set(list_predicted))
    organized_diff_categories.sort()
    penalty = 1 if indexOf(list_predicted, -1) != -1 else 0
    for _ in range(num_diff_categories) :
        label_to_new_label[organized_diff_categories[_]] = _ - penalty

    for old, new in label_to_new_label.items() : predicted[predicted == old] = new
    return predicted

class DBSCAN() :
    """
    DBSCAN is an algorithm for unsupervised clustering. That means that it identifies clusters in data, and assumes
    each cluster is a different category/label. Different than KMeans because it does not assume clusters are
    in spheres/circles; clusters can be in many different shapes (check out the moons dataset). Unscalable time complexity,
    but it does fine for < 2500 data points (less than .5 hour - 1.5 hour runtime.)

    -----
    Methods

    __init__(eps = .5, min_neighbors = 7) :
        ->> eps (default .5): epsilon value for finding nearest points in the algorithm
        ->> min_neighbors (default 7) : the minimum amount of neighbors a point needs to be
        deemed a core instance.
    fit_predict(x_train) :
        ->> takes in x_train - which must be 2D (a list or numpy array like [[]], not 1D like []) and returns labels
        based on which cluster it is assigned to.
        ->> a label may be -1 if it hasn't went into any cluster (change with eps and min_neighbors param to minimize
        that.)
    visualize_clustering(color_dict) :
        ->> if your data is in 2-Dimensions this can visualize the clustering of the data points.
        ->> A new color will be assigned for each point plotted based upon its labels.
        ->> the given color_dict is a dictionary like {-1 : "green", 0 : "red", 1 : "purple", 2 : "blue"}, where each
        label matches to a color (must be a color recognized by matplotlib). For every unique label given
        in the fit_predict() method, there must be that class to its color in the color_dict.
        ->> this method only works after fit_predict(x_train) is done and visualizes the clustering
        of x_train

    """

    def __init__(self, eps = 0.5, min_neighbors = 7):
        """
        :param eps: eps or epsilon value for finding nearest points (in the algorithm)
        :param min_neighbors: the minimum amount of neighbors a point needs to be
        deemed a core instance.
        """
        self.cython_dbscan = cython_unsupervised_clustering.CythonDBSCAN()
        self.eps = eps
        self.min_neighbors = min_neighbors
    def fit_predict(self, x_train) :
        """
        :param x_train: 2D data to cluster.
        :return: Labels, or cluster number, for each data point in x_train. -1 means it belongs
        to no cluster.
        """
        if len(np.array(x_train).shape) != 2 : raise ValueError("x_train must be 2D (even if only one sample.)")
        predicted =  np.array(self.cython_dbscan.fit_predict(x_train, self.eps, self.min_neighbors))
        return _change_labels(predicted)
    def visualize_clustering(self, color_dict):
        """
        This only works if x_train passed in the fit_predict() method has 2 dimensional inputs. This does :
        [[1, 1], [2, 2], [3, 3]] and this doesn't : [[1] ,[2], [3]]
        :param color_dict: Every unique label given by the fit_predict() method to the color you want it
        painted as in matplotlib. So if your labels are like [-1, 0, 4, 2, 1, 2, 3, 4, 5, 5] then your color_dict could
        be like {-1 : "green", 0 : "purple", 1 : "red",  2 : "blue", 3 : "orange", 4 : "yellow", 5 : "black" }
        :return: None, just show the image of the visualized clustering.
        """
        import matplotlib.pyplot as plt
        from collections import defaultdict
        plt.cla()

        # reorganize points to classes -> matplotlib is prettier that way
        point_and_class_list = []
        points_to_classes = self.cython_dbscan.get_points_to_classes()
        for point, label in points_to_classes.items():
            point_and_class_list.append([list(point), label])

        classes_to_points = defaultdict(list)
        for point_and_class in point_and_class_list:
            point, label = point_and_class[0], point_and_class[1]
            classes_to_points[label].append(point)

        for label, points in classes_to_points.items():
            plt.scatter(np.array(points)[:, 0], np.array(points)[:, 1], color=color_dict[label],
                        label="cluster : " + str(label))
        plt.legend()
        plt.title('Clustering of data with DBSCAN')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.show()

class KMeans() :
    """
    The most famous unsupervised clustering method there is. Tries to find k clusters
    (k is a parameter given by you) of the data and assign each data point to its cluster number.
    Makes the assumption that the clusters are circle/spherical, a disadvantage compared to DBSCAN.
    However, scales and performs extremely well (DBSCAN is unscalable for huge data.) This module utilizes
    KMeans++ init for the centroids.

    ----
    Methods

    __init__(k = 5)
        ->> initialization of this KMeans class.
        ->> k : number of clusters you assume are in the data. Default 5, normally from
        3 - 10 (there's a method we have here to help you find that.)
        ->> know that even if you initialize k = 3, your labels may be 0s and 1s only because KMeans found only 2 actual
        clusters, and no points belong to the 3rd cluster (use the visualize_elbow_curve() method for this.)

    fit_predict(x_train) :
        ->> takes in x_train, which must be 2D (numpy array/python list like [[]] not []) and creates
        k - clusters from it
        ->> returns the cluster number for each data point in x_train

    visualize_elbow_curve(min_k = 2, max_k = 10) :
        ->> Given a range of k-values from min_k (default 2) to max_k (default 10), it uses each k-value
        for the fit_predict() method and finds its inertia score of each clustering with the specific k-value to
        see how well it did.
        ->> uses the data given in fit_predict() - no need for entering that here
        ->> plots the k-values on the x-axis and the inertia score on the y-values
        ->> look for the elbow on the curve and the k-value where it happened. That k-value is optimal and most likely
        is the best to be used.

    visualize_clustering(color_dict) :
        ->> if your data is in 2-Dimensions this can visualize the clustering of the data points.
        ->> A new color will be assigned for each point plotted based upon its labels.
        ->> the given color_dict is a dictionary like {-1 : "green", 0 : "red", 1 : "purple", 2 : "blue"}, where each
        label matches to a color (must be a color recognized by matplotlib). For every unique label given
        in the fit_predict() method, there must be that class to its color in the color_dict.
        ->> this method only works after fit_predict(x_train) is done and visualizes the clustering
        of x_train

    """
    def __init__(self, k = 5):

        self.cython_kmeans = cython_unsupervised_clustering.CythonKMeans()
        self.k = k
    def fit_predict(self, x_train):
        """
        :param x_train: 2D data to cluster.
        :return: Labels, or cluster number, for each data point in x_train.
        """

        return np.array(self.cython_kmeans.fit_predict(x_train, self.k))

    def visualize_elbow_curve(self, min_k = 2, max_k = 10) :
        """
        Tests k-values from min_k (default 2) to max_k (default 10) and sees their inertia score in the
        clustering done. Then plots the k-values to their respective inertia score. Look for the
        elbow point on the curve, whichever k-value produces it is optimal (use that.)

        :param min_k: Lowest_k value to be tested.
        :param max_k: Maximum k-value to be tested.
        :return:
        """
        inertia_scores = []
        max_k += 1
        original_k = self.k
        for k_value in range(min_k, max_k):
            self.k = k_value
            KMeans.fit_predict(self, self.cython_kmeans.get_data())
            inertia_scores.append(self.cython_kmeans.inertia_score())
        self.k = original_k
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot([k for k in range(min_k, max_k)], inertia_scores, color="green", label="inertia scores")
        plt.scatter([k for k in range(min_k, max_k)], inertia_scores, color="green")
        plt.legend()
        plt.xlabel("k-value")
        plt.xticks([k for k in range(min_k, max_k)])
        plt.ylabel("inertia score")
        plt.title("Elbow curve in KMeans")

    def _get_centroids(self):
        return self.cython_kmeans.get_centroids()
    def visualize_clustering(self, color_dict) :
        """
        This only works if x_train passed in the fit_predict() method has 2 dimensional inputs. This does :
        [[1, 1], [2, 2], [3, 3]] and this doesn't : [[1] ,[2], [3]]
        :param color_dict: Every unique label given by the fit_predict() method to the color you want it
        painted as in matplotlib. So if your labels are like [-1, 0, 4, 2, 1, 2, 3, 4, 5, 5] then your color_dict could
        be like {-1 : "green", 0 : "purple", 1 : "red",  2 : "blue", 3 : "orange", 4 : "yellow", 5 : "black" }
        :return: None, just show the image of the visualized clustering.
        """
        import matplotlib.pyplot as plt
        plt.cla()
        for label, points in self.cython_kmeans.get_labeled_clusters().items() :
            plt.scatter([point[0] for point in points], [point[1] for point in points], color = color_dict[label], label = "cluster : " + str(label))
        plt.legend()
        plt.title('Clustering of data with KMeans')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.show()