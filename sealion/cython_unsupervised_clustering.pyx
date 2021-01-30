import math
import numpy as np
cimport numpy

def distance(x, y):
    return math.sqrt(sum([(x_i - y_i) ** 2 for x_i, y_i in zip(x, y)]))

cdef int indexOf(list listy, list element):
    indice = -1
    if element in listy :
        indice = listy.index(element)
    return indice

cdef list vector_mean(list vectors):
    mean = np.zeros(len(vectors[0]))
    cdef list vector
    for vector in vectors:
        mean += np.array(vector)
    return (mean / len(vectors)).tolist()

cdef list find_neighbors(list data, list point, double eps):
    cdef list data_point
    neighbors = [data_point for data_point in data if distance(data_point, point) <= eps and data_point != point]
    return neighbors



cdef find_key(dict dictionary, item):
    item_key = -1
    for key in dictionary:
        if indexOf(dictionary[key], item) != - 1: item_key = key
    return item_key


cdef reg_find_key(dictionary, item):
    item_key = None
    for key in dictionary:
        if dictionary[key] == item:
            item_key = key
            break
    return item_key


cpdef bint eventually_connected(list data, list core_instance, list other_core_instance, double eps):  # returns True connected else None
    '''find all neighbors of core_instance, neighbors of neighbors, ... and see if any of them match other core instance'''
    cdef list neighbor
    neighbors = find_neighbors(data, core_instance, eps)
    if other_core_instance in neighbors:
        return True
    else:
        for neighbor in neighbors:
            return eventually_connected(remove(data, core_instance), neighbor, other_core_instance, eps)  # remove the starting point


cdef int count_core_instances_in_cluster(dict clusters):
    cdef int num_core_instances = 0
    cdef list cluster
    for cluster in clusters:
        num_core_instances += len(cluster)
    return num_core_instances

cdef list new_list
cdef list remove(list listy, list element):
    '''new_list = [l_i for l_i in listy]
    new_list.remove(element)
    return new_list'''
    new_list = [l_i for l_i in listy if l_i != element]
    return new_list

import random


cdef class CythonDBSCAN():
    cdef double eps
    cdef dict clusters, points_to_classes, visualize_clusters

    cpdef list fit_predict(self, x_train,  double eps=0.5, int min_neighbors=7):

        try:
            x_train = x_train.tolist()
        except Exception:
            pass

        self.eps = eps
        cdef list core_instances = []
        cdef dict core_instances_to_indices_points = {}
        cdef int point
        cdef list neighbors
        for point in range(len(x_train)):
            neighbors = find_neighbors(x_train, x_train[point], self.eps)
            if len(neighbors) >= min_neighbors:
                core_instances.append(x_train[point])
                core_instances_to_indices_points[tuple(x_train[point])] = [point]  # contains the indices for the points that belong to a core_instance

        # what do we do with the noisy points - nothing right?

        # now we merge the clusters
        cdef list connected_core_instances = []
        cdef int label_idx
        cdef list core_instance, other_core_instance
        cdef dict clusters = {label_idx: [core_instance] for label_idx, core_instance in enumerate(core_instances)}


        cdef list add_indices
        for core_instance in core_instances:
            if core_instance in connected_core_instances:
                continue
            for other_core_instance in core_instances:
                if other_core_instance in connected_core_instances:
                    continue

                if eventually_connected(x_train, core_instance, other_core_instance,
                                        self.eps):
                    del clusters[reg_find_key(clusters, [other_core_instance])]  # delete the other_core_instance cluster
                    clusters[find_key(clusters, core_instance)].append(
                        other_core_instance)  # whichever cluster has the core_instance shall have the other core instance now
                    add_indices = core_instances_to_indices_points[tuple(other_core_instance)]
                    del core_instances_to_indices_points[tuple(other_core_instance)]
                    core_instances_to_indices_points[tuple(core_instance)] += add_indices
                    connected_core_instances.append(
                        other_core_instance)
            connected_core_instances.append(core_instance)


        self.visualize_clusters = {}  # put labels back into 0s and 1s

        cdef int curr_lbl = 0
        cdef int index
        cdef list cluster
        for index, cluster in clusters.items():
            self.visualize_clusters[curr_lbl] = cluster
            curr_lbl += 1

        cdef dict new_clusters = {}
        cdef list new_cluster = []
        cdef int label
        cdef list c_i
        for label, cluster in clusters.items():
            new_cluster = []
            for c_i in cluster:
                if tuple(c_i) in list(core_instances_to_indices_points.keys()):
                    new_cluster.append(c_i)
            new_clusters[label] = new_cluster

        self.clusters = {}
        curr_lbl = 0
        for index, cluster in new_clusters.items():
            self.clusters[curr_lbl] = cluster
            curr_lbl += 1

        # remove any core instances that aren't a part of the merged cluster

        cdef dict predictions_dict = {}
        cdef dict points_to_classes = {} #tuple(point) : label
        cdef list current_prediction_indices = []
        for label, cluster in self.clusters.items():
            '''cluster is a collection of core_instances, each with their own indices of the data points'''
            for core_instance in cluster:
                current_prediction_indices += core_instances_to_indices_points[tuple(core_instance)]
                points_to_classes[tuple(core_instance)] = label #also make sure that this is here
            predictions_dict[label] = current_prediction_indices
            current_prediction_indices = []

        cdef list predictions = [-1] * len(x_train)
        cdef list indices
        for label, indices in predictions_dict.items():
            for indice in indices:
                predictions[indice] = label
                points_to_classes[tuple(x_train[indice])] = label


        #make sure repeats don't get the -1s
        for point in range(len(x_train)) :
            try :
                #assuming not noise!
                predictions[point] = points_to_classes[tuple(x_train[point])]
            except Exception : pass

        self.points_to_classes = points_to_classes
        return predictions
    cpdef dict get_clusters(self):
        return self.clusters
    cpdef dict get_points_to_classes(self) :
        return self.points_to_classes

cdef class CythonKMeans():
    cdef list centroids, x_train, data
    cdef int k , n_features
    cdef dict clusters, labeled_clusters

    cpdef dict get_labeled_clusters(self) :
        return self.labeled_clusters

    cpdef list get_centroids(self) :
        return self.centroids

    cpdef list get_data(self) :
        return self.data
    cpdef list fit_predict(self, x_train, int k=5):
        try : x_train = x_train.tolist()
        except Exception : pass
        self.k = k
        self.centroids = []
        self.n_features = len(x_train[0])
        self.data = x_train

        self.centroids.append(random.sample(x_train, 1)[0]) #first choose a random sample
        cdef int _
        cdef list probabilities = []
        cdef list new_centroid, x_i,  closest_centroid, distances_to_chosen_centroids
        cdef int numerator, denominator
        for _ in range(k - 1) :
            #just choose the instance with the highest probability
            probabilities = []
            for x_i in x_train :
                if indexOf(self.centroids, x_i) != -1 :
                    probabilities.append(0.0)
                    continue
                distances_to_chosen_centroids = [distance(list(x_i), centroid) for centroid in self.centroids]
                closest_centroid  = self.centroids[distances_to_chosen_centroids.index(min(distances_to_chosen_centroids))] #closest centroid already chosen
                numerator = (distance(closest_centroid, x_i)) ** 2
                denominator = np.sum([distance(closest_centroid, x_j) for x_j in x_train])
                probabilities.append(numerator)
            #choose the highest probability
            new_centroid = x_train[probabilities.index(max(probabilities))] #whichever has the highest probabilities find that data point for x and make that the new centroid
            self.centroids.append(new_centroid)

        cdef list old_centroids, distances_to_points, delete_centroid, distances, points, centroid_keys_to_delete
        cdef tuple centroid_key
        cdef dict current_centroid_to_points
        while True:

            old_centroids = self.centroids

            # assign each point to a mean
            distances_to_points = [[[distance(point, centroid) for centroid in self.centroids] , tuple(point)] for point in
                                  x_train]  # point : [distance_centroid1, distance_centroid 2, distance_centroid3]
            current_centroid_to_points = {tuple(centroid): [] for centroid in self.centroids}
            self.clusters = current_centroid_to_points
            for distances, point in distances_to_points :
                closest_centroid = self.centroids[distances.index(min(distances))]  # find centroid it belongs to
                current_centroid_to_points[tuple(closest_centroid)].append(list(point))

            #remove all centroids with 0 points
            centroid_keys_to_delete = []
            for centroid, points in current_centroid_to_points.items() :
                if len(points) == 0 :
                    '''delete'''
                    centroid_keys_to_delete.append(centroid)
                    delete_centroid = list(centroid)
                    self.centroids.remove(delete_centroid)


            for centroid_key in centroid_keys_to_delete : del current_centroid_to_points [centroid_key]

            # update each centroids by taking the mean of its points
            for centroid, points in current_centroid_to_points.items():
                new_centroid = vector_mean(points)
                self.centroids[self.centroids.index(list(centroid))] = new_centroid  # update the centroids

            if old_centroids == self.centroids:
                break  # no change

        cdef list list_of_points = [centroid_to_point[1] for centroid_to_point in current_centroid_to_points.items()]
        cdef int label_idx
        self.labeled_clusters = {label_idx: points for label_idx, points in enumerate(list_of_points)}

        cdef list labels = [-1] * len(x_train)

        cdef list cloned_x_train = [x_i for x_i in x_train] #safe storage
        cdef int label
        for label, points in self.labeled_clusters.items():
            for point in points:
                index_point = cloned_x_train.index(point)
                labels[index_point] = label
                cloned_x_train[index_point] = None #whatever point

        return labels

    cpdef double inertia_score(self) :
        '''returns the inertia score - plot inertia scores'''
        cdef inertia_for_clusters = []
        cdef list points
        cdef tuple centroid
        for centroid, points in self.clusters.items() :
            inertia_for_clusters.append(sum([distance(point, list(centroid)) for point in points]))
        return sum(inertia_for_clusters)/len(inertia_for_clusters)


