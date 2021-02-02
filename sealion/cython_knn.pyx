import numpy as np
cimport numpy as np

cdef double distance(x1, point):
    cdef double distance_double
    x1, point = np.array(x1), np.array(point) #convert to numpy arrays (even if tuple)
    distance_double =  np.sqrt(np.sum((x1 - point) ** 2))
    x1, point = x1.tolist(), point.tolist()
    return distance_double

def r2_score(y_pred, y_test):
    y_pred, y_test = np.array(y_pred), np.array(y_test)
    num = np.sum(np.power(y_test - y_pred, 2))
    denum = np.sum(np.power(y_test - np.mean(y_test), 2))
    return 1 - num / denum


cdef class CythonKNN():

    cdef str information
    cdef bint regression
    cdef int k
    cdef dict points_to_classes

    def __cinit__(self):
        self.information = """KNearestNeighbors is arguably the simplest ML model out there. All it does is just map
        out all of the data it's given and predict a new point based on the labels of k points in the training data
        that are closest to it. If you are using it for regression (please don't - use logistic or softmax),
        it will just find the average of all k points around it, whereas for classification it will find the rounded
        mean of the k points closest to it. K should always be odd, because if it's even there can be a split. A too
        high k value may lead to points too far being considered, whereas a small one can be highly sensitive to
        outliers. Like always, hyperparameters are dependent on experimentation.
        """

    cpdef fit(self, x_train, y_train, int k=5, bint regression=False):
        x_train, y_train = np.array(x_train).tolist(), np.array(y_train).tolist()
        self.regression = regression
        self.k = k
        self.points_to_classes = {tuple(x_i): y_i for x_i, y_i in zip(x_train, y_train)}
    cpdef list predict(self, x_test):
        x_test = np.array(x_test).tolist()
        '''look at the neighbors for the k nearest points'''
        cdef list dataset_points = (list(self.points_to_classes.keys()))  # contains all dataset points
        cdef list pred_point, sorted_distances_of_each_point, closest_distances, point_preds
        cdef dict distances_to_each_point
        cdef list predictions = []
        for pred_point in x_test :
            distances_to_each_point = {distance(list(data_point), list(pred_point)) : list(data_point) for data_point in dataset_points}
            sorted_distances_of_each_point = list(distances_to_each_point.keys())
            sorted_distances_of_each_point.sort()
            closest_distances = sorted_distances_of_each_point[:self.k]
            closest_points = [distances_to_each_point[closest_distance] for closest_distance in closest_distances]
            point_preds = [self.points_to_classes[tuple(closest_point)] for closest_point in closest_points]
            predictions.append(sum(point_preds)/len(point_preds))


        numpy_preds = np.array(predictions)
        if not self.regression : numpy_preds = np.round_(numpy_preds)
        return numpy_preds.tolist()

    cpdef double evaluate(self, x_test, y_test):
        x_test, y_test = np.array(x_test).tolist(), np.array(y_test).tolist()
        cdef list y_pred = CythonKNN.predict(self, x_test)
        cdef double acc
        if not self.regression :
            acc = sum([1 if pred == label else 0 for pred, label in zip(y_pred, y_test)])
            return acc/len(y_test)
        else :
            print("started to get")
            acc = r2_score(y_pred, y_test)
            print("ended")
            return acc