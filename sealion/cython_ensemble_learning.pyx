"""
@author : Anish Lakkapragada
@date : 1-10-2021

Ensemble Learning is an often overshadowed and underestimated field of machine learning. Here we provide 2 algorithms
central to the game - random forests and ensemble/voting classifier. Random Forests are very especially fast
with parallel processing to fit multiple decision trees at the same time.
"""

import numpy as np
cimport numpy as np
import random
import math


cdef _train_new_predictor(predictor, name, x_train, y_train):
    predictor.fit(x_train, y_train)
    return [name, predictor]


cdef double r2_score(np.ndarray y_pred, np.ndarray y_test):

    cdef double num = np.sum(np.power(y_test - y_pred, 2))
    cdef double denum = np.sum(np.power(y_test - np.mean(y_test), 2))
    return 1 - num / denum

cdef double _perc_correct(np.ndarray y_pred, np.ndarray y_test) :
    return np.sum((y_pred == y_test).astype('int'))/len(y_pred)

cdef class CythonEnsembleClassifier() :
    cdef bint classification
    cdef dict predictors, trained_predictors
    def __cinit__(self, predictors, classification = True):
        """
        :param predictors: should be a dictoinary of {"predictor_custom_name" : class}, where the class can be like LogisticRegression(..enter_params..)
        :param classification: classification or not?
        """

        self.classification = classification

        self.predictors = predictors

    cpdef void give_trained_predictors(self, dict trained_predictors) :
        self.trained_predictors = trained_predictors
    cpdef dict get_predictors(self) :
        return self.predictors

    cpdef np.ndarray predict(self, x_test) :
        '''average all predictions, depending on whether classification or regression'''


        cdef np.ndarray[np.float64_t, ndim = 1] predictions = np.zeros(len(x_test)).astype(np.float64)
        for _, trained_predictor in self.trained_predictors.items() :
            predictions += trained_predictor.predict(x_test)

        predictions =  predictions/len(self.trained_predictors)

        if self.classification: predictions = np.round_(predictions)

        return predictions

    cpdef double evaluate(self, x_test, y_test) :

        cdef np.ndarray[np.float64_t, ndim = 1] y_pred = CythonEnsembleClassifier.predict(self, x_test)

        cdef double amount_correct
        cdef int m

        if self.classification :
            return _perc_correct(y_pred, y_test)
        else :
            return r2_score(y_pred, y_test)

    cpdef void evaluate_all_predictors(self, x_test, y_test):

        cdef str name
        for name, trained_predictor in self.trained_predictors.items() :
            print(f"{name} : {trained_predictor.evaluate(x_test, y_test)}")

    cpdef get_best_predictor(self, x_test, y_test) :
        cdef dict evaluation_to_predictors = {}
        cdef double evaluation
        for _, trained_predictor in self.trained_predictors.items() :
            evaluation = trained_predictor.evaluate(x_test, y_test)
            evaluation_to_predictors[evaluation] = trained_predictor
        return evaluation_to_predictors[max(evaluation_to_predictors)]












