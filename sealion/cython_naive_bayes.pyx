import numpy as np
cimport numpy as np
from collections import Counter

cdef np.ndarray[np.float64_t, ndim = 1] _probabilties(float feature, dict label_feature_counts, dict num_class_data) :
    cdef list probs = []
    cdef int label
    cdef float prob
    for label in label_feature_counts :
        prob = (label_feature_counts[label][feature] + 0.5) / (num_class_data[label] + 1) # (num_times_in_that_label + e) / (num_label + 2e)
        probs.append(prob)
    return np.array(probs)

cdef class cy_MultinomialNaiveBayes():
    cdef dict num_class_data, label_feature_counts
    cdef set labels_set

    cpdef fit(self, x_train, y_train):
        x_train, y_train = np.array(x_train), np.array(y_train)
        if x_train.dtype == np.dtype('O') : raise ValueError("Expects same number of features in each row of data.")

        cdef list labels_set = y_train.tolist() #possible labels
        labels_set.sort()

        self.labels_set = set(labels_set)
        self.num_class_data = {lbl : np.count_nonzero(y_train == lbl) for lbl in list(self.labels_set)}

        self.label_feature_counts = {}

        cdef int label
        cdef np.ndarray data_for_label

        for label in list(self.labels_set) :
            data_for_label = x_train[np.where(y_train == label)]
            data_for_label = data_for_label.flatten().flatten().flatten()
            self.label_feature_counts[label] = Counter(data_for_label)  #for each label, give me how common each feature is


    cpdef list predict(self, x_test) :
        cdef list predictions = []
        x_test = np.array(x_test, dtype = np.float64)

        #some declarations for the for-loop
        cdef np.ndarray[np.float64_t, ndim = 1] current_probs, pred_data
        cdef float feature

        for pred_data in x_test :
            current_probs = np.zeros(len(self.labels_set), dtype = np.float64)
            for feature in pred_data :
                current_probs += np.log(_probabilties(feature, self.label_feature_counts, self.num_class_data))
            current_probs = np.exp(current_probs)
            predictions.append(list(self.labels_set)[np.argmax(current_probs)])
        return predictions

    cpdef float evaluate(self, x_test, y_test) :
        cdef list y_pred = cy_MultinomialNaiveBayes.predict(self, x_test)
        return sum([1 if pred == label else 0 for pred, label in zip(y_pred, y_test)]) / len(y_test)

cdef class cy_BernoulliNaiveBayes():
    cdef dict num_class_data, label_feature_counts
    cdef set labels_set, vocabulary

    cpdef fit(self, x_train, y_train):
        x_train, y_train = np.array(x_train), np.array(y_train)
        if x_train.dtype == np.dtype('O') : raise ValueError("Expects same number of features in each row of data.")

        cdef list labels_set = y_train.tolist() #possible labels
        labels_set.sort()

        self.labels_set = set(labels_set)

        if len(self.labels_set) != 2 :
            raise ValueError("Bernoulli Naive Bayes is for binary classification (just 2 classes.)")
        self.num_class_data = {lbl : np.count_nonzero(y_train == lbl) for lbl in list(self.labels_set)}

        self.label_feature_counts = {}

        cdef int label
        cdef np.ndarray data_for_label

        for label in list(self.labels_set) :
            data_for_label = x_train[np.where(y_train == label)]
            data_for_label = data_for_label.flatten().flatten().flatten()
            self.label_feature_counts[label] = Counter(data_for_label)  #for each label, give me how common each feature is

        self.vocabulary = set(x_train.flatten().flatten().flatten().tolist()) # all of the vocabulary

    cpdef list predict(self, x_test) :
        cdef list predictions = []
        x_test = np.array(x_test, dtype = np.float64)

        #some declarations for the for-loop
        cdef np.ndarray[np.float64_t, ndim = 1] current_probs, pred_data
        cdef float feature

        for pred_data in x_test :
            current_probs = np.zeros(len(self.labels_set), dtype = np.float64)
            for vocab_word in self.vocabulary :
                if vocab_word in pred_data :
                    # here we do have the vocab word
                    # so we do take the probabilities
                    current_probs += np.log(_probabilties(vocab_word, self.label_feature_counts, self.num_class_data))
                else :
                    # that means we do not have the vocab word in there
                    # so here we do take probs of 1 - chance of seeing it
                    current_probs += np.log(1.0 - _probabilties(vocab_word, self.label_feature_counts,  self.num_class_data))
            current_probs = np.exp(current_probs)
            predictions.append(list(self.labels_set)[np.argmax(current_probs)])
        return predictions

    cpdef float evaluate(self, x_test, y_test) :
        cdef list y_pred = cy_BernoulliNaiveBayes.predict(self, x_test)
        return sum([1 if pred == label else 0 for pred, label in zip(y_pred, y_test)]) / len(y_test)


def _normal_distribution(value, sigma_and_mu) :
    sigma, mu = sigma_and_mu
    return 1 / (sigma * np.sqrt ( 2 * np.pi)) * np.exp(-0.5 * np.power( (value - mu) / sigma , 2))

cdef class cy_GaussianNaiveBayes() :
    cdef set labels_set
    cdef int num_data
    cdef dict num_class_data, labels_to_feature_distribs

    cpdef void fit(self, x_train, y_train) :
        '''everything is non-numeric'''

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if x_train.dtype == np.dtype('O') : raise ValueError("Expects same number of features in each row of data.")

        cdef list labels_set = y_train.tolist()  # possible labels
        labels_set.sort()
        self.labels_set = set(labels_set)

        cdef float lbl
        self.num_class_data = {lbl: np.count_nonzero(y_train == lbl) for lbl in self.labels_set}
        self.num_data = len(x_train)

        self.labels_to_feature_distribs = {}
        cdef int feature
        cdef dict feature_dict
        cdef np.ndarray dataset
        cdef float label

        for label in list(self.labels_set) :
            feature_dict = {}
            for feature in range(len(x_train[0])) :
                dataset = x_train[np.where(y_train == label)][:, feature]
                feature_dict[feature] = [np.std(dataset), np.mean(dataset)]
            self.labels_to_feature_distribs[label] = feature_dict

    cpdef list predict(self, x_test):
        cdef list predictions = []
        x_test = np.array(x_test, dtype = np.float64)

        cdef np.ndarray[np.float64_t, ndim = 1] pred_data
        cdef list probs, sigma_and_mu
        cdef int label, feature
        cdef dict feature_dict
        cdef float prob_label

        for pred_data in x_test :
            probs = []
            for label in list(self.labels_set) :
                feature_dict = self.labels_to_feature_distribs[label]
                prob_label = np.log(self.num_class_data[label] / self.num_data)
                for feature, sigma_and_mu in feature_dict.items() :
                    prob_label += np.log(_normal_distribution(pred_data[feature], sigma_and_mu) + 1e-29) # add the log() of each feature's likelihood on the curve for that feature and the current label we are seeing probabilities on
                probs.append(np.exp(prob_label)) # needs the exp
            predictions.append(list(self.labels_set)[np.argmax(np.array(probs))])

        return predictions

    cpdef float evaluate(self, x_test, y_test) :
        cdef list y_pred = cy_GaussianNaiveBayes.predict(self, x_test)
        return sum([1 if pred == label else 0 for pred, label in zip(y_pred, y_test)]) / len(y_test)
