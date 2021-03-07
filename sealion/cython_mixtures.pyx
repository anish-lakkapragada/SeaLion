import numpy as np
cimport numpy as np
from scipy import stats

cdef class cy_GaussianMixture() :
    cdef int k, retries, max_iters
    cdef bint kmeans_init, multivariate
    cdef univariate_model, multivariate_model

    def __cinit__(self, int n_clusters = 5, int retries = 3, int max_iters = 200, bint kmeans_init = True):
        self.k = n_clusters
        self.retries = retries
        self.max_iters = max_iters
        self.kmeans_init = kmeans_init
        self.univariate_model = UnivariateGaussianMixture(n_clusters=n_clusters, retries=retries,max_iters=max_iters)
        self.multivariate_model = MultivariateGaussianMixture(n_clusters=n_clusters, retries=retries,max_iters=max_iters, kmeans_init=kmeans_init)

        if self.k == 1 :
            print("It is not recommended to use a gaussian mixture model with data you expect to just have one cluster (one gaussian.)".upper())
    cpdef fit(self, X):
        X = np.array(X)
        cdef int dimensions
        _, dimensions = X.shape

        if dimensions == 1:
            self.multivariate = False
        elif dimensions > 1 :
            self.multivariate = True

        if self.multivariate : self.multivariate_model.fit(X)
        else :
            if self.kmeans_init : print("Gaussian Mixtures with data of 1 dimension do not support KMeans initialization. Such initialization "
                                                "is turned off.".upper())
            self.univariate_model.fit(X)

    cpdef np.ndarray predict(self, X):
        if self.multivariate :
            return self.multivariate_model.predict(X)
        else :
            return self.univariate_model.predict(X)

    cpdef np.ndarray soft_predict(self, X):
        if self.multivariate :
            return self.multivariate_model.soft_predict(X)
        else :
            return self.univariate_model.soft_predict(X)

    cpdef np.ndarray confidence_samples(self, X):
        if self.multivariate :
            return self.multivariate_model.confidence_samples(X)
        else :
            return self.univariate_model.confidence_samples(X)

    cpdef double aic(self):
        if self.multivariate :
            return self.multivariate_model.aic()
        else :
            return self.univariate_model.aic()

    cpdef double bic(self):
        if self.multivariate :
            return self.multivariate_model.bic()
        else :
            return self.univariate_model.bic()

    cpdef np.ndarray anomaly_detect(self, X, threshold): # fix this mehtod
        if self.multivariate :
            return self.multivariate_model.anomaly_detect(X, threshold)
        else :
            return self.univariate_model.anomaly_detect(X, threshold)

    cpdef bint _is_multivariate(self) :
        return self.multivariate


cdef float multivariate_Gaussian(np.ndarray x, np.ndarray mu, np.ndarray cov_matrix) :
    # my slow implementation of a multivariate Gaussian function
    cdef int D = len(x)
    return 1/np.sqrt(np.power(2 * np.pi, D) * np.linalg.det(cov_matrix)) * np.exp(-0.5 * (x - mu).T.dot(np.linalg.inv(cov_matrix)).dot(x-mu))


cdef class MultivariateGaussianMixture :
    cdef int k, retries, max_iters
    cdef bint kmeans_init, instant_retry
    cdef kmeans
    cdef np.ndarray mixture_weights
    cdef np.ndarray mu, X
    cdef np.ndarray responsibilties
    cdef np.ndarray covariance_matrices
    cdef float log_likelihood, max_log_likelihood
    cdef int N, new_ndims


    def __cinit__(self, int n_clusters = 5, int retries = 3, int max_iters = 200, bint kmeans_init = True):
        self.k = n_clusters
        self.retries = retries
        self.max_iters = max_iters
        self.kmeans_init = kmeans_init
        from sealion.unsupervised_clustering import KMeans
        self.kmeans = KMeans
        self.instant_retry = False

    cpdef void _expectation(self, X):
        """here we do the expectation step where we recalculate all responsibilties"""

        cdef np.ndarray new_responsibilties = np.zeros((len(X), self.k))
        cdef int k_index
        cdef double pi_k
        cdef np.ndarray mu_k
        cdef np.ndarray covariance_matrices

        try :
            for k_index, pi_k, mu_k, sigma_k in zip(range(self.k), self.mixture_weights, self.mu, self.covariance_matrices) :
                new_responsibilties[:, k_index][:] = pi_k * stats.multivariate_normal.pdf(X, mean = mu_k, cov = sigma_k)
        except Exception :
            self.instant_retry = True

        # before normalization, find the log likelihood
        self.log_likelihood = np.sum(np.log(np.sum(new_responsibilties, axis = 1)))

        cdef np.ndarray normalization_factor = np.expand_dims(np.sum(new_responsibilties, axis = 1), 1)
        new_responsibilties /= normalization_factor

        self.responsibilties = new_responsibilties

    cpdef void _maximization(self, X):
        """here we update the means, covariance matrices, and mixture weights to increase log-likelihood"""
        cdef int k_index
        cdef np.ndarray responsibilties_at_k, updated_mu_k, data_point
        cdef np.ndarray updated_sigma_k
        cdef float r_nk

        for k_index in range(self.k) :
            responsibilties_at_k = self.responsibilties[:, k_index] # vector of N responsibilities for a given cluster
            N_k = np.sum(responsibilties_at_k) # summation of the the datasets responsibility to a given cluster
            # get updated mu_k
            updated_mu_k = np.sum(np.expand_dims(responsibilties_at_k, 1) * X, axis = 0) / N_k

            # get updated covariance matrix for k cluster
            updated_sigma_k = np.zeros((self.new_ndims, self.new_ndims))
            for data_point, r_nk in zip(X, responsibilties_at_k) :
                data_point = (np.array(data_point) - np.array(self.mu[k_index])).reshape(-1, 1)
                updated_sigma_k += r_nk * np.dot(data_point, data_point.T)
            updated_sigma_k /= N_k

            # get the updated mixture_weight
            updated_pi_k = N_k / len(X)

            # update all
            self.mixture_weights[k_index] = updated_pi_k
            self.covariance_matrices[k_index] = updated_sigma_k
            self.mu[k_index] = updated_mu_k

    cpdef np.ndarray predict(self, X):
        X = np.array(X)
        MultivariateGaussianMixture._expectation(self, X)
        return np.array(np.apply_along_axis(np.argmax, 1, self.responsibilties))

    cpdef np.ndarray soft_predict(self, X):
        X = np.array(X)
        MultivariateGaussianMixture._expectation(self, X)
        return self.responsibilties

    cpdef np.ndarray confidence_samples(self, X):
        cdef np.ndarray prediction_indices = MultivariateGaussianMixture.predict(self, X)
        return np.array([responsibility[chosen_index] for responsibility, chosen_index in zip(self.responsibilties, prediction_indices)])

    cpdef void fit(self, X):
        X = np.array(X)
        self.X = X
        self.N, self.new_ndims = X.shape

        # now we have to init mu, the covariance matrix, and mixture weight for each cluster
        self.mixture_weights = (np.ones(self.k) * 1/self.k)

        # kmeans initialization!
        cdef kmeans

        if self.kmeans_init :
            kmeans = self.kmeans(k = self.k)
            _ = kmeans.fit_predict(X)
            self.mu = kmeans._get_centroids()
        else :
            self.mu = np.random.randn(self.k, self.new_ndims) # for each k cluster, give the number of dims

        self.covariance_matrices = np.array([np.identity(self.new_ndims) for k_index in range(self.k)]) # create an D * D matrix for each k cluster

        # don't init responsibilities, calculate them in the first expectation step

        cdef dict tries_dict = {} # likelihood : [weights, means, sigmas]
        cdef bint converged = False
        cdef int trial, num_converge, iteration
        cdef old_likelihood

        for trial in range(self.retries) :
            # time to start doing the iterations of expectation-maximization!
            old_likelihood = None # this is the old likelihood
            num_converge = 0
            for iteration in range(self.max_iters) :

                # expectation step, evaluate all responsibilties

                MultivariateGaussianMixture._expectation(self, X)

                if self.instant_retry :
                    self.instant_retry = False
                    break

                # now we update (maximization)

                MultivariateGaussianMixture._maximization(self, X)

                if old_likelihood == None :
                    old_likelihood = self.log_likelihood

                else :
                    if (self.log_likelihood - old_likelihood) < 0.001 :
                        num_converge += 1
                    if num_converge == 3 :
                        break # model has converged

                # otherwise, keep going
                old_likelihood = self.log_likelihood

            if num_converge == 3 :
                converged = True # it actually converged here, not just went through the maximum amount of iterations

            tries_dict[self.log_likelihood] = [self.mixture_weights, self.mu, self.covariance_matrices]


        # finally choose the one that did best
        self.mixture_weights, self.mu, self.covariance_matrices = tries_dict[max(tries_dict)]
        self.max_log_likelihood = max(tries_dict)

        if not converged :
            # just went through the loops, never actually converged
            print("GAUSSIAN MIXTURE MODEL FAILED CONVERGENCE. PLEASE RETRY WITH MORE RETRIES IN THE INIT "
                          "AND MAKE SURE YOU ARE USING KMEANS_INIT (DEFAULT PARAMETER.)")

    cpdef float bic(self):
        return np.log(self.N) * 3 * self.k  - 2 * self.max_log_likelihood # no need for log as self.max_likelihood is already the log optimization

    cpdef float aic(self):
        return 2 * 3 * self.k - 2 * self.max_log_likelihood

    cpdef np.ndarray anomaly_detect(self, X, threshold):
        # huge thanks to handsonml v2 for showing me how to do this
        cdef np.ndarray probabilities = MultivariateGaussianMixture.confidence_samples(self, X)
        cdef float prob_threshold = np.percentile(probabilities, threshold)
        cdef np.ndarray anomalies = np.ones(len(X), bool)
        anomalies[np.where(probabilities < prob_threshold)] = True # the indices of all of the outliers
        anomalies[np.where(probabilities >= prob_threshold)] = False
        return anomalies


cdef class UnivariateGaussianMixture :
    cdef int k, retries, max_iters, N
    cdef float log_likelihood, max_log_likelihood
    cdef np.ndarray mixture_weights, sigmas, mu, X
    cdef np.ndarray responsibilties
    cdef bint instant_retry

    def __init__(self, n_clusters = 5, retries = 3, max_iters = 200):
        self.k = n_clusters
        self.retries = retries
        self.max_iters = max_iters
        self.instant_retry = False
    cpdef void _expectation(self, X):
        """here we do the expectation step where we recalculate all responsibilties"""
        cdef np.ndarray new_responsibilties = np.zeros((len(X), self.k))
        cdef int k_index
        cdef float mu_k, sigma_k
        try :
            for k_index, pi_k, mu_k, sigma_k in zip(range(self.k), self.mixture_weights, self.mu, self.sigmas) :
                new_responsibilties[:, k_index] = pi_k * stats.multivariate_normal.pdf(X, mean = mu_k, cov = sigma_k)
        except Exception :
            self.instant_retry = True
        # before normalization, find the log likelihood
        self.log_likelihood = np.sum(np.log(np.sum(new_responsibilties, axis = 1)))

        cdef np.ndarray normalization_factor = np.expand_dims(np.sum(new_responsibilties, axis = 1), 1)

        new_responsibilties /= normalization_factor

        self.responsibilties = new_responsibilties

    cpdef void _maximization(self, X):
        """here we update the means, covariance matrices, and mixture weights to increase log-likelihood"""
        cdef int k_index
        cdef np.ndarray responsibilties_at_k
        cdef float updated_mu_k, updated_pi_k, updated_sigma_k

        for k_index in range(self.k) :
            responsibilties_at_k = self.responsibilties[:, k_index] # vector of N responsibilities for a given cluster
            N_k = np.sum(responsibilties_at_k) # summation of the the datasets responsibility to a given cluster
            # get updated mu_k
            updated_mu_k = np.sum(responsibilties_at_k * X) / N_k # remember X is 1D

            # get the updated sigma over here
            X_de_mean = X - self.mu[k_index]
            updated_sigma_k = np.sum(responsibilties_at_k * np.power(X_de_mean, 2))  /N_k

            # get the updated mixture_weight
            updated_pi_k = N_k / len(X)

            # update all
            self.mixture_weights[k_index] = updated_pi_k
            self.sigmas[k_index] = updated_sigma_k
            self.mu[k_index] = updated_mu_k

    cpdef np.ndarray predict(self, X):
        X = np.array(X)
        UnivariateGaussianMixture._expectation(self, X)
        return np.array(np.apply_along_axis(np.argmax, 1, self.responsibilties))

    cpdef np.ndarray soft_predict(self, X):
        X = np.array(X)
        UnivariateGaussianMixture._expectation(self, X)
        return self.responsibilties

    cpdef np.ndarray confidence_samples(self, X):
        prediction_indices = UnivariateGaussianMixture.predict(self, X)
        return np.array([responsibility[chosen_index] for responsibility, chosen_index in zip(self.responsibilties, prediction_indices)])

    cpdef void fit(self, X):
        X = np.array(X)
        self.X = X.flatten()
        self.N = len(X)

        # now we have to init mu, the covariance matrix, and mixture weight for each cluster
        self.mixture_weights = (np.ones(self.k) * 1/self.k)

        # kmeans initialization!
        self.mu = np.random.randn(self.k) # for each k cluster, give the number of dims

        self.sigmas = np.abs(np.random.randn(self.k)) # give the sigma here for that too

        # don't init responsibilities, calculate them in the first expectation step

        cdef dict tries_dict = {} # likelihood : [weights, means, sigmas]
        cdef bint converged = False
        cdef old_likelihood
        cdef int trial, num_converge, iteration

        for trial in range(self.retries) :
            # time to start doing the iterations of expectation-maximization!
            old_likelihood = None # this is the old likelihood
            num_converge = 0
            for iteration in range(self.max_iters) :

                # expectation step, evaluate all responsibilties

                UnivariateGaussianMixture._expectation(self, X)

                if self.instant_retry :
                    self.instant_retry = False
                    break

                # now we update (maximization)

                UnivariateGaussianMixture._maximization(self, X)

                if old_likelihood == None :
                    old_likelihood = self.log_likelihood

                else :
                    if (self.log_likelihood - old_likelihood) < 0.001 :
                        num_converge += 1
                    if num_converge == 3 :
                        break # model has converged

                # otherwise, keep going
                old_likelihood = self.log_likelihood

            if num_converge == 3 :
                converged = True # it actually converged here, not just went through the maximum amount of iterations

            tries_dict[self.log_likelihood] = [self.mixture_weights, self.mu, self.sigmas]


        # finally choose the one that did best
        self.mixture_weights, self.mu, self.sigmas = tries_dict[max(tries_dict)]
        self.max_log_likelihood = max(tries_dict)

        if not converged :
            # just went through the loops, never actually converged
            print("GAUSSIAN MIXTURE MODEL FAILED CONVERGENCE. PLEASE RETRY WITH MORE RETRIES IN THE INIT "
                          "AND MAKE SURE YOU ARE USING KMEANS_INIT (DEFAULT PARAMETER.)")

    cpdef float bic(self):
        return np.log(self.N) * 3  - 2 * self.max_log_likelihood # no need for log as self.max_likelihood is already the log optimization

    cpdef float aic(self):
        return 2 * 3 - 2 * self.max_log_likelihood

    cpdef np.ndarray anomaly_detect(self, X, threshold):
        # huge thanks to handsonml v2 for showing me how to do this
        cdef np.ndarray probabilities = UnivariateGaussianMixture.confidence_samples(self, X)
        cdef float prob_threshold = np.percentile(probabilities, threshold)

        cdef np.ndarray anomalies = np.ones(len(X), bool)
        anomalies[np.where(probabilities < prob_threshold)] = True
        anomalies[np.where(probabilities >= prob_threshold)] = False
        return anomalies
