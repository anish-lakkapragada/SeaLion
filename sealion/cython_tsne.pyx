import numpy as np
cimport numpy as np

cpdef np.float64_t negative_squared_euclid_distance( np.ndarray[np.float64_t, ndim=1] vec1, np.ndarray[np.float64_t, ndim=1] vec2) :
    '''take in numpy arrays'''
    return np.float64(-1 * np.sqrt(np.sum(np.square(vec1 - vec2))) ** 2)


cpdef np.ndarray[np.float64_t, ndim=1]  give_pi(np.ndarray[np.float64_t, ndim=1]  x_i,  np.ndarray[np.float64_t, ndim=2]  X, float sigma) :
    '''somehow this will work'''
    cdef np.ndarray distances = np.apply_along_axis(negative_squared_euclid_distance, 1, X, vec2 = x_i)
    distances = np.exp(distances/(2 * sigma ** 2))
    distances[distances == 1.0] = 0
    return np.float64(distances/(np.sum(distances) + 1e-10))

cpdef np.float64_t perplexity(np.ndarray[np.float64_t, ndim=1]  p_i):
    '''perplexity for binary search'''
    return np.float64(2 ** -np.sum(p_i * np.log2(p_i + 1e-10)))

cpdef tuple binary_search_find_sigma(float desired_perplexity, np.ndarray[np.float64_t, ndim=1]  x_i, np.ndarray[np.float64_t, ndim=2]  X, float low = 1e-10, float high = 500, float tolerance = 0.05, int max_iters = 1000) :
  '''function is a function, target_num is the desired number'''
  cdef list perplexities = []
  cdef float new_sigma, new_perplexity
  cdef np.ndarray[np.float64_t, ndim=1] p_i
  while True :
    new_sigma = (high + low)/2 #midpoint aka our guess
    p_i = give_pi(x_i, X, new_sigma)
    new_perplexity = perplexity(p_i)
    perplexities.append(new_perplexity)
    if new_perplexity < desired_perplexity :
        low = new_sigma #increase low , increase mid
    if new_perplexity > desired_perplexity :
        high = new_sigma #decrease high , decrease mid
    if np.abs(new_perplexity - desired_perplexity)/desired_perplexity <= tolerance :
        break
    if np.round_(np.abs(high - low))/ high == 0  or max_iters == 0 :
        break
    max_iters -= 1
  return new_sigma, perplexities

cpdef np.ndarray[np.float64_t, ndim=1] get_sigma_vector(np.ndarray[np.float64_t, ndim=2] X, float desired_perplexity) :
  cdef list sigmas = []
  cdef np.ndarray[np.float64_t, ndim=1] x_i
  cdef list _
  cdef float sigma
  for x_i in X :
    sigma, _ = binary_search_find_sigma(desired_perplexity, x_i,  X)
    sigmas.append(sigma)
  return np.array(sigmas, dtype = np.float64)

cpdef np.ndarray[np.float64_t, ndim=2] y_init( np.ndarray[np.float64_t, ndim=2] observations, int new_ndims) :
  cdef dict variances = {} #variance : dimennsion
  cdef int dim
  for dim in range(len(observations[0])) :
    variances[np.var(observations[:, dim])] = dim
  cdef list highest_variances = sorted(variances, reverse = True)[:new_ndims] #contains the highest variances
  cdef list highest_dimensions = [variances[highest_variance] for highest_variance in highest_variances]
  return np.array(observations[:, tuple(sorted(highest_dimensions))], dtype = np.float64)


cpdef np.ndarray[np.float64_t, ndim=2] get_Q_matrix( np.ndarray[np.float64_t, ndim=2] Y) :
    cdef np.ndarray[np.float64_t, ndim=2] q_matrix = np.apply_along_axis(get_qi, 1, Y, Y = Y)
    np.fill_diagonal(q_matrix, 0)
    return np.array(q_matrix/q_matrix.sum(), dtype = np.float64)

cpdef np.ndarray[np.float64_t, ndim=1] get_qi( np.ndarray[np.float64_t, ndim=1] y_i, np.ndarray[np.float64_t, ndim=2] Y) :
    '''used in get_Q_matrix for vectorization, just returns the Eij-1 not the denum'''
    cdef np.ndarray q_i = -1 * np.apply_along_axis(negative_squared_euclid_distance, 1, Y, vec2= y_i)
    q_i = np.power((q_i + 1), -1)
    return np.array(q_i, dtype = np.float64)

#get the P matrix (to be symmetrized)
cpdef np.ndarray[np.float64_t, ndim=2] get_P_matrix(np.ndarray[np.float64_t, ndim=2]  X, np.ndarray[np.float64_t, ndim=1] sigmas) :
  cdef int i
  cdef np.ndarray[np.float64_t, ndim = 2] P_matrix = np.array([give_pi(X[i], X, sigmas[i]) for i in range(len(X))])
  np.fill_diagonal(P_matrix, 0)
  return np.array(P_matrix/P_matrix.sum(), dtype = np.float64)

cpdef np.ndarray[np.float64_t, ndim=2] get_symmetrized_P_matrix(np.ndarray[np.float64_t, ndim=2] P_matrix) :
  '''P_matrix should be square'''
  return np.array((P_matrix + P_matrix.T)/(2 * len(P_matrix[0])), dtype = np.float64)


#let's build the tSNE algorithm
cdef class cy_tSNE() :
  cdef int new_ndims, max_iters
  cdef float learning_rate, perplexity, momentum

  def __cinit__(self, new_ndims, perplexity = 10, learning_rate = 200, momentum = 0.9,  max_iters = 10000) :
    self.new_ndims = new_ndims
    self.perplexity = perplexity
    self.learning_rate = learning_rate
    self.max_iters = max_iters
    self.momentum = momentum

  cpdef np.ndarray[np.float64_t, ndim = 2] transform(self, X) :
    '''X is the observations'''

    X = np.array(X) #turn to numpy for speed
    X = X.astype(np.float64)

    #find sigmas
    cdef np.ndarray[np.float64_t, ndim= 1] sigmas = get_sigma_vector(X, self.perplexity)

    cdef np.ndarray[np.float64_t, ndim= 2] P_matrix = get_P_matrix(X, sigmas) #get the P_matrix
    P_matrix = get_symmetrized_P_matrix(P_matrix) #symmetrize  matrix

    #y_init

    cdef np.ndarray[np.float64_t, ndim= 2] Y = y_init(X, self.new_ndims)


    #gradient descent
    #∂C/∂yi = 4 Sigma (j != i) (pij - qij)(yi - yj)(1 + ||yi - yj||^2)^-1

    cdef int iteration
    cdef np.ndarray dCdYi #changed dimension
    cdef np.ndarray[np.float64_t, ndim=2] eij_matrix_neg1, matrix1, old_dCdYi, Q_matrix
    cdef np.ndarray[np.float64_t, ndim = 3] low_dim_differences
    old_dCdYi = np.zeros((Y.shape[0], Y.shape[1]), dtype= np.float64)

    for iteration in range(self.max_iters) :
      Q_matrix = get_Q_matrix(Y)

      # vectorized gradient descent (momentum factor needed)
      eij_matrix_neg1 = np.apply_along_axis(get_qi, 1, Y, Y = Y)
      np.fill_diagonal(eij_matrix_neg1, 0)
      matrix1 = (P_matrix - Q_matrix) * eij_matrix_neg1
      low_dim_differences = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
      dCdYi = np.expand_dims(matrix1, 2) * np.expand_dims(low_dim_differences, 0 )
      dCdYi = (dCdYi * 4).sum(2)[0]
      Y -= self.learning_rate * dCdYi  + self.momentum * (dCdYi - old_dCdYi)

      if np.sum(np.mean(np.abs(dCdYi), axis = 0)/np.mean(np.abs(Y), axis = 0)) <= 0.01:
        break
      old_dCdYi = dCdYi
    return np.array(Y, dtype = np.float64)
