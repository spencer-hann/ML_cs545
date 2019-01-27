# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport exp


def preprocess_data(filename, max_rows=None):
    cdef np.ndarray data = np.genfromtxt(filename, delimiter=',', max_rows=max_rows)

    np.random.shuffle(data)
    # use dtype=np.intc for compatibility w/ C layer
    cdef np.ndarray targets = data[:,0].astype(np.intc)
    data[1:,:] /= 255 # all values into [0,1] range
    data[:,0] = 1 # adds bias(=1) column

    return data, targets

cdef inline double sigmoid(double z) nogil:
    return 1 / (1 + exp(-z))

cdef inline double sigmoid_deriv(double z) nogil:
    sig_z = sigmoid(z)
    return sig_z * (1 - sig_z)

cdef void activate_vector(double[:] vector, Py_ssize_t vector_size) nogil:
    cdef Py_ssize_t i
    for i in prange(vector_size,nogil=True):
        vector[i] = sigmoid(vector[i])

cdef class NN: # Neural_Network:
    cdef double[:,::1] weights_x
    cdef double[:,::1] weights_h
    cdef double[::1] hidden_layer
    cdef double[::1] output_layer
    cdef int n_hidden
    cdef int n_input
    cdef int n_output
    cdef double momentum
    cdef double learning_rate

    def __init__(
            NN self,
            int n_hidden = 10,
            int n_input = 1 + 784,
            int n_output = 10,
            double momentum = 0.9,
            double learning_rate = 0.1
            ):
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_output = n_output
        self.momentum = momentum
        self.learning_rate = learning_rate

        weights_x = np.random.rand(n_hidden, n_input)
        weights_h = np.random.rand(n_output, n_hidden)
        weights_x -= .5 # range was [0,1]
        weights_h -= .5 # is now [-.5,.5]

        self.hidden_layer = np.empty(n_hidden, dtype=np.float)
        self.output_layer = np.empty(n_output, dtype=np.float)

    cdef void feed_forward(NN self, double[::1] inputs):
        np.matmul(inputs, self.weights_x, out=self.hidden_layer)
        activate_vector(self.hidden_layer, self.n_hidden)

        np.matmul(self.hidden_layer, self.weights_h, out=self.output_layer)
        activate_vector(self.output_layer, self.n_output)

    cdef int get_result(NN self, double[::1] inputs):
        cdef double[::1] results
        results = self.feed_forward(inputs)
        return np.argmax(results)

    cdef double get_accuracy(
            NN self,
            double[:,::1] examples,
            int[::1] targets
            ):
        cdef int result
        cdef int[::1] results = np.zeros(targets.shape[0], dtype=np.intc)
        cdef Py_ssize_t i

        print "testing", len(examples)
        for i in range(len(examples)):
            result = get_result(self.weights_x,self.weights_h,examples[i])
            results[i] = targets[i] == result

        print("meaning")

        return np.mean(results)

    cdef void back_prop(NN self, int[::1] targets):
        cdef Py_ssize_t k, j
        cdef double[::1] hidden_errors = np.empty(n_hidden)
        cdef double[::1] output_errors = np.empty(n_output)
        cdef double[::1] target = np.zeros(n_output)
        cdef double[::1] prev_weight_x_changes = np.zeros(n_input)
        cdef double[::1] prev_weight_h_changes = np.zeros(n_hidden)

        for i in range(targes.shape[0]):
            target[targets[i]] = 1
            # determine error from hidden layer to output layer
            for k in prange(self.n_output, nogil=True):
                output_errors[k] = self.output_layer[k] * \
                                (1-self.output_layer[k]) * \
                                (target[k] - self.output_layer[k])

            # determine error from input layer to hidden layer
            for j in prange(self.n_hidden, nogil=True):
                for k in prange(n_output):
                    weighted_output_error += self.weights_h[k,j] * output_errors[k]
                hidden_errors[j] = self.hidden_layer[j] * \
                                (1-self.hidden_layer[j]) * \
                                weighted_ouput_error


            target[targets[i]] = 0
