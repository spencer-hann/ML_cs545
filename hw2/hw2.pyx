# cython: boundscheck=False, wraparound=False, nonecheck=False

import sys
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport exp


def preprocess_data(filename, max_rows=None):
    cdef np.ndarray data = np.genfromtxt(filename, delimiter=',', max_rows=max_rows)

    np.random.shuffle(data)
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

cdef inline double get_output_error(double output, int target) nogil:
    return output * (1-output) * (target - output)

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

    def __cinit__(
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

        self.weights_x = np.random.rand(n_hidden, n_input) - .5
        self.weights_h = np.random.rand(n_output, n_hidden) - .5

        #self.hidden_layer = np.empty(n_hidden, dtype=np.float)
        #self.output_layer = np.empty(n_output, dtype=np.float)

    cdef void feed_forward(NN self, double[::1] inputs) with gil:
        self.hidden_layer = np.matmul(self.weights_x, inputs)
            #, out=self.hidden_layer)
        activate_vector(self.hidden_layer, self.n_hidden)

        self.output_layer = np.matmul(self.weights_h, self.hidden_layer)
            #, out=self.output_layer)
        activate_vector(self.output_layer, self.n_output)

    cdef int get_result(NN self, double[::1] inputs):
        self.feed_forward(inputs)
        return np.argmax(self.output_layer)

    cdef double get_accuracy(
            NN self,
            double[:,::1] examples,
            int[::1] targets
            ):
        cdef int result
        cdef int[::1] results = np.zeros(targets.shape[0], dtype=np.intc)
        cdef Py_ssize_t i

        for i in range(len(examples)):
            result = self.get_result(examples[i])
            results[i] = targets[i] == result

        return np.mean(results)

    cdef void back_prop(NN self, double[:,::1] inputs, int[::1] targets):
        cdef Py_ssize_t i, k, j
        cdef int[::1] target = np.zeros(self.n_output, dtype=np.intc)
        cdef double[::1] errors_xh = np.empty(self.n_hidden)
        cdef double[::1] errors_ho = np.empty(self.n_output)
        cdef double[::1] prev_weight_x_change = np.zeros(self.n_input)
        cdef double[::1] prev_weight_h_change = np.zeros(self.n_hidden)
        cdef double weighted_output_error
        cdef double weight_change

        for n in range(targets.shape[0]):
            target[targets[n]] = 1
            self.feed_forward(inputs[n])

           ## Output Layer loop
            # determine error from hidden layer to output layer
            for k in prange(self.n_output, nogil=True):
                errors_ho[k] = get_output_error(self.output_layer[k], target[k])

           ## Hidden Layer loop
            # determine error from input layer to hidden layer
            # while in j loop, update weights
            for j in prange(self.n_hidden, nogil=True):
                weighted_output_error = 0.0
                for k in prange(self.n_output):
                    weighted_output_error += self.weights_h[k,j] * errors_ho[k]

                    # update weights_h for every jth hidden 
                    # unit and every kth output unit
                    weight_change = self.learning_rate * \
                                    errors_ho[k] * \
                                    self.hidden_layer[j] + \
                                    self.momentum * \
                                    prev_weight_h_change[j]
                    prev_weight_h_change[j] = weight_change
                    self.weights_h[k,j] += weight_change

                # error from input layer to hidden layer
                errors_xh[j] = self.hidden_layer[j] * \
                                (1-self.hidden_layer[j]) * \
                                weighted_output_error

                # update weights_x for every ith input 
                # feature and jth hidden unit
                for i in prange(self.n_input):
                    weight_change = self.learning_rate * \
                                    errors_xh[j] * \
                                    inputs[n,i] + \
                                    self.momentum * \
                                    prev_weight_x_change[i]
                    prev_weight_x_change[i] = weight_change
                    self.weights_x[j,i] += weight_change

            target[targets[n]] = 0

    def speed_test(self,
            double[:,::1] training_examples,
            int[::1] training_targets,
            double[:,::1] testing_examples,
            int[::1] testing_targets
            ):
        print "initial accuracy: "
        print self.get_accuracy(testing_examples, testing_targets)
        print "..."

        cdef int epoch
        for epoch in range(50):
            self.back_prop(training_examples, training_targets)
            epoch += 1
            if epoch % 10 == 0:
                print "\nEpoch: " + str(epoch)
                print self.get_accuracy(training_examples, training_targets)
                print self.get_accuracy(testing_examples, testing_targets)
            print str(epoch) + '\t',
            sys.stdout.flush()
