# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import sys
import numpy as np
cimport numpy as np
import cython
from libc.math cimport exp
from tqdm import tqdm
import matplotlib.pyplot as plt

##########################################################
##                                                      ##
##                Math/Utility Functions                ##
##                                                      ##
##########################################################

cdef inline double sigmoid(double z):
    return 1 / (1 + exp(-z))

cdef inline double sigmoid_deriv(double z):
    sig_z = sigmoid(z)
    return sig_z * (1 - sig_z)

cdef int argmax(double[::1] array):
    cdef Py_ssize_t i, max = 0
    for i in range(1,10):
        if array[i] > array[max]:
            max = i
    return max

# applies activation function piece-wise to input vector
cdef void activate_vector(double[:] vector, Py_ssize_t vector_size):
    cdef Py_ssize_t i
    for i in range(vector_size):
        vector[i] = sigmoid(vector[i])

cdef inline double get_output_error(double output, int target):
    return output * (1-output) * (target - output)

# I modified the basic matrix multiplication algorithm to 
# simply be a matrix-vector multiplication algorithm. 
# this is simpler, and, therefore, faster.
cdef void mat_vec_mul(
        double[:,::1] mat,
        double[::1] vec,
        double[::1] out
        ):
    cdef Py_ssize_t i, j
    cdef double d

    for i in range(mat.shape[0]):
        out[i] = 0.0
        for j in range(mat.shape[1]):
            out[i] += mat[i,j] * vec[j]


###########################################################
##                                                       ##
##              Neural Network Cython Class              ##
##                                                       ##
###########################################################



cdef class NN: # Neural_Network:
    cdef double[:,::1] weights_x
    cdef double[:,::1] weights_h
    cdef double[:,::1] prev_weight_x_change
    cdef double[:,::1] prev_weight_h_change
    cdef double[::1] hidden_layer
    cdef double[::1] output_layer
    cdef int n_hidden
    cdef int n_input
    cdef int n_output
    cdef double momentum
    cdef double learning_rate

    def __cinit__(
            NN self,
            int n_hidden = 100,
            int n_input = 784,
            int n_output = 10,
            double momentum = 0.9,
            double learning_rate = 0.1
            ):
        n_hidden += 1 # +1 for bias
        n_input += 1
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_output = n_output
        self.momentum = momentum
        self.learning_rate = learning_rate

        # initialize random weights from uniform distrobution [-0.5,0.5]
        self.weights_x = np.random.rand(n_hidden, n_input) - .5
        self.weights_h = np.random.rand(n_output, n_hidden) - .5
        
        self.prev_weight_x_change = np.zeros((n_hidden, n_input))
        self.prev_weight_h_change = np.zeros((n_output, n_hidden))

        self.hidden_layer = np.empty(n_hidden, dtype=np.float_)
        self.output_layer = np.empty(n_output, dtype=np.float_)

    cdef void feed_forward(NN self, double[::1] inputs):
        mat_vec_mul(self.weights_x, inputs, self.hidden_layer)
        activate_vector(self.hidden_layer, self.n_hidden)
        self.hidden_layer[0] = 1.0 # reset bias term

        mat_vec_mul(self.weights_h, self.hidden_layer, self.output_layer)
        activate_vector(self.output_layer, self.n_output)

    cdef int get_result(NN self, double[::1] inputs):
        self.feed_forward(inputs)
        return argmax(self.output_layer)

    cdef double get_accuracy(
            NN self,
            double[:,::1] examples,
            int[::1] targets
            ):
        cdef Py_ssize_t n = examples.shape[0]
        cdef int[::1] results = np.zeros(n, dtype=np.intc)
        cdef Py_ssize_t i

        for i in range(n):
            results[i] = targets[i] == self.get_result(examples[i])

        return np.mean(results)

    cdef void back_prop(NN self, double[:,::1] inputs, int[::1] targets):
        cdef Py_ssize_t i, k, j, n
        cdef int[::1] target = np.zeros(self.n_output, dtype=np.intc)
        cdef double[::1] errors_xh = np.empty(self.n_hidden)
        cdef double[::1] errors_ho = np.empty(self.n_output)
#         cdef double[::1] prev_weight_x_change = np.zeros(self.n_input)
#         cdef double[::1] prev_weight_h_change = np.zeros(self.n_hidden)
        cdef double[::1] weighted_output_error = np.empty(self.n_hidden)
        cdef double weight_change

        for n in range(targets.shape[0]):
            target[targets[n]] = 1
            self.feed_forward(inputs[n])

           ## Output Layer loop
            # determine error from hidden layer to output layer
            for k in range(self.n_output):#, nogil=True):
                errors_ho[k] = get_output_error(self.output_layer[k], target[k])

           ## Hidden Layer loop
            # determine error from input layer to hidden layer
            # while in j loop, update weights
            for j in range(self.n_hidden):
                weighted_output_error[j] = 0.0
                for k in range(self.n_output):
                    weighted_output_error[j] += self.weights_h[k,j] * errors_ho[k]

                    # update weights_h for every jth hidden 
                    # unit and every kth output unit
                    weight_change = self.learning_rate * \
                                    errors_ho[k] * \
                                    self.hidden_layer[j] + \
                                    self.momentum * \
                                    self.prev_weight_h_change[k,j]
                    self.prev_weight_h_change[k,j] = weight_change
                    self.weights_h[k,j] += weight_change

                # error from input layer to hidden layer
                errors_xh[j] = self.hidden_layer[j] * \
                                (1-self.hidden_layer[j]) * \
                                weighted_output_error[j]

                # update weights_x for every ith input 
                # feature and jth hidden unit
                for i in range(self.n_input):#, nogil=True):
                    weight_change = self.learning_rate * \
                                    errors_xh[j] * \
                                    inputs[n,i] + \
                                    self.momentum * \
                                    self.prev_weight_x_change[j,i]
                    self.prev_weight_x_change[j,i] = weight_change
                    self.weights_x[j,i] += weight_change

            target[targets[n]] = 0

    def speed_test(self,
            double[:,::1] training_examples,
            int[::1] training_targets,
            double[:,::1] testing_examples,
            int[::1] testing_targets,
            details=True
            ):
        cdef int e, n_epochs = 50;
        cdef np.ndarray[np.float64_t,ndim=1] acc_training = np.empty(n_epochs + 1)
        cdef np.ndarray[np.float64_t,ndim=1] acc_testing = np.empty(n_epochs + 1)
        
        
        prev_weight_x_change = np.zeros(785)
        prev_weight_h_change = np.zeros(101)

        if details:
            print("N training examples:",training_examples.shape[0])

            print("\nNeural Net details: " + \
                "\n\tHidden Units: " + str(self.n_hidden-1) + \
                "\n\tLearning Rate: " + str(self.learning_rate) + \
                "\n\tMomentum: " + str(self.momentum) + \
                "\n\tTraining for " + str(n_epochs) + " epochs"
                )
        sys.stdout.flush()

        acc_training[0] = self.get_accuracy(training_examples, training_targets)
        acc_testing[0] = self.get_accuracy(testing_examples, testing_targets)

        #for e in range(n_epochs):
        for e in tqdm(range(n_epochs)):
            # Train on all examples
            self.back_prop(training_examples, training_targets)

            # Evaluate
            e += 1
            acc_training[e] = self.get_accuracy(training_examples, training_targets)
            acc_testing[e] = self.get_accuracy(testing_examples, testing_targets)

        acc_training *= 100
        acc_testing *= 100
        
        np.around(acc_training, decimals=1, out=acc_training)
        np.around(acc_testing, decimals=1, out=acc_testing)

        print("Training: " + str(acc_training[e]) + '%')
        print("Testing:  " + str(acc_testing[e]) + '%')

        return acc_training, acc_testing

    @cython.wraparound(True)
    def plot_accuracy(NN self,
            np.ndarray training,
            np.ndarray testing):
        plt.plot(training, label="Training")
        plt.plot(testing, label="Testing")

        plt.title("Network Accuracy by Epoch" + \
                "\nHidden Units: " + str(self.n_hidden-1) + \
                ";   Learning Rate: " + str(self.learning_rate) + \
                ";   Momentum: " + str(self.momentum) + \
                ";\nMax Testing Accuracy:  " + str(np.max(testing)) + '%'\
                "\nFinal Testing Accuracy: " + str(testing[-1]) + '%'
                )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.yticks(list(range(0,101,10))[1:])

        plt.legend()
        plt.grid()
        plt.show()

    def confusion_matrix(NN self, double[:,::1] inputs, int[::1] targets):
        cdef int i, result, n = self.n_output
        cdef np.ndarray matrix = np.zeros((n+1,n+1),dtype=np.int_)
        
        for i in range(inputs.shape[0]):
            result = self.get_result(inputs[i])
            
            matrix[targets[i],result] += 1
            
            matrix[n,result] += 1
            matrix[targets[i],n] += 1
            
#         assert(np.sum(matrix[:,n]) == np.sum(matrix[n]))
        
        matrix[n,n] = np.sum(matrix[n])
        
        return matrix
