import numpy as np
cimport numpy as np
from libc.math cimport exp

cdef inline double sigmoid(double z) nogil:
    return 1 / (1 + exp(-z))

cdef inline double sigmoid_deriv(double z) nogil:
    sig_z = sigmoid(z)
    return sig_z * (1 - sig_z)


def preprocess_data(filename, max_rows=None):
    cdef np.ndarray data = np.genfromtxt(filename, delimiter=',', max_rows=max_rows)

    np.random.shuffle(data)
    # use dtype=np.intc for compatibility w/ C layer
    cdef np.ndarray targets = data[:,0].astype(np.intc)
    data[1:,:] /= 255 # all values into [0,1] range
    data[:,0] = 1 # adds bias(=1) column

    return data, targets

cdef double[:] feed_forward(
        double[:,:] weights_x,
        double[:,:] weights_h,
        double[:] inputs
        ):
    cdef double[:] hidden_output, final_output
    cdef Py_ssize_t i, j, I, J

    np.matmul(inputs, weights_x, out=hidden_output)

    hidden_output[:] = sigmoid(hidden_output[:])
    #I = hidden_output.shape[0]
    #for i in prange(I, nogil=True):
    #    hidden_output[i] = sigmoid(hidden_output[i])

    np.matmul(hidden_output, weights_h, out=final_output)

    final_output[:] = sigmoid(final_output[:])

    return final_ouput

cdef int get_result(weigths_x, weights_h, inputs):

