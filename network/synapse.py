"""
Synapse
=======



Example usage
-------------

>>>from seed.network.synapse import Synapse
>>>synapse = Synapse(3, 2, 'sigmoid')
>>>arr = np.array([0.78, 0.21, -.16])
>>>synapse.fire(arr)
array([ 0.78,  0.21])

"""
import networkutil
import numpy as np


class Synapse:
    """
    Synapse it the base level building block in seed. All the major computation is done
    on this level. It has support of different activation functions, training to adjust
    the weights, bias nodes and learning rate.  A fully functional network can be created
    by coupling multiple Synapses. It can also be used along side other computation units
    seamlessly. Except some crucial aspects Synapse gives control to outside world so that
    it can be used creating any kind of model.
    """

    def __init__(self, in_dim, out_dim, activation_func):
        # type: (np.array , np.array, str) -> None
        """
        Initializes a synapse with given input and output dimensions and an
        activation function along with bias. For the list of activation functions check
        networkutil

        :param in_dim: input dimension
        :param out_dim: output dimension
        :param activation_func: activation function
        """
        self._in_dim = in_dim
        self._out_dim = out_dim

        self._activation_func = getattr(networkutil, activation_func)
        self._activation_derv_func = getattr(networkutil, activation_func + '_output_to_derivative')

        self._bias = np.ones((1, self._in_dim))
        self.weight_matrix = Synapse._generate_random_weights(self._in_dim, self._out_dim)
        self.bias_matrix = Synapse._generate_random_weights(self._in_dim, self._out_dim)

    def fire(self, data_i):
        # type: (np.array) -> np.array
        """
        Fires the synapse on given input data

        :param data_i: data to be feed
        :return: synapse output
        """
        return self._activation_func(np.dot(data_i, self.weight_matrix) + np.dot(self._bias, self.bias_matrix))

    def _get_adjustments(self, data_i, data_o, err, alpha):
        # type: (np.array, np.array, np.array, float) -> (np.array, np.array)
        """
        calculate adjustments of weight matrix and bias matrix of the synapse
        on a given input data , output data, error and alpha

        :param data_i: data feed as input
        :param data_o: synapse output on the input data
        :param err: error
        :param alpha: learning rate
        :return: returns adjustments of weight matrix and bias matrix as a tuple
        """
        adj_weight = alpha * np.dot(data_i.T, err) * self._activation_derv_func(data_o)
        adj_bias = alpha * np.dot(self._bias.T, err) * self._activation_derv_func(data_o)
        return adj_weight, adj_bias

    def adjust_weights(self, data_i, data_o, err, alpha=.001, ret_err_p=False):
        # type: (np.array, np.array, np.array, float, bool ) -> np.array
        """
        Adjusts the network weights depending on error.

        :param data_i: data feed as input
        :param data_o: synapse output on the input data
        :param err: ideal output - real output
        :param alpha: the learning rate. default: 0.001
        :param ret_err_p: return propagated error
        :return: returns propagated error if ret_err_p is true
        otherwise returns None
        """
        change_weight_matrix, change_bias_matrix = self._get_adjustments(data_i, data_o, err, alpha)
        self.weight_matrix += change_weight_matrix
        self.bias_matrix += change_bias_matrix

        if ret_err_p:
            return self._get_err_propagated(err, data_o)

    def _get_err_propagated(self, err, data_o):
        # type: (np.array, np.array) -> np.array
        """
        Computes the error to propagated in downstream synapses

        :param err: error
        :param data_o: output data
        :return: the error to be propagated
        """
        return np.dot(err * self._activation_derv_func(data_o), self.weight_matrix.T)

    @property
    def get_shape(self):
        """
        Returns the [ input x output ] shape of the synapse

        :return: shape of the synapse
        """
        return self._in_dim, self._out_dim

    @staticmethod
    def _generate_random_weights(in_dim, out_dim):
        # type: (int, int) -> np.array
        """
        Generates random numbers in [-1, 1] to initialize the synapse with random weights

        :param in_dim: input dimension
        :param out_dim: output dimension
        :return: a matrix of random weights in [-1, 1]
        """
        return 2 * np.random.random((in_dim, out_dim)) - 1
