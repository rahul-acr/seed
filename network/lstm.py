import numpy as np

import networkutil
from synapse import Synapse
from serializable import Serializable


class LSTM(Serializable):

    def __init__(self, hidden_size, input_size, output_size, projection_layer):
        Serializable.__init__(self)

        self.hidden_size = hidden_size

        self.input_size = input_size
        self.output_size = output_size

        self.hidden_state = np.zeros((1, self.hidden_size))
        self.cell_state = np.zeros((1, self.hidden_size))

        self.syn_forget = Synapse(self.hidden_size + self.input_size, self.hidden_size, networkutil.sigmoid_activation)
        self.syn_input = Synapse(self.hidden_size + self.input_size, self.hidden_size, networkutil.sigmoid_activation)
        self.syn_output = Synapse(self.hidden_size + self.input_size, self.hidden_size, networkutil.sigmoid_activation)
        self.syn_candidate_cell = Synapse(self.hidden_size + self.input_size,
                                          self.hidden_size, networkutil.tanh_activation)

        self.syn_projection = Synapse(self.hidden_size, self.output_size, projection_layer)

    def set_hidden_state(self, hidden_state):
        self.hidden_state = hidden_state

    def reset_cell_state(self):
        # type: (None) -> None
        """
        Reset cell state of LSTM

        :return: None
        """
        self.cell_state *= 0

    def reset_hidden_state(self):
        # type: (None) -> None
        """
        Reset hidden state of LSTM

        :return: None
        """
        self.hidden_state *= 0

    def _feed(self, data):
        # type: (np.array) -> None
        """
        Feed a single feature frame into LSTM. Hidden state and cell state are updated accordingly.
        Projection layer is not fired and output of LSTM is not calculated.

        :return: None
        """
        h_i = np.hstack((self.hidden_state, data))

        f = self.syn_forget.fire(h_i)
        i = self.syn_input.fire(h_i)
        ca = self.syn_candidate_cell.fire(h_i)
        o = self.syn_output.fire(h_i)

        self.cell_state = self.cell_state * f + ca * i
        self.hidden_state = o * networkutil.tanh(self.cell_state)

    def fire(self, data):
        # type: (np.array) -> np.array
        """
        Feed a single feature frame into LSTM. Hidden state and cell state are updated accordingly.
        Projection layer is also fired and output of LSTM is returned.

        :param data: the data to feed
        :return: output of LSTM
        """
        self._feed(data)
        return self.syn_projection.fire(self.hidden_state)

    def fire_extend(self, data_i):
        """
        Feed a list of token/vector into the network. This works similar to fire()
        but is faster for a list of data. fire() supports only one token/vector

        :param data_i: list of tokens/vectors to be to be feed
        :return: the hidden state created after feeding the data
        """
        for data in data_i:
            self._feed(data)

        return self.syn_projection.fire(self.hidden_state)
