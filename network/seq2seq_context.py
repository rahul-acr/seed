from network.lstm import LSTM
import networkutil
from serializable import Serializable
import numpy as np


class Seq2seqContext(Serializable):

    def __init__(self, encoder, decoder, hidden_size):
        Serializable.__init__(self)

        assert encoder is not None
        assert decoder is not None

        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = hidden_size

        self.memory_lstm = LSTM(self.hidden_size, self.hidden_size, self.hidden_size, networkutil.tanh_activation)

    def fire(self, data, sos, eos, max_length=10):
        assert sos is not None

        self.encoder.reset_cell_state()
        self.encoder.reset_hidden_state()

        # generate thought vector from data
        thought_vector = self.encoder.fire_extend(data)
        memory_vector = self.memory_lstm.fire(thought_vector)

        self.decoder.set_hidden_state(memory_vector)
        self.decoder.reset_cell_state()

        res = self.decoder.fire(sos)
        output = [res]

        for i in xrange(max_length):
            res = self.decoder.fire(res)
            if np.argmax(res) == np.argmax(eos):
                output.append(res)
                break
            output.append(res)

        return output
