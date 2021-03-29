from serializable import Serializable
import numpy as np


class Seq2seq(Serializable):

    def __init__(self, encoder, decoder):
        Serializable.__init__(self)

        assert encoder is not None
        assert decoder is not None

        self.encoder = encoder
        self.decoder = decoder

    def fire(self, data, sos, eos, max_length=10):
        assert sos is not None

        self.encoder.reset_cell_state()
        self.encoder.reset_hidden_state()

        vector = self.encoder.fire_extend(data)

        self.decoder.set_hidden_state(vector)
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
