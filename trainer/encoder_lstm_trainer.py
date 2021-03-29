import numpy as np
from seed.network import networkutil
from seed.network.lstm import LSTM


def train(lstm, x, error, alpha=0.001):
    # type: (LSTM, list, float, float) -> float
    net_err = None
    cell_state = np.full((1, lstm.hidden_size), 0.0)
    hidden_state = np.full((1, lstm.hidden_size), 0.0)
    length = len(x)
    for idx in xrange(length):
        # hidden_state concatenated with input
        h_i = np.hstack((hidden_state, x[idx]))
        f = lstm.syn_forget.fire(h_i)
        i = lstm.syn_input.fire(h_i)
        ca = lstm.syn_candidate_cell.fire(h_i)
        o = lstm.syn_output.fire(h_i)

        cell_state = cell_state * f + ca * i
        hidden_state = o * networkutil.tanh(cell_state)
        out = lstm.syn_projection.fire(hidden_state)

        if idx == length - 1:
            err = error

            net_err = np.abs(err)

            err = lstm.syn_projection.adjust_weights(hidden_state, out, err, ret_err_p=True)
            lstm.syn_output.adjust_weights(h_i, o, err, alpha)
            lstm.syn_forget.adjust_weights(h_i, f, err, alpha)
            lstm.syn_input.adjust_weights(h_i, i, err, alpha)
            lstm.syn_candidate_cell.adjust_weights(h_i, ca, err, alpha)

    return net_err
