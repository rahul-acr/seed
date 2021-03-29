import numpy as np
from network import networkutil


def train(lstm, x, hidden_state, cell_state, error, alpha=0.001):

    net_error = 0

    # hidden_state concatenated with input
    h_i = np.hstack((hidden_state, x))

    f = lstm.syn_forget.fire(h_i)
    i = lstm.syn_input.fire(h_i)
    ca = lstm.syn_candidate_cell.fire(h_i)
    o = lstm.syn_output.fire(h_i)

    cell_state = cell_state * f + ca * i
    hidden_state = o * networkutil.tanh(cell_state)

    out = lstm.syn_projection.fire(hidden_state)
    err = error

    net_error += np.abs(err)

    err = lstm.syn_projection.adjust_weights(hidden_state, out, err, alpha, ret_err_p=True)

    e1 = lstm.syn_output.adjust_weights(h_i, o, err, alpha, ret_err_p=True)
    e2 = lstm.syn_forget.adjust_weights(h_i, f, err, alpha, ret_err_p=True)
    e3 = lstm.syn_input.adjust_weights(h_i, i, err, alpha, ret_err_p=True)
    e4 = lstm.syn_candidate_cell.adjust_weights(h_i, ca, err, alpha, ret_err_p=True)

    propagated_err = (e1 + e2 + e3 + e4)

    propagated_err = propagated_err[:, :lstm.hidden_size]

    return net_error, propagated_err
