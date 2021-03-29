import numpy as np
import decoder_lstm_trainer
import encoder_lstm_trainer


def train(seq2seq, dataset_x, dataset_y, data_length, epochs, alpha=0.001, v_step=100):
    # type: (seq2seq, list, list, int, int, float, int) -> None

    for epoch in xrange(epochs):
        net_err = 0
        enc_err = 0
        for di in xrange(data_length):
            seq2seq.encoder.reset_cell_state()
            seq2seq.encoder.reset_hidden_state()
            seq2seq.decoder.reset_cell_state()

            hidden_state = seq2seq.encoder.fire_extend(dataset_x[di])
            decoder_err, err_propagated = decoder_lstm_trainer.train(seq2seq.decoder, hidden_state, dataset_y[di], alpha)
            encoder_err = encoder_lstm_trainer.train(seq2seq.encoder, dataset_x[di], err_propagated, alpha)

            net_err += np.abs(decoder_err)
            enc_err += np.abs(encoder_err)

        if epoch % v_step == 0:
            print epoch, sum(net_err[0]), sum(enc_err[0])
