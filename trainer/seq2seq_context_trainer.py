import numpy as np
import decoder_lstm_trainer
import encoder_lstm_trainer
import context_lstm_trainer
from network.seq2seq_context import Seq2seqContext


def train(seq2seq_context, dataset_x, dataset_y, data_length, epochs, alpha=0.001, v_step=100):
    # type: (Seq2seqContext, list, list, int, int , float, int) -> None
    for epoch in xrange(epochs):
        net_err = 0
        enc_err = 0

        seq2seq_context.memory_lstm.reset_cell_state()
        seq2seq_context.memory_lstm.reset_hidden_state()

        for di in xrange(data_length):
            if dataset_x[di] == 'RESET':
                seq2seq_context.memory_lstm.reset_cell_state()
                seq2seq_context.memory_lstm.reset_hidden_state()
                continue

            seq2seq_context.encoder.reset_cell_state()
            seq2seq_context.encoder.reset_hidden_state()

            seq2seq_context.decoder.reset_cell_state()

            thought_vector = seq2seq_context.encoder.fire_extend(dataset_x[di])

            prev_hidden_state = seq2seq_context.memory_lstm.hidden_state
            prev_cell_state = seq2seq_context.memory_lstm.cell_state

            memory_vector = seq2seq_context.memory_lstm.fire(thought_vector)

            decoder_err, err_propagated = decoder_lstm_trainer.train(seq2seq_context.decoder,
                                                                     memory_vector, dataset_y[di],
                                                                     alpha)

            err_memory, err_propagated = context_lstm_trainer.train(seq2seq_context.memory_lstm, thought_vector, prev_hidden_state, prev_cell_state,
                                       err_propagated, alpha=alpha)

            encoder_err = encoder_lstm_trainer.train(seq2seq_context.encoder, dataset_x[di], err_propagated, alpha)

            net_err += np.abs(decoder_err)
            enc_err += np.abs(encoder_err)

        if epoch % v_step == 0:
            print epoch, sum(net_err[0]), sum(enc_err[0])
