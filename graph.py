from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import pdb

class Shared_Model(object):

    def __init__(self, config, is_training):
        """Initialisation
            basically set the self-variables up, so that we can call them
            as variables to the model.
        """
        self.max_grad_norm = max_grad_norm = config.max_grad_norm
        self.num_steps = num_steps = config.num_steps
        self.encoder_size = encoder_size = config.encoder_size
        self.pos_decoder_size = pos_decoder_size = config.pos_decoder_size
        self.chunk_decoder_size = chunk_decoder_size = config.chunk_decoder_size
        self.batch_size = batch_size = config.batch_size
        self.vocab_size = vocab_size = config.vocab_size
        self.num_pos_tags = num_pos_tags = config.num_pos_tags
        self.num_chunk_tags = num_chunk_tags = config.num_chunk_tags
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.word_embedding_size = word_embedding_size = config.word_embedding_size
        self.pos_embedding_size = pos_embedding_size = config.pos_embedding_size
        self.num_shared_layers = num_shared_layers = config.num_shared_layers
        self.argmax = config.argmax

        # add input size - size of pos tags
        self.pos_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                          num_pos_tags])
        self.chunk_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                            num_chunk_tags])

        def _shared_layer(input_data, config):
            """Build the model to decoding

            Args:
                input_data = size batch_size X num_steps X embedding size

            Returns:
                output units
            """
            cell = rnn_cell.BasicLSTMCell(config.encoder_size)

            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, config.num_steps, input_data)]

            if is_training and config.keep_prob < 1:
                cell = rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)

            cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

            initial_state = cell.zero_state(config.batch_size, tf.float32)

            encoder_outputs, encoder_states = rnn.rnn(cell, inputs,
                                                      initial_state=initial_state,
                                                      scope="encoder_rnn")

            return encoder_outputs, initial_state

        def _pos_private(encoder_units, config):
            """Decode model for pos

            Args:
                encoder_units - these are the encoder units
                num_pos - the number of pos tags there are (output units)

            returns:
                logits
            """
            with tf.variable_scope("pos_decoder"):
                cell = rnn_cell.BasicLSTMCell(config.pos_decoder_size,
                                              forget_bias=1.0)

                if is_training and config.keep_prob < 1:
                    cell = rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=config.keep_prob)

                initial_state = cell.zero_state(config.batch_size, tf.float32)

                # puts it into batch_size X input_size
                inputs = [tf.squeeze(input_, [1])
                          for input_ in tf.split(1, config.num_steps,
                                                 encoder_units)]

                decoder_outputs, decoder_states = rnn.rnn(cell, inputs,
                                                          initial_state=initial_state,
                                                          scope="pos_rnn")

                output = tf.reshape(tf.concat(1, decoder_outputs),
                                    [-1, config.pos_decoder_size])

                softmax_w = tf.get_variable("softmax_w",
                                            [config.pos_decoder_size,
                                             config.num_pos_tags])
                softmax_b = tf.get_variable("softmax_b", [config.num_pos_tags])
                logits = tf.matmul(output, softmax_w) + softmax_b

            return logits, decoder_states

        def _chunk_private(encoder_units, pos_prediction, config):
            """Decode model for chunks

            Args:
                encoder_units - these are the encoder units:
                [batch_size X encoder_size] with the one the pos prediction
                pos_prediction:
                must be the same size as the encoder_size

            returns:
                logits
            """
            # concatenate the encoder_units and the pos_prediction

            pos_prediction = tf.reshape(pos_prediction,
                [batch_size, num_steps, pos_embedding_size])
            chunk_inputs = tf.concat(2, [pos_prediction, encoder_units])

            with tf.variable_scope("chunk_decoder"):
                cell = rnn_cell.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0)

                if is_training and config.keep_prob < 1:
                    cell = rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=config.keep_prob)

                initial_state = cell.zero_state(config.batch_size, tf.float32)

                # this function puts the 3d tensor into a 2d tensor: batch_size x input size
                inputs = [tf.squeeze(input_, [1])
                          for input_ in tf.split(1, config.num_steps,
                                                 chunk_inputs)]

                decoder_outputs, decoder_states = rnn.rnn(cell,
                                                          inputs, initial_state=initial_state,
                                                          scope="chunk_rnn")

                output = tf.reshape(tf.concat(1, decoder_outputs),
                                    [-1, config.chunk_decoder_size])

                softmax_w = tf.get_variable("softmax_w",
                                            [config.chunk_decoder_size,
                                             config.num_chunk_tags])
                softmax_b = tf.get_variable("softmax_b", [config.num_chunk_tags])
                logits = tf.matmul(output, softmax_w) + softmax_b

            return logits, decoder_states

        def _loss(logits, labels):
            """Calculate loss for both pos and chunk
                Args:
                    logits from the decoder
                    labels - one-hot
                returns:
                    loss as tensor of type float
            """
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                    labels,
                                                                    name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            (_, int_targets) = tf.nn.top_k(labels, 1)
            (_, int_predictions) = tf.nn.top_k(logits, 1)
            num_true = tf.reduce_sum(tf.cast(tf.equal(int_targets, int_predictions), tf.float32))
            accuracy = num_true / (num_steps*batch_size)
            return loss, accuracy, int_predictions, int_targets

        def _training(loss, config, m):
            """Sets up training ops and also...

            Create a summarisor for tensorboard

            Creates the optimiser

            The op returned from this is what is passed to session run

                Args:
                    loss float
                    learning_rate float

                returns:

                Op for training
            """
            # Create the gradient descent optimizer with the
            # given learning rate.
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.apply_gradients(zip(grads, tvars))
            return train_op


        word_embedding = tf.get_variable("word_embedding", [vocab_size, word_embedding_size])
        inputs = tf.nn.embedding_lookup(word_embedding, self.input_data)
        pos_embedding = tf.get_variable("pos_embedding", [num_pos_tags, pos_embedding_size])

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        encoding, intial_state = _shared_layer(inputs, config)
        self.initial_state = intial_state

        encoding = tf.pack(encoding)
        encoding = tf.transpose(encoding, perm=[1, 0, 2])

        pos_logits, pos_states = _pos_private(encoding, config)
        pos_loss, pos_accuracy, pos_int_pred, pos_int_targ = _loss(pos_logits, self.pos_targets)
        self.pos_loss = pos_loss

        self.pos_int_pred = pos_int_pred
        self.pos_int_targ = pos_int_targ

        # choose either argmax or dot product for pos
        if config.argmax==1:
            pos_to_chunk_embed = tf.nn.embedding_lookup(pos_embedding,pos_int_pred)
        else:
            pos_to_chunk_embed = tf.matmul(tf.nn.softmax(pos_logits),pos_embedding)


        chunk_logits, chunk_states = _chunk_private(encoding, pos_to_chunk_embed, config)
        chunk_loss, chunk_accuracy, chunk_int_pred, chunk_int_targ = _loss(chunk_logits, self.chunk_targets)
        self.chunk_loss = chunk_loss

        self.chunk_int_pred = chunk_int_pred
        self.chunk_int_targ = chunk_int_targ
        self.joint_loss = chunk_loss + pos_loss

        # return pos embedding
        self.pos_embedding = pos_embedding

        if not is_training:
            return

        self.pos_op = _training(pos_loss, config, self)
        self.chunk_op = _training(chunk_loss, config, self)
        self.joint_op = _training(chunk_loss + pos_loss, config, self)
