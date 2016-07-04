from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import random

import tensorflow as tf
import tensorflow.python.platform

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import model_reader as reader
import numpy as np
import pdb
from graph import Shared_Model

import saveload


def run_epoch(session, m, words, pos, chunk, pos_vocab_size, chunk_vocab_size,
              verbose=False, valid=False, model_type='JOINT'):
    """Runs the model on the given data."""
    epoch_size = ((len(words) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    comb_loss = 0.0
    pos_total_loss = 0.0
    chunk_total_loss = 0.0
    iters = 0
    accuracy = 0.0
    pos_predictions = []
    pos_true = []
    chunk_predictions = []
    chunk_true = []
    state = m.initial_state.eval()

    for step, (x, y_pos, y_chunk) in enumerate(reader.create_batches(words, pos, chunk, m.batch_size,
                                               m.num_steps, pos_vocab_size, chunk_vocab_size)):

        if model_type == 'POS':
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.pos_op
        elif model_type == 'CHUNK':
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.chunk_op
        else:
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.joint_op

        joint_loss, _, pos_int_pred, chunk_int_pred, pos_int_true, \
            chunk_int_true, pos_loss, chunk_loss = \
            session.run([m.joint_loss, eval_op, m.pos_int_pred,
                         m.chunk_int_pred, m.pos_int_targ, m.chunk_int_targ,
                         m.pos_loss, m.chunk_loss],
                        {m.input_data: x,
                         m.pos_targets: y_pos,
                         m.chunk_targets: y_chunk,
                         m.initial_state: state})
        comb_loss += joint_loss
        chunk_total_loss += chunk_loss
        pos_total_loss += pos_loss
        iters += 1
        if verbose and step % 5 == 0:
            if model_type == 'POS':
                costs = pos_total_loss
                cost = pos_loss
            elif model_type == 'CHUNK':
                costs = chunk_total_loss
                cost = chunk_loss
            else:
                costs = comb_loss
                cost = joint_loss
            print("Type: %s,cost: %3f, total cost: %3f" % (model_type, cost, costs))

        pos_int_pred = np.reshape(pos_int_pred, [m.batch_size, m.num_steps])
        pos_predictions.append(pos_int_pred)
        pos_true.append(pos_int_true)

        chunk_int_pred = np.reshape(chunk_int_pred, [m.batch_size, m.num_steps])
        chunk_predictions.append(chunk_int_pred)
        chunk_true.append(chunk_int_true)

    return (comb_loss / iters), pos_predictions, chunk_predictions, pos_true, \
        chunk_true, (pos_total_loss / iters), (chunk_total_loss / iters)
