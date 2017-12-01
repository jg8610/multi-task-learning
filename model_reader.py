"""Utilities for parsing CONll text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time
import pandas as pd
import csv
import pdb
import pickle

import numpy as np

"""
    1.0. Utility Methods
"""


def read_tokens(filename, padding_val, col_val=-1):
    # Col Values
    # 0 - words
    # 1 - POS
    # 2 - tags

    with open(filename, 'rt', encoding='utf8') as csvfile:
            r = csv.reader(csvfile, delimiter=' ')
            words = np.transpose(np.array([x for x in list(r) if x != []])).astype(object)
    # padding token '0'
    print('reading ' + str(col_val) + ' ' + filename)
    if col_val!=-1:
        words = words[col_val]
    return np.pad(
        words, pad_width=(padding_val, 0), mode='constant', constant_values=0)


def _build_vocab(filename, padding_width, col_val):
    # can be used for input vocab
    data = read_tokens(filename, padding_width, col_val)
    counter = collections.Counter(data)
    # get rid of all words with frequency == 1
    counter = {k: v for k, v in counter.items() if v > 1}
    counter['<unk>'] = 10000
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def _build_tags(filename, padding_width, col_val):
    # can be used for classifications and input vocab
    data = read_tokens(filename, padding_width, col_val)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    tag_to_id = dict(zip(words, range(len(words))))
    if col_val == 1:
        pickle.dump(tag_to_id,open('pos_to_id.pkl','wb'))
        pickle.dump(count_pairs,open('pos_counts.pkl','wb'))

    return tag_to_id


"""
    1.1. Word Methods
"""


def _file_to_word_ids(filename, word_to_id, padding_width):
    # assumes _build_vocab has been called first as is called word to id
    data = read_tokens(filename, padding_width, 0)
    default_value = word_to_id['<unk>']
    return [word_to_id.get(word, default_value) for word in data]

"""
    1.2. tag Methods
"""


def _int_to_tag(tag_int, tag_vocab_size):
    # creates the one-hot vector
    a = np.empty(tag_vocab_size)
    a.fill(0)
    np.put(a, tag_int, 1)
    return a


def _seq_tag(tag_integers, tag_vocab_size):
    # create the array of one-hot vectors for your sequence
    return np.vstack(_int_to_tag(
                     tag, tag_vocab_size) for tag in tag_integers)


def _file_to_tag_classifications(filename, tag_to_id, padding_width, col_val):
    # assumes _build_vocab has been called first and is called tag to id
    data = read_tokens(filename, padding_width, col_val)
    return [tag_to_id[tag] for tag in data]


def raw_x_y_data(data_path, num_steps):
    train = "train.txt"
    valid = "validation.txt"
    train_valid = "train_val_combined.txt"
    comb = "all_combined.txt"
    test = "test.txt"

    train_path = os.path.join(data_path, train)
    valid_path = os.path.join(data_path, valid)
    train_valid_path = os.path.join(data_path, train_valid)
    comb_path = os.path.join(data_path, comb)
    test_path = os.path.join(data_path, test)

    # checking for all combined
    if not os.path.exists(data_path + '/train_val_combined.txt'):
        print('writing train validation combined')
        train_data = pd.read_csv(data_path + '/train.txt', sep= ' ',header=None)
        validation_data = pd.read_csv(data_path + '/validation.txt', sep= ' ',header=None)

        comb = pd.concat([train_data,validation_data])
        comb.to_csv(data_path + '/train_val_combined.txt', sep=' ', index=False, header=False)

    if not os.path.exists(data_path + '/all_combined.txt'):
        print('writing combined')
        test_data = pd.read_csv(data_path + '/test.txt', sep= ' ',header=None)
        train_data = pd.read_csv(data_path + '/train.txt', sep= ' ',header=None)
        val_data = pd.read_csv(data_path + '/validation.txt', sep=' ', header=None)

        comb = pd.concat([train_data,val_data,test_data])
        comb.to_csv(data_path + '/all_combined.txt', sep=' ', index=False, header=False)

    word_to_id = _build_vocab(train_path, num_steps-1, 0)
    # use the full training set for building the target tags
    pos_to_id = _build_tags(comb_path, num_steps-1, 1)

    chunk_to_id = _build_tags(comb_path, num_steps-1, 2)

    word_data_t = _file_to_word_ids(train_path, word_to_id, num_steps-1)
    pos_data_t = _file_to_tag_classifications(train_path, pos_to_id, num_steps-1, 1)
    chunk_data_t = _file_to_tag_classifications(train_path, chunk_to_id, num_steps-1, 2)

    word_data_v = _file_to_word_ids(valid_path, word_to_id, num_steps-1)
    pos_data_v = _file_to_tag_classifications(valid_path, pos_to_id, num_steps-1, 1)
    chunk_data_v = _file_to_tag_classifications(valid_path, chunk_to_id, num_steps-1, 2)

    word_data_c = _file_to_word_ids(train_valid_path, word_to_id, num_steps-1)
    pos_data_c = _file_to_tag_classifications(train_valid_path, pos_to_id, num_steps-1, 1)
    chunk_data_c = _file_to_tag_classifications(train_valid_path, chunk_to_id, num_steps-1, 2)

    word_data_test = _file_to_word_ids(test_path, word_to_id, num_steps-1)
    pos_data_test = _file_to_tag_classifications(test_path, pos_to_id, num_steps-1, 1)
    chunk_data_test = _file_to_tag_classifications(test_path, chunk_to_id, num_steps-1, 2)

    return word_data_t, pos_data_t, chunk_data_t, word_data_v, \
        pos_data_v, chunk_data_v, word_to_id, pos_to_id, chunk_to_id, \
        word_data_test, pos_data_test, chunk_data_test, word_data_c, \
        pos_data_c, chunk_data_c


def create_batches(raw_words, raw_pos, raw_chunk, batch_size, num_steps, pos_vocab_size,
                   chunk_vocab_size):
    """Tokenize and create batches From words (inputs), raw_pos (output 1), raw_chunk(output 2). The parameters
    of the minibatch are defined by the batch_size, the length of the sequence.

    :param raw_words:
    :param raw_pos:
    :param raw_chunk:
    :param batch_size:
    :param num_steps:
    :param pos_vocab_size:
    :param chunk_vocab_size:
    :return:
    """

    def _reshape_and_pad(tokens, batch_size, num_steps):
        tokens = np.array(tokens, dtype=np.int32)
        data_len = len(tokens)
        post_padding_required = (batch_size*num_steps) - np.mod(data_len, batch_size*num_steps)

        tokens = np.pad(tokens, (0, post_padding_required), 'constant',
                        constant_values=0)
        epoch_length = len(tokens) // (batch_size*num_steps)
        tokens = tokens.reshape([batch_size, num_steps*epoch_length])
        return tokens

    """
    1. Prepare the input (word) data
    """
    word_data = _reshape_and_pad(raw_words, batch_size, num_steps)
    pos_data = _reshape_and_pad(raw_pos, batch_size, num_steps)
    chunk_data = _reshape_and_pad(raw_chunk, batch_size, num_steps)

    """
    3. Do the epoch thing and iterate
    """
    data_len = len(raw_words)
    # how many times do you iterate to reach the end of the epoch
    epoch_size = (data_len // (batch_size*num_steps)) + 1

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = word_data[:, i*num_steps:(i+1)*num_steps]
        y_pos = np.vstack(_seq_tag(pos_data[tag, i*num_steps:(i+1)*num_steps],
                          pos_vocab_size) for tag in range(batch_size))
        y_chunk = np.vstack(_seq_tag(chunk_data[tag, i*num_steps:(i+1)*num_steps],
                            chunk_vocab_size) for tag in range(batch_size))
        y_pos = y_pos.astype(np.int32)
        y_chunk = y_chunk.astype(np.int32)
        yield (x, y_pos, y_chunk)


def _int_to_string(int_pred, d):

    # integers are the Values
    keys = []
    for x in int_pred:
        keys.append([k for k, v in d.items() if v == (x)])

    return keys


def res_to_list(res, batch_size, num_steps, to_id, w_length):

    tmp = np.concatenate([x.reshape(batch_size, num_steps)
                          for x in res], axis=1).reshape(-1)
    tmp = np.squeeze(_int_to_string(tmp, to_id))
    return tmp[range(num_steps-1, w_length)].reshape(-1,1)
