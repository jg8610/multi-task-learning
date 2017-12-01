from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.python.platform

import model_reader as reader
import numpy as np
import pdb
# import pandas as pd
from graph import Shared_Model
from run_epoch import run_epoch
import argparse
import saveload


class Config(object):
    """Configuration for the network"""
    init_scale = 0.1  # initialisation scale
    learning_rate = 0.001  # learning_rate (if you are using SGD)
    max_grad_norm = 5  # for gradient clipping
    num_steps = 20  # length of sequence
    word_embedding_size = 400  # size of the embedding
    encoder_size = 200  # first layer
    pos_decoder_size = 200  # second layer
    chunk_decoder_size = 200  # second layer
    max_epoch = 1  # maximum number of epochs
    keep_prob = 0.5  # for dropout
    batch_size = 64  # number of sequence
    vocab_size = 20000  # this isn't used - need to look at this
    num_pos_tags = 45  # hard coded, should it be?
    num_chunk_tags = 23  # as above
    pos_embedding_size = 400
    num_shared_layers = 1
    argmax = 0


def main(model_type, dataset_path, save_path):
    """Main"""
    config = Config()
    raw_data = reader.raw_x_y_data(
        dataset_path, config.num_steps)
    words_t, pos_t, chunk_t, words_v, \
    pos_v, chunk_v, word_to_id, pos_to_id, \
    chunk_to_id, words_test, pos_test, chunk_test, \
    words_c, pos_c, chunk_c = raw_data

    config.num_pos_tags = len(pos_to_id)
    config.num_chunk_tags = len(chunk_to_id)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        # model to train hyperparameters on
        with tf.variable_scope("hyp_model", reuse=None, initializer=initializer):
            m = Shared_Model(is_training=True, config=config)
        with tf.variable_scope("hyp_model", reuse=True, initializer=initializer):
            mvalid = Shared_Model(is_training=False, config=config)

        # model that trains, given hyper-parameters
        with tf.variable_scope("final_model", reuse=None, initializer=initializer):
            mTrain = Shared_Model(is_training=True, config=config)
        with tf.variable_scope("final_model", reuse=True, initializer=initializer):
            mTest = Shared_Model(is_training=False, config=config)

        tf.initialize_all_variables().run()

        # Create an empty array to hold [epoch number, loss]
        best_epoch = [0, 100000]

        print('finding best epoch parameter')
        # ====================================
        # Create vectors for training results
        # ====================================

        # Create empty vectors for loss
        train_loss_stats = np.array([])
        train_pos_loss_stats = np.array([])
        train_chunk_loss_stats = np.array([])
        # Create empty vectors for accuracy
        train_pos_stats = np.array([])
        train_chunk_stats = np.array([])

        # ====================================
        # Create vectors for validation results
        # ====================================
        # Create empty vectors for loss
        valid_loss_stats = np.array([])
        valid_pos_loss_stats = np.array([])
        valid_chunk_loss_stats = np.array([])
        # Create empty vectors for accuracy
        valid_pos_stats = np.array([])
        valid_chunk_stats = np.array([])

        for i in range(config.max_epoch):
            print("Epoch: %d" % (i + 1))
            mean_loss, posp_t, chunkp_t, post_t, chunkt_t, pos_loss, chunk_loss = \
                run_epoch(session, m,
                          words_t, pos_t, chunk_t,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, model_type=model_type)

            # Save stats for charts
            train_loss_stats = np.append(train_loss_stats, mean_loss)
            train_pos_loss_stats = np.append(train_pos_loss_stats, pos_loss)
            train_chunk_loss_stats = np.append(train_chunk_loss_stats, chunk_loss)

            # get predictions as list
            posp_t = reader.res_to_list(posp_t, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_t))
            chunkp_t = reader.res_to_list(chunkp_t, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_t))
            post_t = reader.res_to_list(post_t, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_t))
            chunkt_t = reader.res_to_list(chunkt_t, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_t))

            # find the accuracy
            pos_acc = np.sum(posp_t == post_t) / float(len(posp_t))
            chunk_acc = np.sum(chunkp_t == chunkt_t) / float(len(chunkp_t))

            # add to array
            train_pos_stats = np.append(train_pos_stats, pos_acc)
            train_chunk_stats = np.append(train_chunk_stats, chunk_acc)

            # print for tracking
            print("Pos Training Accuracy After Epoch %d :  %3f" % (i + 1, pos_acc))
            print("Chunk Training Accuracy After Epoch %d : %3f" % (i + 1, chunk_acc))

            valid_loss, posp_v, chunkp_v, post_v, chunkt_v, pos_v_loss, chunk_v_loss = \
                run_epoch(session, mvalid, words_v, pos_v, chunk_v,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, valid=True, model_type=model_type)

            # Save loss for charts
            valid_loss_stats = np.append(valid_loss_stats, valid_loss)
            valid_pos_loss_stats = np.append(valid_pos_loss_stats, pos_v_loss)
            valid_chunk_loss_stats = np.append(valid_chunk_loss_stats, chunk_v_loss)

            # get predictions as list

            posp_v = reader.res_to_list(posp_v, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_v))
            chunkp_v = reader.res_to_list(chunkp_v, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_v))
            chunkt_v = reader.res_to_list(chunkt_v, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_v))
            post_v = reader.res_to_list(post_v, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_v))

            # find accuracy
            pos_acc = np.sum(posp_v == post_v) / float(len(posp_v))
            chunk_acc = np.sum(chunkp_v == chunkt_v) / float(len(chunkp_v))

            print("Pos Validation Accuracy After Epoch %d :  %3f" % (i + 1, pos_acc))
            print("Chunk Validation Accuracy After Epoch %d : %3f" % (i + 1, chunk_acc))

            # add to stats
            valid_pos_stats = np.append(valid_pos_stats, pos_acc)
            valid_chunk_stats = np.append(valid_chunk_stats, chunk_acc)

            # update best parameters
            if (valid_loss < best_epoch[1]):
                best_epoch = [i + 1, valid_loss]

        # Save loss & accuracy plots
        np.savetxt(save_path + '/loss/valid_loss_stats.txt', valid_loss_stats)
        np.savetxt(save_path + '/loss/valid_pos_loss_stats.txt', valid_pos_loss_stats)
        np.savetxt(save_path + '/loss/valid_chunk_loss_stats.txt', valid_chunk_loss_stats)
        np.savetxt(save_path + '/accuracy/valid_pos_stats.txt', valid_pos_stats)
        np.savetxt(save_path + '/accuracy/valid_chunk_stats.txt', valid_chunk_stats)

        np.savetxt(save_path + '/loss/train_loss_stats.txt', train_loss_stats)
        np.savetxt(save_path + '/loss/train_pos_loss_stats.txt', train_pos_loss_stats)
        np.savetxt(save_path + '/loss/train_chunk_loss_stats.txt', train_chunk_loss_stats)
        np.savetxt(save_path + '/accuracy/train_pos_stats.txt', train_pos_stats)
        np.savetxt(save_path + '/accuracy/train_chunk_stats.txt', train_chunk_stats)

        # Train given epoch parameter
        print('Train Given Best Epoch Parameter :' + str(best_epoch[0]))
        for i in range(best_epoch[0]):
            print("Epoch: %d" % (i + 1))
            _, posp_c, chunkp_c, _, _, _, _ = \
                run_epoch(session, mTrain,
                          words_c, pos_c, chunk_c,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, model_type=model_type)

        print('Getting Testing Predictions')
        _, posp_test, chunkp_test, _, _, _, _ = \
            run_epoch(session, mTest,
                      words_test, pos_test, chunk_test,
                      config.num_pos_tags, config.num_chunk_tags,
                      verbose=True, valid=True, model_type=model_type)

        print('Writing Predictions')
        # prediction reshaping
        posp_c = reader.res_to_list(posp_c, config.batch_size, config.num_steps,
                                    pos_to_id, len(words_c))
        posp_test = reader.res_to_list(posp_test, config.batch_size, config.num_steps,
                                       pos_to_id, len(words_test))
        chunkp_c = reader.res_to_list(chunkp_c, config.batch_size,
                                      config.num_steps, chunk_to_id, len(words_c))
        chunkp_test = reader.res_to_list(chunkp_test, config.batch_size, config.num_steps,
                                         chunk_to_id, len(words_test))

        # save pickle - save_path + '/saved_variables.pkl'
        print('saving variables (pickling)')
        saveload.save(save_path + '/saved_variables.pkl', session)

        train_custom = reader.read_tokens(dataset_path + '/train.txt', 0)
        valid_custom = reader.read_tokens(dataset_path + '/validation.txt', 0)
        combined = reader.read_tokens(dataset_path + '/train_val_combined.txt', 0)
        test_data = reader.read_tokens(dataset_path + '/test.txt', 0)

        print('loaded text')

        chunk_pred_train = np.concatenate((np.transpose(train_custom), chunkp_t), axis=1)
        chunk_pred_val = np.concatenate((np.transpose(valid_custom), chunkp_v), axis=1)
        chunk_pred_c = np.concatenate((np.transpose(combined), chunkp_c), axis=1)
        chunk_pred_test = np.concatenate((np.transpose(test_data), chunkp_test), axis=1)
        pos_pred_train = np.concatenate((np.transpose(train_custom), posp_t), axis=1)
        pos_pred_val = np.concatenate((np.transpose(valid_custom), posp_v), axis=1)
        pos_pred_c = np.concatenate((np.transpose(combined), posp_c), axis=1)
        pos_pred_test = np.concatenate((np.transpose(test_data), posp_test), axis=1)

        print('finished concatenating, about to start saving')

        np.savetxt(save_path + '/predictions/chunk_pred_train.txt',
                   chunk_pred_train, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_train.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_val.txt',
                   chunk_pred_val, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_combined.txt',
                   chunk_pred_c, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_test.txt',
                   chunk_pred_test, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_train.txt',
                   pos_pred_train, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_val.txt',
                   pos_pred_val, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_combined.txt',
                   pos_pred_c, fmt='%s')
        np.savetxt(save_path + '/predictions/pos_pred_test.txt',
                   pos_pred_test, fmt='%s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type")
    parser.add_argument("--dataset_path")
    parser.add_argument("--save_path")
    args = parser.parse_args()
    if (str(args.model_type) != "POS") and (str(args.model_type) != "CHUNK"):
        args.model_type = 'JOINT'
    print('Model Selected : ' + str(args.model_type))
    main(str(args.model_type), str(args.dataset_path), str(args.save_path))
