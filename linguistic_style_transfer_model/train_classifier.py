import argparse
import datetime
import logging
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf

from linguistic_style_transfer_model.config import model_config, global_config
from linguistic_style_transfer_model.models.text_classifier import TextCNN
from linguistic_style_transfer_model.utils import data_processor, log_initializer

logger = None


def train_classifier_model(options):
    # Load data
    logger.info("Loading data...")

    [word_index, _, x, _, _, text_tokenizer, _, _] = \
        data_processor.get_text_sequences(options['text_file_path'], options['vocab_size'])

    with open(global_config.classifier_text_tokenizer_path, 'wb') as pickle_file:
        pickle.dump(word_index, pickle_file)

    x = np.asarray(x)

    [_, y, _] = data_processor.get_labels(options['label_file_path'])

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    logger.info("Vocabulary Size: {:d}".format(global_config.vocab_size))
    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    sess = get_tensorflow_session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=options['vocab_size'],
            embedding_size=128,
            filter_sizes=list(map(int, [3, 4, 5])),
            num_filters=128,
            l2_reg_lambda=0.0)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # Write vocabulary
        with open(global_config.classifier_vocab_save_path, 'wb') as pickle_file:
            pickle.dump(word_index, pickle_file)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_processor.batch_iter(
            list(zip(x_train, y_train)), model_config.batch_size, options['training_epochs'])
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                logger.info("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                logger.info("")
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--label-file-path", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--training-epochs", type=int, default=10)
    parser.add_argument("--logging-level", type=str, default="INFO")

    options = vars(parser.parse_args(args=argv))
    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options['logging_level'])

    train_classifier_model(options)

    logger.info("Training Complete!")


def get_tensorflow_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    config_proto = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True,
        gpu_options=gpu_options)

    return tf.Session(config=config_proto)


if __name__ == "__main__":
    main(sys.argv[1:])
