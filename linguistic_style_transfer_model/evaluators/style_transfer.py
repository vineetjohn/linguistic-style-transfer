import argparse
import logging
import sys

import numpy as np
import tensorflow as tf

from linguistic_style_transfer_model.config import global_config, model_config
from linguistic_style_transfer_model.utils import data_processor

logger = logging.getLogger(global_config.logger_name)


def get_style_transfer_score(checkpoint_dir, text_sequences, label):
    x_test = np.asarray(text_sequences)
    y_test = np.asarray([label] * len(text_sequences))

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_conf = tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_processor.batch_iter(list(x_test), model_config.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        logger.info("Total number of test examples: {}".format(len(y_test)))
        logger.info("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
        # logger.info("F1-Score: {:g}".format(f1_score(y_true=y_test, y_pred=all_predictions)))
        # logger.info("Confusion matrix: {}".format(confusion_matrix(y_true=y_test, y_pred=all_predictions)))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args_namespace = parser.parse_args(argv[1:])
    command_line_args = vars(args_namespace)
    checkpoint_dir = command_line_args['checkpoint_dir']

    get_style_transfer_score(checkpoint_dir, None, None)


if __name__ == '__main__':
    main(sys.argv[1:])
