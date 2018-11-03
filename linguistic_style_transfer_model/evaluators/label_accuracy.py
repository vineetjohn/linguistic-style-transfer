import sys

import argparse
import json
import logging
import os
from sklearn import metrics

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = logging.getLogger(global_config.logger_name)


def get_label_accuracy(predictions_file_path, gold_labels_file_path, saved_model_path):
    with open(os.path.join(saved_model_path,
                           global_config.label_to_index_dict_file), 'r') as json_file:
        label_to_index_map = json.load(json_file)

    gold_labels = list()
    prediction_labels = list()

    with open(gold_labels_file_path) as gold_labels_file:
        for text_label in gold_labels_file:
            gold_labels.append(label_to_index_map[text_label.strip()])

    with open(predictions_file_path) as predictions_file:
        for label in predictions_file:
            prediction_labels.append(int(label.strip()))

    accuracy = metrics.accuracy_score(y_true=gold_labels, y_pred=prediction_labels)
    logger.info("Classification Accuracy: {}".format(accuracy))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file-path", type=str, required=True)
    parser.add_argument("--gold-labels-file-path", type=str, required=True)
    parser.add_argument("--saved-model-path", type=str)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "DEBUG")

    options = parser.parse_args(args=argv)
    get_label_accuracy(options.predictions_file_path, options.gold_labels_file_path,
                                options.saved_model_path)


if __name__ == "__main__":
    main(sys.argv[1:])
