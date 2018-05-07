import argparse
import json
import os
import sys

from matplotlib import pyplot as plt

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']


def plot_scores(epochs, style_transfer_scores, content_preservation_scores, saved_model_path):
    plt.figure(0)
    for i in range(len(epochs)):
        plt.plot(epochs, style_transfer_scores, 'ro-')
    plt.savefig(fname=saved_model_path + "/validation_style_transfer.svg", format="svg", dpi=1200)
    plt.show()

    plt.figure(1)
    for i in range(len(epochs)):
        plt.plot(epochs, content_preservation_scores, 'bo-')
    plt.savefig(fname=saved_model_path + "/validation_content_preservation.svg", format="svg", dpi=1200)
    plt.show()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-model-path", type=str)

    args = vars(parser.parse_args(args=argv))

    logger.info(args)
    with open(os.path.join(args["saved_model_path"],
                           global_config.validation_scores_file)) as validation_scores_file:
        validation_score_json_lines = validation_scores_file.readlines()
        epochs, style_transfer_scores, content_preservation_scores = list(), list(), list()
        for validation_score_json in validation_score_json_lines:
            validation_score_json = validation_score_json.strip()
            validation_scores = json.loads(validation_score_json)
            epochs.append(validation_scores['epoch'])
            style_transfer_scores.append(validation_scores['style-transfer'])
            content_preservation_scores.append(validation_scores['content-preservation'])

        logger.info(epochs)
        logger.info(style_transfer_scores)
        logger.info(content_preservation_scores)

        plot_scores(epochs, style_transfer_scores, content_preservation_scores, args["saved_model_path"])


if __name__ == "__main__":
    main(sys.argv[1:])
