import sys

import argparse
import json
import logging
import os
import statistics
from types import SimpleNamespace

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.evaluators import \
    style_transfer, content_preservation, language_fluency
from linguistic_style_transfer_model.utils import log_initializer

logger = logging.getLogger(global_config.logger_name)


class Options(SimpleNamespace):

    def __init__(self):
        self.classifier_model_path = None
        self.training_path = None
        self.inference_path = None
        self.embeddings_path = None
        self.language_model_path = None


def main(argv):
    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-model-path", type=str, required=True)
    parser.add_argument("--training-path", type=str, required=True)
    parser.add_argument("--inference-path", type=str, required=True)
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--language-model-path", type=str, required=True)
    parser.parse_known_args(args=argv, namespace=options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")
    logger.info(options)

    index_label_file_path = os.path.join(options.training_path, global_config.index_to_label_dict_file)
    with open(index_label_file_path, 'r') as index_label_file:
        index_label_dict = json.load(index_label_file)

    style_transfer_scores = list()
    content_preservation_scores = list()
    word_overlap_scores = list()
    ll_scores = list()

    for label_index in index_label_dict:
        actual_text_file_path = os.path.join(
            options.inference_path, "actual_sentences_{}.txt".format(label_index))
        generated_text_file_path = os.path.join(
            options.inference_path, "generated_sentences_{}.txt".format(label_index))

        [style_transfer_score, _] = style_transfer.get_style_transfer_score(
            options.classifier_model_path, generated_text_file_path, str(label_index), None)
        [content_preservation_score, word_overlap_score] = \
            content_preservation.run_content_preservation_evaluator(
                actual_text_file_path, generated_text_file_path, options.embeddings_path)
        ll_score = language_fluency.score_generated_sentences(
            generated_text_file_path, options.language_model_path)

        style_transfer_scores.append(style_transfer_score)
        content_preservation_scores.append(content_preservation_score)
        word_overlap_scores.append(word_overlap_score)
        ll_scores.append(ll_score)

    logger.info("style_transfer_scores: {}".format(style_transfer_scores))
    logger.info("content_preservation_scores: {}".format(content_preservation_scores))
    logger.info("word_overlap_scores: {}".format(word_overlap_scores))
    logger.info("ll_scores: {}".format(ll_scores))

    logger.info("transfer-strength: {}".format(statistics.mean(style_transfer_scores)))
    logger.info("content-preservation: {}".format(statistics.mean(content_preservation_scores)))
    logger.info("word-overlap: {}".format(statistics.mean(word_overlap_scores)))
    logger.info("log-likelihood: {}".format(statistics.mean(ll_scores)))


if __name__ == '__main__':
    main(sys.argv[1:])
