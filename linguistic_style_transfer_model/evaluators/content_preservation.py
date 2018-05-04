import argparse
import logging
import sys

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = logging.getLogger(global_config.logger_name)


def load_glove_model(glove_file):
    logger.debug("Loading Glove Model")
    model = dict()
    with open(glove_file) as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
        logger.debug("Done. {} words loaded!".format(len(model)))
    return model


def get_sentence_embedding(tokens, model):
    embeddings = np.asarray([model[token] for token in tokens if token in model])

    min_embedding = np.min(embeddings, axis=0)
    max_embedding = np.max(embeddings, axis=0)
    mean_embedding = np.mean(embeddings, axis=0)
    sentence_embedding = np.concatenate([min_embedding, max_embedding, mean_embedding], axis=0)

    return sentence_embedding


def get_content_preservation_score(actual_word_lists, generated_word_lists, embedding_model):
    cosine_distances = list()
    skip_count = 0
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        try:
            cosine_distance = 1 - cosine(
                get_sentence_embedding(word_list_1, embedding_model),
                get_sentence_embedding(word_list_2, embedding_model))
            cosine_distances.append(cosine_distance)
        except ValueError:
            skip_count += 1
            logger.debug("Skipped lines: {} :-: {}".format(word_list_1, word_list_2))

    logger.debug("{} lines skipped due to errors".format(skip_count))
    mean_cosine_distance = np.mean(np.asarray(cosine_distances), axis=0)

    return mean_cosine_distance


def run_content_preservation_evaluator(source_file, target_file, embeddings_file):
    glove_model = load_glove_model(embeddings_file)
    actual_word_lists, generated_word_lists = list(), list()
    with open(source_file) as source_file, open(target_file) as target_file:
        for line_1, line_2 in zip(source_file, target_file):
            actual_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_1))
            generated_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_2))

    content_preservation_score = get_content_preservation_score(
        actual_word_lists, generated_word_lists, glove_model)
    logger.info("Aggregate content preservation: {}".format(content_preservation_score))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file-path", type=str, required=True)
    parser.add_argument("--source-file-path", type=str, required=True)
    parser.add_argument("--target-file-path", type=str, required=True)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "DEBUG")

    options = vars(parser.parse_args(args=argv))
    run_content_preservation_evaluator(
        options["source_file_path"], options["target_file_path"], options["embeddings_file_path"])


if __name__ == "__main__":
    main(sys.argv[1:])
