import argparse
import sys

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")


def load_glove_model(glove_file):
    logger.info("Loading Glove Model")
    model = dict()
    with open(glove_file) as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
        logger.info("Done. {} words loaded!".format(len(model)))
    return model


def get_sentence_embedding(sentence, model):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(sentence)
    embeddings = np.asarray([model[token] for token in tokens if token in model])

    min_embedding = np.min(embeddings, axis=0)
    max_embedding = np.max(embeddings, axis=0)
    mean_embedding = np.mean(embeddings, axis=0)
    sentence_embedding = np.concatenate([min_embedding, max_embedding, mean_embedding], axis=0)

    return sentence_embedding


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file-path", type=str, required=True)
    parser.add_argument("--source-file-path", type=str, required=True)
    parser.add_argument("--target-file-path", type=str, required=True)

    options = vars(parser.parse_args(args=argv))
    glove_model = load_glove_model(options["embeddings_file_path"])

    cosine_distances = list()
    with \
            open(options["source_file_path"]) as source_file, \
            open(options["target_file_path"]) as target_file:
        for line_tuple in zip(source_file, target_file):
            line_1, line_2 = line_tuple
            cosine_distance = cosine(
                get_sentence_embedding(line_1, glove_model),
                get_sentence_embedding(line_2, glove_model))
            cosine_distances.append(cosine_distance)
    logger.info("Aggregate content preservation: {}".format(sum(cosine_distances) / len(cosine_distances)))


if __name__ == "__main__":
    main(sys.argv[1:])
