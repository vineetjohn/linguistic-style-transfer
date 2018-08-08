import sys

import argparse
import logging
import numpy as np
import statistics
import tensorflow as tf
from scipy.spatial.distance import cosine

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer, lexicon_helper

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
    sentiment_words = lexicon_helper.get_sentiment_words()
    cosine_distances = list()
    skip_count = 0
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        cosine_similarity = 0
        words_1 = set(word_list_1)
        words_2 = set(word_list_2)

        words_1 -= sentiment_words
        words_2 -= sentiment_words
        try:
            cosine_similarity = 1 - cosine(
                get_sentence_embedding(words_1, embedding_model),
                get_sentence_embedding(words_2, embedding_model))
            cosine_distances.append(cosine_similarity)
        except ValueError:
            skip_count += 1
            logger.debug("Skipped lines: {} :-: {}".format(word_list_1, word_list_2))

    logger.debug("{} lines skipped due to errors".format(skip_count))
    mean_cosine_distance = statistics.mean(cosine_distances) if cosine_distances else 0

    del sentiment_words

    return mean_cosine_distance


def get_word_overlap_score(actual_word_lists, generated_word_lists):
    english_stopwords = lexicon_helper.get_stopwords()
    sentiment_words = lexicon_helper.get_sentiment_words()

    scores = list()
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        score = 0
        words_1 = set(word_list_1)
        words_2 = set(word_list_2)

        words_1 -= sentiment_words
        words_1 -= english_stopwords
        words_2 -= sentiment_words
        words_2 -= english_stopwords

        word_intersection = words_1 & words_2
        word_union = words_1 | words_2
        if word_union:
            score = len(word_intersection) / len(word_union)
            scores.append(score)

    word_overlap_score = statistics.mean(scores) if scores else 0

    del english_stopwords
    del sentiment_words

    return word_overlap_score


def run_content_preservation_evaluator(source_file_path, target_file_path, embeddings_file):
    glove_model = load_glove_model(embeddings_file)
    actual_word_lists, generated_word_lists = list(), list()
    with open(source_file_path) as source_file, open(target_file_path) as target_file:
        for line_1, line_2 in zip(source_file, target_file):
            actual_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_1))
            generated_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_2))

    content_preservation_score = get_content_preservation_score(
        actual_word_lists, generated_word_lists, glove_model)
    word_overlap_score = get_word_overlap_score(
        actual_word_lists, generated_word_lists)

    return [content_preservation_score, word_overlap_score]


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file-path", type=str, required=True)
    parser.add_argument("--source-file-path", type=str, required=True)
    parser.add_argument("--target-file-path", type=str, required=True)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "DEBUG")

    options = vars(parser.parse_args(args=argv))
    [content_preservation_score, word_overlap_score] = run_content_preservation_evaluator(
        options["source_file_path"], options["target_file_path"], options["embeddings_file_path"])

    logger.info("Aggregate content preservation: {}".format(content_preservation_score))
    logger.info("Aggregate word overlap: {}".format(word_overlap_score))


if __name__ == "__main__":
    main(sys.argv[1:])
