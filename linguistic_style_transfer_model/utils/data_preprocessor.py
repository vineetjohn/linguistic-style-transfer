import logging
import pickle

import numpy as np
import tensorflow as tf
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords

from linguistic_style_transfer_model.config import global_config

logger = logging.getLogger(global_config.logger_name)


def get_cleaned_word_index(word_index):
    english_stopwords = set(stopwords.words('english'))

    def is_sentiment_word(word_to_test):
        synsetlist = list(swn.senti_synsets(word_to_test))
        return not synsetlist or synsetlist[0].pos_score() != synsetlist[0].neg_score()

    cleaned_word_index = dict()
    index = 0
    for word in word_index:
        if word_index[word] > 2 and word not in english_stopwords and not is_sentiment_word(word):
            cleaned_word_index[word] = index
            index += 1

    del english_stopwords

    logger.debug("cleaned_word_index: {}".format(cleaned_word_index))
    global_config.bow_size = len(cleaned_word_index)

    return cleaned_word_index


def get_bow_representation(index_sequence, cleaned_word_index, inverse_word_index):
    bow_representation = np.zeros(shape=len(cleaned_word_index), dtype=np.int32)

    for index in index_sequence:
        if inverse_word_index[index] in cleaned_word_index:
            bow_representation[cleaned_word_index[inverse_word_index[index]]] = 1

    # bow_representation = np.divide(bow_representation, len(index_sequence))

    return bow_representation


def get_text_sequences(text_file_path, vocab_size):
    word_index = global_config.predefined_word_index
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer()

    with open(text_file_path) as text_file:
        text_tokenizer.fit_on_texts(text_file)
    available_vocab = len(text_tokenizer.word_index)
    logger.info("available_vocab: {}".format(available_vocab))

    num_predefined_tokens = len(word_index)
    for index, word in enumerate(text_tokenizer.word_index):
        word_index[word] = index + num_predefined_tokens
    text_tokenizer.word_index = word_index

    with open(text_file_path) as text_file:
        actual_sequences = text_tokenizer.texts_to_sequences(text_file)

    text_sequence_lengths = np.asarray(
        a=[len(x) for x in actual_sequences], dtype=np.int32)

    global_config.vocab_size = vocab_size if len(word_index) > vocab_size else len(word_index)
    trimmed_sequences = [
        [x if x < vocab_size else word_index[global_config.unk_token] for x in sequence]
        for sequence in actual_sequences]
    inverse_word_index = {v: k for k, v in word_index.items()}

    cleaned_word_index = get_cleaned_word_index(word_index)
    bow_representations = np.asarray([
        get_bow_representation(x, cleaned_word_index, inverse_word_index) for x in trimmed_sequences])

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        trimmed_sequences, maxlen=global_config.max_sequence_length, padding='post',
        truncating='post', value=word_index[global_config.eos_token])

    text_sequence_lengths = np.asarray(
        [global_config.max_sequence_length if x >= global_config.max_sequence_length
         else x + 1 for x in text_sequence_lengths])  # x + 1 to accomodate a single EOS token

    return [word_index, actual_sequences, padded_sequences, text_sequence_lengths,
            bow_representations, text_tokenizer, inverse_word_index]


def get_test_sequences(text_file_path, word_index, text_tokenizer):
    with open(text_file_path) as text_file:
        actual_sequences = text_tokenizer.texts_to_sequences(text_file)

    trimmed_sequences = [
        [x if x < global_config.vocab_size else word_index[global_config.unk_token] for x in sequence]
        for sequence in actual_sequences]

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        trimmed_sequences, maxlen=global_config.max_sequence_length, padding='post',
        truncating='post', value=word_index[global_config.eos_token])

    text_sequence_lengths = np.asarray(
        a=[len(x) for x in actual_sequences], dtype=np.int32)

    text_sequence_lengths = np.asarray(
        [global_config.max_sequence_length if x >= global_config.max_sequence_length
         else x + 1 for x in text_sequence_lengths])  # x + 1 to accomodate a single EOS token

    return [actual_sequences, padded_sequences, text_sequence_lengths]


def get_labels(label_file_path):
    label_tokenizer = tf.keras.preprocessing.text.Tokenizer()

    with open(label_file_path) as label_file:
        label_tokenizer.fit_on_texts(label_file)

    with open(label_file_path) as label_file:
        label_sequences = label_tokenizer.texts_to_sequences(label_file)

    label_map = {v: k for k, v in label_tokenizer.word_index.items()}
    with open(global_config.label_names_path, 'wb') as pickle_file:
        pickle.dump(label_map, pickle_file)
    logger.info("labels: {}".format(label_map))

    num_labels = len(label_tokenizer.word_index)
    one_hot_labels = np.asarray(
        [np.eye(num_labels, k=(x[0] - 1))[0] for x in label_sequences])

    return [label_sequences, one_hot_labels, num_labels]
