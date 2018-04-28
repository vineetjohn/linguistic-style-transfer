import logging
import pickle

import numpy as np
import tensorflow as tf
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords

from linguistic_style_transfer_model.config import global_config, model_config
from linguistic_style_transfer_model.utils import tsne_interface

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
            bow_representations, text_tokenizer, inverse_word_index, cleaned_word_index]


def get_test_sequences(text_file_path, word_index, text_tokenizer, inverse_word_index, cleaned_word_index):
    with open(text_file_path) as text_file:
        actual_sequences = text_tokenizer.texts_to_sequences(text_file)

    actual_word_lists = \
        [generate_words_from_indices(x, inverse_word_index)
         for x in actual_sequences]

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

    bow_representations = np.asarray([
        get_bow_representation(x, cleaned_word_index, inverse_word_index) for x in trimmed_sequences])

    return [actual_sequences, actual_word_lists, padded_sequences, text_sequence_lengths, bow_representations]


def get_labels(label_file_path):
    all_labels = list(open(label_file_path, "r").readlines())
    all_labels = [label.strip() for label in all_labels]

    labels = sorted(list(set(all_labels)))
    num_labels = len(labels)

    label_to_index_map = dict()
    index_to_label_map = dict()
    counter = 0
    for label in labels:
        label_to_index_map[label] = counter
        index_to_label_map[counter] = label
        counter += 1

    with open(global_config.index_to_label_dict_path, 'wb') as pickle_file:
        pickle.dump(index_to_label_map, pickle_file)
    with open(global_config.label_to_index_dict_path, 'wb') as pickle_file:
        pickle.dump(label_to_index_map, pickle_file)
    logger.info("labels: {}".format(label_to_index_map))

    one_hot_labels = list()
    label_sequences = list()
    for label in all_labels:
        label_sequences.append(label_to_index_map[label])

        one_hot_label = np.zeros(shape=num_labels, dtype=np.int32)
        one_hot_label[label_to_index_map[label]] = 1
        one_hot_labels.append(one_hot_label)

    return [label_sequences, np.asarray(one_hot_labels), num_labels]


def get_test_labels(label_file_path):
    all_labels = list(open(label_file_path, "r").readlines())
    all_labels = [label.strip() for label in all_labels]

    with open(global_config.label_to_index_dict_path, 'rb') as pickle_file:
        label_to_index_map = pickle.load(pickle_file)

    one_hot_labels = list()
    label_sequences = list()
    for label in all_labels:
        label_sequences.append(label_to_index_map[label])

        one_hot_label = np.zeros(shape=len(label_to_index_map), dtype=np.int32)
        one_hot_label[label_to_index_map[label]] = 1
        one_hot_labels.append(one_hot_label)

    return [label_sequences, one_hot_labels]


def generate_word(word_embedding):
    return np.argmax(word_embedding)


def generate_words_from_indices(index_sequence, inverse_word_index):
    words = [inverse_word_index[x] for x in index_sequence]
    return words


def generate_sentence_from_beam_indices(index_sequence, inverse_word_index):
    words = [inverse_word_index[x[0]] for x in index_sequence]
    return words


def generate_sentence_from_logits(floating_index_sequence, inverse_word_index):
    word_indices = [generate_word(x) for x in floating_index_sequence]
    word_indices = [x for x in word_indices if x > 0]
    words = [inverse_word_index[x] for x in word_indices]
    return words


def get_average_label_embeddings(data_size, label_sequences, dump_embeddings):
    with open(global_config.all_style_embeddings_path, 'rb') as pickle_file:
        all_style_embeddings = pickle.load(pickle_file)
    with open(global_config.all_content_embeddings_path, 'rb') as pickle_file:
        all_content_embeddings = pickle.load(pickle_file)

    style_embeddings = np.asarray(all_style_embeddings)
    content_embeddings = np.asarray(all_content_embeddings)

    style_embedding_map = dict()
    content_embedding_map = dict()

    for i in range(data_size - (data_size % model_config.batch_size)):
        label = label_sequences[i]

        if label not in style_embedding_map:
            style_embedding_map[label] = list()
        style_embedding_map[label].append(style_embeddings[i])

        if label not in content_embedding_map:
            content_embedding_map[label] = list()
        content_embedding_map[label].append(content_embeddings[i])

    if dump_embeddings:
        tsne_interface.generate_plot_coordinates(
            style_embedding_map, global_config.style_coordinates_path)
        tsne_interface.generate_plot_coordinates(
            content_embedding_map, global_config.content_coordinates_path)

    with open(global_config.label_mapped_style_embeddings_path, 'wb') as pickle_file:
        pickle.dump(style_embedding_map, pickle_file)
    logger.debug("Pickled label mapped style embeddings")

    average_label_embeddings = dict()
    for label in style_embedding_map:
        average_label_embeddings[label] = np.median(style_embedding_map[label], axis=0)

    return average_label_embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
