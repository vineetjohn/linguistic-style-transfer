import json
import logging
import numpy as np
import os
import pickle
import tensorflow as tf

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import tsne_interface, lexicon_helper

logger = logging.getLogger(global_config.logger_name)

label_to_index_map = dict()
index_to_label_map = dict()
bow_filtered_vocab_indices = dict()


def populate_word_blacklist(word_index):
    blacklisted_words = set()
    blacklisted_words |= set(global_config.predefined_word_index.values())
    if global_config.filter_sentiment_words:
        blacklisted_words |= lexicon_helper.get_sentiment_words()
    if global_config.filter_stopwords:
        blacklisted_words |= lexicon_helper.get_stopwords()

    global bow_filtered_vocab_indices
    allowed_vocab = word_index.keys() - blacklisted_words
    i = 0
    for word in allowed_vocab:
        vocab_index = word_index[word]
        bow_filtered_vocab_indices[vocab_index] = i
        i += 1

    global_config.bow_size = len(allowed_vocab)
    logger.info("Created word index blacklist for BoW")
    logger.info("BoW size: {}".format(global_config.bow_size))


def get_text_sequences(text_file_path, vocab_size, vocab_save_path):
    word_index = global_config.predefined_word_index
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=global_config.vocab_size, filters=global_config.tokenizer_filters)

    with open(text_file_path) as text_file:
        text_tokenizer.fit_on_texts(text_file)
    available_vocab = len(text_tokenizer.word_index)
    logger.info("available_vocab: {}".format(available_vocab))

    num_predefined_tokens = len(word_index)
    for index, word in enumerate(text_tokenizer.word_index):
        new_index = index + num_predefined_tokens
        if new_index == vocab_size:
            break
        word_index[word] = new_index

    populate_word_blacklist(word_index)
    text_tokenizer.word_index = word_index

    with open(text_file_path) as text_file:
        actual_sequences = text_tokenizer.texts_to_sequences(text_file)

    text_sequence_lengths = np.asarray(
        a=[len(x) for x in actual_sequences], dtype=np.int32)

    global_config.vocab_size = len(word_index)
    trimmed_sequences = [
        [x if x < vocab_size else word_index[global_config.unk_token] for x in sequence]
        for sequence in actual_sequences]
    inverse_word_index = {v: k for k, v in word_index.items()}

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        trimmed_sequences, maxlen=global_config.max_sequence_length, padding='post',
        truncating='post', value=word_index[global_config.eos_token])

    text_sequence_lengths = np.asarray(
        [global_config.max_sequence_length if x >= global_config.max_sequence_length
         else x + 1 for x in text_sequence_lengths])  # x + 1 to accomodate a single EOS token

    with open(vocab_save_path, 'w') as json_file:
        json.dump(word_index, json_file)

    return [word_index, padded_sequences, text_sequence_lengths, text_tokenizer, inverse_word_index]


def get_test_sequences(text_file_path, text_tokenizer, word_index, inverse_word_index):
    if not bow_filtered_vocab_indices:
        populate_word_blacklist(word_index)

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

    return [actual_sequences, actual_word_lists, padded_sequences, text_sequence_lengths]


def get_labels(label_file_path, store_labels, store_path):
    all_labels = list(open(label_file_path, "r").readlines())
    all_labels = [label.strip() for label in all_labels]

    labels = sorted(list(set(all_labels)))
    num_labels = len(labels)

    counter = 0
    for label in labels:
        label_to_index_map[label] = counter
        index_to_label_map[counter] = label
        counter += 1

    if store_labels:
        with open(os.path.join(store_path, global_config.index_to_label_dict_file), 'w') as file:
            json.dump(index_to_label_map, file)
        with open(os.path.join(store_path, global_config.label_to_index_dict_file), 'w') as file:
            json.dump(label_to_index_map, file)
    logger.info("labels: {}".format(label_to_index_map))

    one_hot_labels = list()
    for label in all_labels:
        one_hot_label = np.zeros(shape=num_labels, dtype=np.int32)
        one_hot_label[label_to_index_map[label]] = 1
        one_hot_labels.append(one_hot_label)

    return [np.asarray(one_hot_labels), num_labels]


def get_test_labels(label_file_path, model_save_directory):
    all_labels = list(open(label_file_path, "r").readlines())
    all_labels = [label.strip() for label in all_labels]

    with open(os.path.join(model_save_directory,
                           global_config.label_to_index_dict_file), 'r') as json_file:
        global label_to_index_map
        label_to_index_map = json.load(json_file)

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


def get_average_label_embeddings(data_size, dump_embeddings, epoch):
    style_embeddings = np.load(file=global_config.all_style_embeddings_path)
    content_embeddings = np.load(file=global_config.all_content_embeddings_path)
    with open(global_config.all_shuffled_labels_path, 'rb') as pickle_file:
        all_one_hot_labels = pickle.load(pickle_file)

    style_embedding_map = dict()
    content_embedding_map = dict()

    for i in range(data_size):
        label = all_one_hot_labels[i].tolist().index(1)

        if label not in style_embedding_map:
            style_embedding_map[label] = list()
        if label not in content_embedding_map:
            content_embedding_map[label] = list()

        style_embedding_map[label].append(style_embeddings[i])
        content_embedding_map[label].append(content_embeddings[i])

    if dump_embeddings:
        if not os.path.exists(global_config.tsne_plot_folder):
            os.makedirs(global_config.tsne_plot_folder)

        style_plot_path = \
            global_config.tsne_plot_folder + \
            global_config.style_embedding_plot_file.format(epoch)
        tsne_interface.generate_plot_coordinates(
            style_embedding_map, global_config.style_coordinates_path,
            index_to_label_map, style_plot_path, len(index_to_label_map) * epoch + 0)

        content_plot_path = \
            global_config.tsne_plot_folder + \
            global_config.content_embedding_plot_file.format(epoch)
        tsne_interface.generate_plot_coordinates(
            content_embedding_map, global_config.content_coordinates_path,
            index_to_label_map, content_plot_path, len(index_to_label_map) * epoch + 1)

    with open(global_config.label_mapped_style_embeddings_path, 'wb') as pickle_file:
        pickle.dump(style_embedding_map, pickle_file)
    logger.debug("Pickled label mapped style embeddings")

    average_label_embeddings = dict()
    for label in style_embedding_map:
        average_label_embeddings[label] = np.mean(style_embedding_map[label], axis=0)

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


def get_bow_representations(text_sequences):
    bow_representation = list()
    for text_sequence in text_sequences:
        sequence_bow_representation = np.zeros(shape=global_config.bow_size, dtype=np.float32)
        for index in text_sequence:
            if index in bow_filtered_vocab_indices:
                bow_index = bow_filtered_vocab_indices[index]
                sequence_bow_representation[bow_index] += 1
        sequence_bow_representation /= np.max([np.sum(sequence_bow_representation), 1])
        bow_representation.append(sequence_bow_representation)

    return np.asarray(bow_representation)
