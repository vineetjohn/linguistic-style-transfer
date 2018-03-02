import tensorflow as tf
import numpy as np

MAX_SEQUENCE_LENGTH = 20


def get_text_sequences(text_file_path, vocab_size):

    word_index = {
        'pad': 0,
        'sos': 1,
        'eos': 2,
        'unk': 3
    }

    text_tokenizer = tf.keras.preprocessing.text.Tokenizer()

    with open(text_file_path) as text_file:
        text_tokenizer.fit_on_texts(text_file)
    available_vocab = len(text_tokenizer.word_index)
    print("available_vocab: {}".format(available_vocab))

    num_predefined_tokens = len(word_index)
    for index, word in enumerate(text_tokenizer.word_index):
        word_index[word] = index + num_predefined_tokens
    text_tokenizer.word_index = word_index

    with open(text_file_path) as text_file:
        integer_text_sequences = text_tokenizer.texts_to_sequences(text_file)

    text_sequence_lengths = np.asarray(
        a=[len(x) for x in integer_text_sequences], dtype=np.int32)

    max_sequence_length = int(np.median(text_sequence_lengths))
    print("max_sequence_length: ", max_sequence_length)
    
    integer_text_sequences = [
        [x if x < vocab_size else word_index['unk'] for x in sequence]
        for sequence in integer_text_sequences]
    integer_text_sequences = [x + [word_index['eos']] for x in integer_text_sequences]

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        integer_text_sequences, maxlen=max_sequence_length, padding='post', truncating='post',
        value=word_index['pad'])

    text_sequence_lengths = np.asarray(
        [max_sequence_length if x > max_sequence_length else x for x in text_sequence_lengths])

    return [padded_sequences, text_sequence_lengths, text_tokenizer.word_index,
            integer_text_sequences, max_sequence_length]


def get_labels(label_file_path):

    label_tokenizer = tf.keras.preprocessing.text.Tokenizer()

    with open(label_file_path) as label_file:
        label_tokenizer.fit_on_texts(label_file)

    with open(label_file_path) as label_file:
        label_sequences = label_tokenizer.texts_to_sequences(label_file)

    num_labels = len(label_tokenizer.word_index)
    one_hot_labels = np.asarray(
        [np.eye(num_labels, k=x[0])[0] for x in label_sequences])

    return [one_hot_labels, num_labels, label_sequences]
