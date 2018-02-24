import tensorflow as tf
import numpy as np

def get_text_sequences(text_file_path, vocab_size, max_sequence_length):

    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')

    with open(text_file_path) as text_file:
        text_tokenizer.fit_on_texts(text_file)

    with open(text_file_path) as text_file:
        integer_text_sequences = text_tokenizer.texts_to_sequences(text_file)

    text_sequence_lengths = np.asarray(
        a=list(map(lambda x: len(x), integer_text_sequences)), dtype=np.int32)

    # MAX_SEQUENCE_LENGTH = np.amax(text_sequence_lengths)

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        integer_text_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    return padded_sequences, text_sequence_lengths, text_tokenizer.word_index, integer_text_sequences


def get_labels(label_file_path):

    label_tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False)

    with open(label_file_path) as label_file:
        label_tokenizer.fit_on_texts(label_file)

    with open(label_file_path) as label_file:
        label_sequences = label_tokenizer.texts_to_sequences(label_file)

    num_labels = len(label_tokenizer.word_index)
    one_hot_labels = np.asarray(list(
        map(lambda x: np.eye(num_labels, k=x[0])[0], label_sequences)))

    return one_hot_labels, num_labels
