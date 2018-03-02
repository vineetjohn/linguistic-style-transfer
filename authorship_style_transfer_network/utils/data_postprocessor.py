import numpy as np


def generate_word(word_embedding):
    return np.argmax(word_embedding)


def generate_sentence_from_indices(index_sequence, inverse_word_index):
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
