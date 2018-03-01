import numpy as np


def generate_word(word_embedding):
    return np.argmax(word_embedding)


def generate_sentence_from_indices(index_sequence, inverse_word_index):
    words = list(map(lambda x: inverse_word_index[x], index_sequence))
    return words


def generate_sentence_from_beam_indices(index_sequence, inverse_word_index):
    words = list(map(lambda x: inverse_word_index[x[np.random.randint(0, len(x))]], index_sequence))
    return words


def generate_sentence_from_logits(floating_index_sequence, inverse_word_index):
    word_indices = map(generate_word, floating_index_sequence)
    word_indices = list(filter(lambda x: x > 0, word_indices))
    words = list(map(lambda x: inverse_word_index[x], word_indices))
    return words
