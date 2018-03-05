import logging

import gensim

from authorship_style_transfer_network.utils import global_constants

logger = logging.getLogger(global_constants.LOGGER_NAME)


def add_word_vectors_to_embeddings(word_index, word_vector_path, encoder_embedding_matrix,
                                   decoder_embedding_matrix, vocab_size):
    wv_model_path = word_vector_path + "GoogleNews-vectors-negative300.bin.gz"
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(
        wv_model_path, binary=True, unicode_errors='ignore')
    logger.info("Embeddings loaded into memory")

    i = 0
    for word in word_index:
        try:
            word_embedding = wv_model[word]
            encoder_embedding_matrix[i] = word_embedding
            decoder_embedding_matrix[i] = word_embedding
        except KeyError:
            pass

        i += 1
        if i >= vocab_size:
            break

    del wv_model

    return encoder_embedding_matrix, decoder_embedding_matrix
