import logging

import gensim

from linguistic_style_transfer_model.config import global_config

logger = logging.getLogger(global_config.logger_name)


def add_word_vectors_to_embeddings(word_index, word_vector_path, encoder_embedding_matrix,
                                   decoder_embedding_matrix):
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
        if i >= global_config.vocab_size:
            break

    del wv_model

    return encoder_embedding_matrix, decoder_embedding_matrix
