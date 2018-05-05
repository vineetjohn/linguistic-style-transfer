import logging

import gensim

from linguistic_style_transfer_model.config import global_config

logger = logging.getLogger(global_config.logger_name)


def add_word_vectors_to_embeddings(word_index, encoder_embedding_matrix, decoder_embedding_matrix,
                                   embedding_model_path):
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
        embedding_model_path, binary=True, unicode_errors='ignore')
    logger.info("Embeddings loaded into memory")

    i = 0
    for word in word_index:
        try:
            word_embedding = embedding_model[word]
            encoder_embedding_matrix[i] = word_embedding
            decoder_embedding_matrix[i] = word_embedding
        except KeyError:
            pass

        i += 1
        if i >= global_config.vocab_size:
            break

    del embedding_model

    return encoder_embedding_matrix, decoder_embedding_matrix
