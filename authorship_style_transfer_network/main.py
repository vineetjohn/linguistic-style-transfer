import sys
import random

import numpy as np
import tensorflow as tf

from authorship_style_transfer_network.models import adversarial_autoencoder
from authorship_style_transfer_network.utils import data_preprocessor
from authorship_style_transfer_network.utils import data_postprocessor
from authorship_style_transfer_network.utils import word_embedder
from authorship_style_transfer_network.utils import bleu_scorer

DEV_MODE = 1
EMBEDDING_SIZE = 300
MAX_SEQUENCE_LENGTH = 20
WORD_VECTOR_PATH = "word-embeddings/"


def main(argv):

    sess = get_tensorflow_session()

    if DEV_MODE:
        print("In dev mode")
        text_file_path = "data/c50-articles-dev.txt"
        label_file_path = "data/c50-labels-dev.txt"
        training_epochs = 3
        vocab_size = 1000
    else:
        text_file_path = "data/c50-articles.txt"
        label_file_path = "data/c50-labels.txt"
        training_epochs = 50
        vocab_size = 10000

    padded_sequences, text_sequence_lengths, word_index, integer_text_sequences = \
        data_preprocessor.get_text_sequences(
            text_file_path, vocab_size, MAX_SEQUENCE_LENGTH)
    print("text_sequence_lengths: {}".format(text_sequence_lengths.shape))
    print("padded_sequences: {}".format(padded_sequences.shape))

    sos_index = word_index['<sos>']
    eos_index = word_index['<eos>']
    data_size = padded_sequences.shape[0]

    one_hot_labels, num_labels = data_preprocessor.get_labels(label_file_path)
    print("one_hot_labels.shape: {}".format(one_hot_labels.shape))

    encoder_embedding_matrix = np.random.rand(vocab_size + 1, EMBEDDING_SIZE).astype('float32')
    decoder_embedding_matrix = np.random.rand(vocab_size + 1, EMBEDDING_SIZE).astype('float32')

    if not DEV_MODE:
        encoder_embedding_matrix, decoder_embedding_matrix = word_embedder.add_word_vectors_to_embeddings(
            word_index, WORD_VECTOR_PATH, encoder_embedding_matrix,
            decoder_embedding_matrix, vocab_size, EMBEDDING_SIZE)

    print("encoder_embedding_matrix: {}".format(encoder_embedding_matrix.shape))
    print("decoder_embedding_matrix: {}".format(decoder_embedding_matrix.shape))

    network = adversarial_autoencoder.AdversarialAutoencoder(
        num_labels, MAX_SEQUENCE_LENGTH, vocab_size, sos_index, eos_index,
        encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences,
        one_hot_labels, text_sequence_lengths)
    network.build_model()
    network.train(sess, data_size, training_epochs)

    inference_set_size = 1 * network.batch_size
    offset = random.randint(0, (data_size - 1) - inference_set_size)
    print("range: {}-{}".format(offset, (offset + inference_set_size)))

    actual_sequences = integer_text_sequences[offset: (offset + inference_set_size)]
    generated_sequences = network.infer(sess, offset, inference_set_size)

    inverse_word_index = {v: k for k, v in word_index.items()}

    actual_word_lists = list(map(
        lambda x: data_postprocessor.generate_sentence_from_indices(x, inverse_word_index),
        actual_sequences))
    generated_word_lists = list(map(
        lambda x: data_postprocessor.generate_sentence_from_logits(x, inverse_word_index),
        generated_sequences))
    # print(actual_word_lists)

    bleu_scores = bleu_scorer.get_corpus_bleu_scores(
        list(map(lambda x: [x], actual_word_lists)),
        generated_word_lists)
    print("bleu_scores: {}".format(bleu_scores))

    actual_sentences = list()
    generated_sentences = list()
    print(list(map(lambda x: len(list(x)), generated_word_lists)))
    for i in range(100):
        print(len(generated_word_lists[i]))
        actual_sentence = " ".join(actual_word_lists[i][1:])
        #     print("actual_sentence: {}".format(actual_sentence))
        actual_sentences.append(actual_sentence)

        generated_sentence = " ".join(generated_word_lists[i])
        print("generated_sentence: {}".format(generated_sentence))
        generated_sentences.append(generated_sentence)


def get_tensorflow_session():

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True

    return tf.Session(config=config_proto)


if __name__ == "__main__":
    main(sys.argv[1:])
