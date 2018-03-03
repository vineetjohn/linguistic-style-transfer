import argparse
import random
import sys
from datetime import datetime as dt

import numpy as np
import tensorflow as tf

from authorship_style_transfer_network.utils import log
from authorship_style_transfer_network.models import adversarial_autoencoder
from authorship_style_transfer_network.utils import bleu_scorer
from authorship_style_transfer_network.utils import data_postprocessor
from authorship_style_transfer_network.utils import data_preprocessor
from authorship_style_transfer_network.utils import word_embedder

EMBEDDING_SIZE = 300
WORD_VECTOR_PATH = "word-embeddings/"
logger = None


def get_data(text_file_path, vocab_size, label_file_path, use_pretrained_embeddings):

    padded_sequences, text_sequence_lengths, word_index, \
        integer_text_sequences, max_sequence_length = \
        data_preprocessor.get_text_sequences(text_file_path, vocab_size)
    logger.debug("text_sequence_lengths: {}".format(text_sequence_lengths.shape))
    logger.debug("padded_sequences: {}".format(padded_sequences.shape))

    sos_index = word_index['sos']
    eos_index = word_index['eos']
    data_size = padded_sequences.shape[0]

    one_hot_labels, num_labels, label_sequences = data_preprocessor.get_labels(label_file_path)
    logger.debug("one_hot_labels.shape: {}".format(one_hot_labels.shape))

    encoder_embedding_matrix = np.random.uniform(
        low=-0.05, high=0.05, size=(vocab_size, EMBEDDING_SIZE)).astype(dtype=np.float32)
    decoder_embedding_matrix = np.random.uniform(
        low=-0.05, high=0.05, size=(vocab_size, EMBEDDING_SIZE)).astype(dtype=np.float32)
    logger.debug("encoder_embedding_matrix: {}".format(encoder_embedding_matrix.shape))
    logger.debug("decoder_embedding_matrix: {}".format(decoder_embedding_matrix.shape))

    if use_pretrained_embeddings:
        logger.info("Loading pretrained embeddings")
        encoder_embedding_matrix, decoder_embedding_matrix = word_embedder.add_word_vectors_to_embeddings(
            word_index, WORD_VECTOR_PATH, encoder_embedding_matrix,
            decoder_embedding_matrix, vocab_size)

    return num_labels, max_sequence_length, vocab_size, sos_index, eos_index, \
        encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences, \
        one_hot_labels, text_sequence_lengths, label_sequences, encoder_embedding_matrix, \
        decoder_embedding_matrix, data_size, word_index, integer_text_sequences


def execute_post_training_operations(all_style_representations, data_size, batch_size, label_sequences):
    # Extract style embeddings
    style_embeddings = np.asarray(all_style_representations)
    logger.info("style_embeddings_shape: {}".format(style_embeddings.shape))

    all_author_embeddings = dict()
    for i in range(data_size - (data_size % batch_size)):
        author_label = label_sequences[i][0]
        if author_label not in all_author_embeddings:
            all_author_embeddings[author_label] = list()
        all_author_embeddings[author_label].append(style_embeddings[i])

    average_author_embeddings = dict()
    for author_label in all_author_embeddings:
        average_author_embeddings[author_label] = np.mean(all_author_embeddings[author_label], axis=0)
    # logger.info("average_author_embeddings: {}".format(average_author_embeddings))


def execute_post_inference_operations(word_index, integer_text_sequences, start_index, final_index,
                                      generated_sequences):

    inverse_word_index = {v: k for k, v in word_index.items()}
    actual_sequences = integer_text_sequences[start_index:final_index]
    actual_word_lists = \
        [data_postprocessor.generate_sentence_from_indices(x, inverse_word_index)
         for x in actual_sequences]
    generated_word_lists = \
        [data_postprocessor.generate_sentence_from_indices(x, inverse_word_index)
         for x in generated_sequences]

    # Evaluate model scores
    bleu_scores = bleu_scorer.get_corpus_bleu_scores(
        [[x] for x in actual_word_lists], generated_word_lists)
    logger.info("bleu_scores: {}".format(bleu_scores))

    logger.debug("Minimum generated sentence length: {}"
                 .format(min(len(x) for x in generated_word_lists)))

    actual_sentences = [" ".join(x) for x in actual_word_lists]
    generated_sentences = [" ".join(x) for x in generated_word_lists]

    for i in range(3):
        logger.debug("actual_sentence: {}".format(actual_sentences[i]))
        logger.debug("generated_sentence: {}".format(generated_sentences[i]))

    timestamped_file_suffix = dt.now().strftime("%Y%m%d%H%M%S")
    output_file_path = "output/actual_sentences_{}.txt".format(timestamped_file_suffix)
    with open(output_file_path, 'w') as output_file:
        for sentence in actual_sentences:
            output_file.write(sentence + "\n")

    output_file_path = "output/generated_sentences_{}.txt".format(timestamped_file_suffix)
    with open(output_file_path, 'w') as output_file:
        for sentence in generated_sentences:
            output_file.write(sentence + "\n")


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-mode", action="store_true", default=True)
    parser.add_argument("--use-pretrained-embeddings", action="store_true", default=False)
    parser.add_argument("--training-epochs", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--logging-level", type=str, default="INFO")

    args_namespace = parser.parse_args(argv)
    command_line_args = vars(args_namespace)

    global logger
    logger = log.setup_custom_logger('root', command_line_args['logging_level'])

    if command_line_args['dev_mode']:
        logger.info("Running in dev mode")
        text_file_path = "data/c50-articles-dev.txt"
        label_file_path = "data/c50-labels-dev.txt"
    else:
        text_file_path = "data/c50-articles.txt"
        label_file_path = "data/c50-labels.txt"

    # Retrieve all data
    num_labels, max_sequence_length, vocab_size, sos_index, eos_index, \
        encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences, \
        one_hot_labels, text_sequence_lengths, label_sequences, encoder_embedding_matrix, \
        decoder_embedding_matrix, data_size, word_index, integer_text_sequences = \
        get_data(text_file_path, command_line_args['vocab_size'], label_file_path,
                 command_line_args['use_pretrained_embeddings'])

    # Build model
    logger.info("Building model architecture")
    network = adversarial_autoencoder.AdversarialAutoencoder(
        num_labels, max_sequence_length, vocab_size, sos_index, eos_index,
        encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences,
        one_hot_labels, text_sequence_lengths, label_sequences)
    network.build_model()

    # Train and save model
    logger.info("Training model")
    sess = get_tensorflow_session()
    network.train(sess, data_size, command_line_args['training_epochs'])
    sess.close()
    logger.info("Training complete!")

    # Restore model and run inference
    logger.info("Inferring test samples")
    sess = get_tensorflow_session()
    inference_set_size = data_size
    offset = 0
    logger.debug("inference range: {}-{}".format(offset, (offset + inference_set_size)))
    generated_sequences, final_index = network.infer(sess, offset, inference_set_size)
    execute_post_inference_operations(
        word_index, integer_text_sequences, offset, final_index, generated_sequences)
    logger.info("Inference complete!")


def get_tensorflow_session():
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True

    return tf.Session(config=config_proto)


if __name__ == "__main__":
    main(sys.argv[1:])
