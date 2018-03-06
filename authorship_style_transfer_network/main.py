import argparse
import sys
from datetime import datetime as dt

import numpy as np
import tensorflow as tf

from authorship_style_transfer_network.models import adversarial_autoencoder
from authorship_style_transfer_network.utils import bleu_scorer
from authorship_style_transfer_network.utils import data_postprocessor
from authorship_style_transfer_network.utils import data_preprocessor
from authorship_style_transfer_network.utils import global_constants
from authorship_style_transfer_network.utils import log_initializer
from authorship_style_transfer_network.utils import word_embedder

logger = None


def get_data(text_file_path, vocab_size, label_file_path):
    padded_sequences, text_sequence_lengths, word_index, max_sequence_length, actual_sequences = \
        data_preprocessor.get_text_sequences(text_file_path, vocab_size)
    logger.debug("text_sequence_lengths: {}".format(text_sequence_lengths.shape))
    logger.debug("padded_sequences: {}".format(padded_sequences.shape))

    sos_index = word_index['sos']
    eos_index = word_index['eos']
    data_size = padded_sequences.shape[0]

    one_hot_labels, num_labels, label_sequences = data_preprocessor.get_labels(label_file_path)
    logger.debug("one_hot_labels.shape: {}".format(one_hot_labels.shape))

    return num_labels, max_sequence_length, vocab_size, sos_index, eos_index, padded_sequences, \
           one_hot_labels, text_sequence_lengths, label_sequences, data_size, word_index, actual_sequences


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
    logger.debug("average_author_embeddings: {}".format(average_author_embeddings))


def execute_post_inference_operations(word_index, actual_sequences, start_index, final_index,
                                      generated_sequences, final_sequence_lengths, max_sequence_length):
    logger.debug("Minimum generated sentence length: {}".format(min(final_sequence_lengths)))

    inverse_word_index = {v: k for k, v in word_index.items()}
    actual_sequences = actual_sequences[start_index:final_index]
    trimmed_generated_sequences = [x[:y] for (x, y) in zip(generated_sequences, final_sequence_lengths)]

    actual_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        actual_sequences, maxlen=max_sequence_length, padding='post', truncating='post',
        value=word_index['eos'])
    generated_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        trimmed_generated_sequences, maxlen=max_sequence_length, padding='post', truncating='post',
        value=word_index['eos'])

    actual_word_lists = \
        [data_postprocessor.generate_words_from_indices(x, inverse_word_index)
         for x in actual_sequences]
    generated_word_lists = \
        [data_postprocessor.generate_words_from_indices(x, inverse_word_index)
         for x in generated_sequences]

    # Evaluate model scores
    bleu_scores = bleu_scorer.get_corpus_bleu_scores(
        [[x] for x in actual_word_lists], generated_word_lists)
    logger.info("bleu_scores: {}".format(bleu_scores))

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


def get_word_embeddings(vocab_size, word_index, use_pretrained_embeddings, train_model):
    encoder_embedding_matrix = np.random.uniform(
        low=-0.05, high=0.05, size=(vocab_size, global_constants.embedding_size)).astype(dtype=np.float32)
    decoder_embedding_matrix = np.random.uniform(
        low=-0.05, high=0.05, size=(vocab_size, global_constants.embedding_size)).astype(dtype=np.float32)
    logger.debug("encoder_embedding_matrix: {}".format(encoder_embedding_matrix.shape))
    logger.debug("decoder_embedding_matrix: {}".format(decoder_embedding_matrix.shape))

    if train_model and use_pretrained_embeddings:
        logger.info("Loading pretrained embeddings")
        encoder_embedding_matrix, decoder_embedding_matrix = word_embedder.add_word_vectors_to_embeddings(
            word_index, global_constants.word_vector_path, encoder_embedding_matrix,
            decoder_embedding_matrix, vocab_size)

    return encoder_embedding_matrix, decoder_embedding_matrix


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-mode", action="store_true", default=False)
    parser.add_argument("--train-model", action="store_true", default=False)
    parser.add_argument("--infer-sequences", action="store_true", default=False)
    parser.add_argument("--use-pretrained-embeddings", action="store_true", default=False)
    parser.add_argument("--training-epochs", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--logging-level", type=str, default="INFO")

    args_namespace = parser.parse_args(argv)
    command_line_args = vars(args_namespace)

    global logger
    logger = log_initializer.setup_custom_logger(
        global_constants.logger_name, command_line_args['logging_level'])

    if command_line_args['dev_mode']:
        logger.info("Running in dev mode")
        text_file_path = "data/c50-articles-dev.txt"
        label_file_path = "data/c50-labels-dev.txt"
    else:
        text_file_path = "data/c50-articles.txt"
        label_file_path = "data/c50-labels.txt"

    if not (command_line_args['train_model'] or command_line_args['infer_sequences']):
        logger.info("Nothing to do. Exiting ...")
        sys.exit(0)

    # Retrieve all data
    num_labels, max_sequence_length, vocab_size, sos_index, eos_index, padded_sequences, \
    one_hot_labels, text_sequence_lengths, label_sequences, data_size, word_index, actual_sequences = \
        get_data(text_file_path, command_line_args['vocab_size'], label_file_path)

    encoder_embedding_matrix, decoder_embedding_matrix = \
        get_word_embeddings(vocab_size, word_index, command_line_args['use_pretrained_embeddings'],
                            command_line_args['train_model'])

    # Build model
    logger.info("Building model architecture")
    network = adversarial_autoencoder.AdversarialAutoencoder(
        num_labels, max_sequence_length, vocab_size, sos_index, eos_index,
        encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences,
        one_hot_labels, text_sequence_lengths, label_sequences)
    network.build_model()

    # Train and save model
    if command_line_args['train_model']:
        logger.info("Training model")
        sess = get_tensorflow_session()
        network.train(sess, data_size, command_line_args['training_epochs'])
        sess.close()
        logger.info("Training complete!")

    # Restore model and run inference
    if command_line_args['infer_sequences']:
        logger.info("Inferring test samples")
        sess = get_tensorflow_session()
        inference_set_size = data_size
        offset = 0
        logger.debug("inference range: {}-{}".format(offset, (offset + inference_set_size)))
        generated_sequences, final_index, final_sequence_lengths = \
            network.infer(sess, offset, inference_set_size)
        sess.close()
        logger.debug("final_sequence_lengths: {}".format(final_sequence_lengths))
        execute_post_inference_operations(
            word_index, actual_sequences, offset, final_index, generated_sequences, final_sequence_lengths,
            max_sequence_length)
        logger.info("Inference complete!")


def get_tensorflow_session():
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True

    return tf.Session(config=config_proto)


if __name__ == "__main__":
    main(sys.argv[1:])
