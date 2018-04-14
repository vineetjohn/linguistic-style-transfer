import argparse
import os
import pickle
import sys
from datetime import datetime as dt

import numpy as np
import tensorflow as tf

from linguistic_style_transfer_model.config import global_config, model_config
from linguistic_style_transfer_model.config.options import Options
from linguistic_style_transfer_model.models import adversarial_autoencoder
from linguistic_style_transfer_model.utils import bleu_scorer, data_postprocessor, \
    data_preprocessor, log_initializer, word_embedder, tsne_interface

logger = None


def get_data(options):
    [word_index, actual_sequences, padded_sequences, text_sequence_lengths, bow_representations] = \
        data_preprocessor.get_text_sequences(options.text_file_path, options.vocab_size)
    logger.debug("text_sequence_lengths: {}".format(text_sequence_lengths.shape))
    logger.debug("padded_sequences: {}".format(padded_sequences.shape))

    label_sequences, one_hot_labels, num_labels = \
        data_preprocessor.get_labels(options.label_file_path)
    logger.debug("one_hot_labels.shape: {}".format(one_hot_labels.shape))

    return [word_index, actual_sequences, padded_sequences, text_sequence_lengths,
            label_sequences, one_hot_labels, num_labels, bow_representations]


def get_average_label_embeddings(data_size, label_sequences):
    with open(global_config.all_style_embeddings_path, 'rb') as pickle_file:
        all_style_embeddings = pickle.load(pickle_file)
    with open(global_config.all_content_embeddings_path, 'rb') as pickle_file:
        all_content_embeddings = pickle.load(pickle_file)

    style_embeddings = np.asarray(all_style_embeddings)
    content_embeddings = np.asarray(all_content_embeddings)

    style_embedding_map = dict()
    content_embedding_map = dict()

    for i in range(data_size - (data_size % model_config.batch_size)):
        label = label_sequences[i][0]

        if label not in style_embedding_map:
            style_embedding_map[label] = list()
        style_embedding_map[label].append(style_embeddings[i])

        if label not in content_embedding_map:
            content_embedding_map[label] = list()
        content_embedding_map[label].append(content_embeddings[i])

    tsne_interface.generate_plot_coordinates(style_embedding_map, global_config.style_coordinates_path)
    tsne_interface.generate_plot_coordinates(content_embedding_map, global_config.content_coordinates_path)

    with open(global_config.label_mapped_style_embeddings_path, 'wb') as pickle_file:
        pickle.dump(style_embedding_map, pickle_file)
    logger.debug("Pickled label mapped style embeddings")

    average_label_embeddings = dict()
    for label in style_embedding_map:
        average_label_embeddings[label] = np.mean(style_embedding_map[label], axis=0)

    return average_label_embeddings


def flush_ground_truth_sentences(actual_sequences, start_index, final_index,
                                 inverse_word_index, timestamped_file_suffix):
    actual_sequences = actual_sequences[start_index:final_index]

    actual_word_lists = \
        [data_postprocessor.generate_words_from_indices(x, inverse_word_index)
         for x in actual_sequences]

    actual_sentences = [" ".join(x) for x in actual_word_lists]

    output_file_path = "output/{}/actual_sentences.txt".format(timestamped_file_suffix)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as output_file:
        for sentence in actual_sentences:
            output_file.write(sentence + "\n")

    return actual_word_lists


def execute_post_inference_operations(actual_word_lists, generated_sequences, final_sequence_lengths,
                                      inverse_word_index, timestamped_file_suffix, mode):
    logger.debug("Minimum generated sentence length: {}".format(min(final_sequence_lengths)))

    # first trims the generates sentences down to the length the decoder returns
    # then trim any <eos> token
    trimmed_generated_sequences = \
        [[index for index in sequence
          if index != global_config.predefined_word_index[global_config.eos_token]]
         for sequence in [x[:(y - 1)] for (x, y) in zip(generated_sequences, final_sequence_lengths)]]

    generated_word_lists = \
        [data_postprocessor.generate_words_from_indices(x, inverse_word_index)
         for x in trimmed_generated_sequences]

    # Evaluate model scores
    bleu_scores = bleu_scorer.get_corpus_bleu_scores(
        [[x] for x in actual_word_lists], generated_word_lists)
    logger.info("bleu_scores: {}".format(bleu_scores))
    generated_sentences = [" ".join(x) for x in generated_word_lists]

    output_file_path = "output/{}/generated_{}.txt".format(timestamped_file_suffix, mode)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as output_file:
        for sentence in generated_sentences:
            output_file.write(sentence + "\n")


def get_word_embeddings(word_index, use_pretrained_embeddings, train_model):
    encoder_embedding_matrix = np.random.uniform(
        size=(global_config.vocab_size, global_config.embedding_size),
        low=-0.05, high=0.05).astype(dtype=np.float32)
    logger.debug("encoder_embedding_matrix: {}".format(encoder_embedding_matrix.shape))

    decoder_embedding_matrix = np.random.uniform(
        size=(global_config.vocab_size, global_config.embedding_size),
        low=-0.05, high=0.05).astype(dtype=np.float32)
    logger.debug("decoder_embedding_matrix: {}".format(decoder_embedding_matrix.shape))

    if train_model and use_pretrained_embeddings:
        logger.info("Loading pretrained embeddings")
        encoder_embedding_matrix, decoder_embedding_matrix = \
            word_embedder.add_word_vectors_to_embeddings(
                word_index, global_config.word_vector_path, encoder_embedding_matrix,
                decoder_embedding_matrix)

    return encoder_embedding_matrix, decoder_embedding_matrix


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-mode", action="store_true", default=False)
    parser.add_argument("--train-model", action="store_true", default=False)
    parser.add_argument("--infer-sequences", action="store_true", default=False)
    parser.add_argument("--generate-novel-text", action="store_true", default=False)
    parser.add_argument("--use-pretrained-embeddings", action="store_true", default=False)
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--label-file-path", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--training-epochs", type=int, default=10)
    parser.add_argument("--logging-level", type=str, default="INFO")

    options = parser.parse_args(args=argv, namespace=Options())

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options.logging_level)

    if not (options.train_model or options.infer_sequences or options.generate_novel_text):
        logger.info("Nothing to do. Exiting ...")
        sys.exit(0)

    global_config.training_epochs = options.training_epochs

    # Retrieve all data
    logger.info("Reading data ...")
    [word_index, actual_sequences, padded_sequences, text_sequence_lengths,
     label_sequences, one_hot_labels, num_labels, bow_representations] = get_data(options)
    data_size = padded_sequences.shape[0]

    encoder_embedding_matrix, decoder_embedding_matrix = \
        get_word_embeddings(word_index, options.use_pretrained_embeddings, options.train_model)

    # Build model
    logger.info("Building model architecture ...")
    network = adversarial_autoencoder.AdversarialAutoencoder(
        padded_sequences, text_sequence_lengths, one_hot_labels, num_labels,
        word_index, encoder_embedding_matrix, decoder_embedding_matrix, bow_representations)
    network.build_model()

    # Train and save model
    if options.train_model:
        logger.info("Training model ...")
        sess = get_tensorflow_session()
        network.train(sess, data_size)
        sess.close()
        logger.info("Training complete!")

    if options.infer_sequences or options.generate_novel_text:
        samples_size = data_size - (data_size % model_config.batch_size)
        offset = 0
        logger.debug("Sampling range: {}-{}".format(offset, (offset + samples_size)))

        inverse_word_index = {v: k for k, v in word_index.items()}
        timestamped_file_suffix = dt.now().strftime("%Y%m%d%H%M%S")

        actual_word_lists = flush_ground_truth_sentences(
            actual_sequences, offset, offset + samples_size,
            inverse_word_index, timestamped_file_suffix)

        # Restore model and run inference
        if options.infer_sequences:
            logger.info("Inferring test samples ...")
            sess = get_tensorflow_session()
            generated_sequences, final_sequence_lengths = \
                network.infer(sess, offset, samples_size)
            sess.close()
            execute_post_inference_operations(
                actual_word_lists, generated_sequences, final_sequence_lengths,
                inverse_word_index, timestamped_file_suffix,
                "reconstructed_sentences")
            logger.info("Inference complete!")

        # Enforce a particular style embedding and regenerate text
        if options.generate_novel_text:

            logger.info("Generating novel text ...")
            average_label_embeddings = get_average_label_embeddings(
                data_size, label_sequences)
            for i in range(num_labels):
                style_choice = i + 1
                logger.info("Style chosen: {}".format(style_choice))

                style_embedding = np.asarray(average_label_embeddings[style_choice])

                sess = get_tensorflow_session()
                generated_sequences, final_sequence_lengths = \
                    network.generate_novel_sentences(sess, offset, samples_size, style_embedding)
                sess.close()

                execute_post_inference_operations(
                    actual_word_lists, generated_sequences, final_sequence_lengths,
                    inverse_word_index, timestamped_file_suffix,
                    "novel_sentences_{}".format(style_choice))

                logger.info("Generation complete!")


def get_tensorflow_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    config_proto = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True,
        gpu_options=gpu_options)

    return tf.Session(config=config_proto)


if __name__ == "__main__":
    main(sys.argv[1:])
