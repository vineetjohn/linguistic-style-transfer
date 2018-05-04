import logging
import os
import pickle
from datetime import datetime as dt

import numpy as np
import tensorflow as tf

from linguistic_style_transfer_model.config import global_config, model_config
from linguistic_style_transfer_model.evaluators import content_preservation, style_transfer
from linguistic_style_transfer_model.utils import data_processor

logger = logging.getLogger(global_config.logger_name)


class AdversarialAutoencoder:

    def __init__(self, padded_sequences, text_sequence_lengths, one_hot_labels, num_labels,
                 word_index, encoder_embedding_matrix, decoder_embedding_matrix):
        self.padded_sequences = padded_sequences
        self.text_sequence_lengths = text_sequence_lengths
        self.one_hot_labels = one_hot_labels
        self.num_labels = num_labels
        self.word_index = word_index
        self.encoder_embedding_matrix = encoder_embedding_matrix
        self.decoder_embedding_matrix = decoder_embedding_matrix

    def get_sentence_embedding(self, encoder_embedded_sequence):
        scope_name = "sentence_embedding"
        with tf.name_scope(scope_name):
            encoder_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=model_config.encoder_rnn_size),
                input_keep_prob=model_config.recurrent_state_keep_prob,
                output_keep_prob=model_config.recurrent_state_keep_prob,
                state_keep_prob=model_config.recurrent_state_keep_prob)
            encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=model_config.encoder_rnn_size),
                input_keep_prob=model_config.recurrent_state_keep_prob,
                output_keep_prob=model_config.recurrent_state_keep_prob,
                state_keep_prob=model_config.recurrent_state_keep_prob)

            _, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw,
                cell_bw=encoder_cell_bw,
                inputs=encoder_embedded_sequence,
                scope=scope_name,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            return tf.concat(values=encoder_states, axis=1)

    def get_style_embedding(self, sentence_embedding):

        style_embedding = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=sentence_embedding, units=model_config.style_embedding_size,
                activation=tf.nn.leaky_relu, name="style_embedding"),
            keep_prob=model_config.fully_connected_keep_prob)

        return style_embedding

    def get_content_embedding(self, sentence_embedding):

        content_embedding = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=sentence_embedding, units=model_config.content_embedding_size,
                activation=tf.nn.leaky_relu, name="content_embedding"),
            keep_prob=model_config.fully_connected_keep_prob)

        return content_embedding

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def get_style_label_prediction(self, style_embedding):

        style_label_prediction = tf.layers.dense(
            inputs=style_embedding, units=self.num_labels,
            activation=tf.nn.softmax, name="style_label_prediction")

        return style_label_prediction

    def get_adversarial_label_prediction(self, content_embedding):

        adversarial_label_mlp = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=content_embedding, units=model_config.content_embedding_size,
                activation=tf.nn.leaky_relu, name="adversarial_label_prediction_dense"),
            keep_prob=model_config.fully_connected_keep_prob)

        adversarial_label_prediction = tf.layers.dense(
            inputs=adversarial_label_mlp, units=self.num_labels,
            activation=tf.nn.softmax, name="adversarial_label_prediction")

        return adversarial_label_prediction

    def generate_output_sequence(self, embedded_sequence, generative_embedding, decoder_embeddings):

        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.contrib.rnn.GRUCell(num_units=model_config.decoder_rnn_size),
            input_keep_prob=model_config.recurrent_state_keep_prob,
            output_keep_prob=model_config.recurrent_state_keep_prob,
            state_keep_prob=model_config.recurrent_state_keep_prob)

        projection_layer = tf.layers.Dense(units=global_config.vocab_size, use_bias=False)

        with tf.name_scope("training_decoder"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_sequence,
                sequence_length=self.sequence_lengths)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=training_helper,
                initial_state=generative_embedding,
                output_layer=projection_layer)
            training_decoder.initialize("training_decoder")

            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, impute_finished=True,
                maximum_iterations=global_config.max_sequence_length,
                scope="training_decoder")

        with tf.name_scope('inference_decoder'):
            greedy_embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=decoder_embeddings,
                start_tokens=tf.fill(dims=[model_config.batch_size],
                                     value=self.word_index[global_config.sos_token]),
                end_token=self.word_index[global_config.eos_token])

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=greedy_embedding_helper,
                initial_state=generative_embedding,
                output_layer=projection_layer)
            inference_decoder.initialize("inference_decoder")

            inference_decoder_output, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, impute_finished=True,
                maximum_iterations=global_config.max_sequence_length,
                scope="inference_decoder")

        return [training_decoder_output.rnn_output, inference_decoder_output.sample_id,
                final_sequence_lengths]

    def build_model(self):

        # model inputs
        self.input_sequence = tf.placeholder(
            dtype=tf.int32, shape=[model_config.batch_size, global_config.max_sequence_length],
            name="input_sequence")
        logger.debug("input_sequence: {}".format(self.input_sequence))

        self.input_label = tf.placeholder(
            dtype=tf.float32, shape=[model_config.batch_size, self.num_labels],
            name="input_label")
        logger.debug("input_label: {}".format(self.input_label))

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[model_config.batch_size],
            name="sequence_lengths")
        logger.debug("sequence_lengths: {}".format(self.sequence_lengths))

        self.conditioned_generation_mode = tf.placeholder(
            dtype=tf.bool, name="conditioned_generation_mode")
        logger.debug("conditioned_generation_mode: {}".format(self.conditioned_generation_mode))

        self.conditioning_embedding = tf.placeholder(
            dtype=tf.float32,
            shape=[model_config.batch_size, model_config.style_embedding_size],
            name="conditioning_embedding")
        logger.debug("conditioning_embedding: {}".format(self.conditioning_embedding))

        self.epoch = tf.placeholder(dtype=tf.int32, shape=(), name="epoch")
        logger.debug("epoch: {}".format(self.epoch))

        decoder_input = tf.concat(
            values=[
                tf.fill(
                    dims=[model_config.batch_size, 1],
                    value=self.word_index[global_config.sos_token],
                    name="start_tokens"),
                self.input_sequence],
            axis=1, name="decoder_input")

        with tf.device('/cpu:0'):
            # word embeddings matrices
            encoder_embeddings = tf.get_variable(
                initializer=self.encoder_embedding_matrix, dtype=tf.float32,
                trainable=True, name="encoder_embeddings")
            logger.debug("encoder_embeddings: {}".format(encoder_embeddings))

            decoder_embeddings = tf.get_variable(
                initializer=self.decoder_embedding_matrix, dtype=tf.float32,
                trainable=True, name="decoder_embeddings")
            logger.debug("decoder_embeddings: {}".format(decoder_embeddings))

            # embedded sequences
            encoder_embedded_sequence = tf.nn.dropout(
                x=tf.nn.embedding_lookup(
                    params=encoder_embeddings, ids=self.input_sequence),
                keep_prob=model_config.sequence_word_keep_prob,
                name="encoder_embedded_sequence")
            logger.debug("encoder_embedded_sequence: {}".format(encoder_embedded_sequence))

            decoder_embedded_sequence = tf.nn.dropout(
                x=tf.nn.embedding_lookup(params=decoder_embeddings, ids=decoder_input),
                keep_prob=model_config.sequence_word_keep_prob,
                name="decoder_embedded_sequence")
            logger.debug("decoder_embedded_sequence: {}".format(decoder_embedded_sequence))

        sentence_embedding = self.get_sentence_embedding(encoder_embedded_sequence)

        # style embedding
        self.style_embedding = self.get_style_embedding(sentence_embedding)
        final_style_embedding = tf.cond(
            pred=self.conditioned_generation_mode,
            true_fn=lambda: self.conditioning_embedding,
            false_fn=lambda: self.style_embedding)
        logger.debug("style_embedding: {}".format(final_style_embedding))

        # content embedding
        self.content_embedding = self.get_content_embedding(sentence_embedding)
        logger.debug("content_embedding: {}".format(self.content_embedding))

        # concatenated generative embedding
        generative_embedding = tf.layers.dense(
            inputs=tf.concat(values=[final_style_embedding, self.content_embedding], axis=1),
            units=model_config.decoder_rnn_size, activation=tf.nn.leaky_relu,
            name="generative_embedding")
        logger.debug("generative_embedding: {}".format(generative_embedding))

        # sequence predictions
        with tf.name_scope('sequence_prediction'):
            training_output, self.inference_output, self.final_sequence_lengths = \
                self.generate_output_sequence(
                    decoder_embedded_sequence, generative_embedding, decoder_embeddings)
            logger.debug("training_output: {}".format(training_output))
            logger.debug("inference_output: {}".format(self.inference_output))

        # adversarial loss
        with tf.name_scope('adversarial_loss'):
            adversarial_label_prediction = self.get_adversarial_label_prediction(self.content_embedding)
            logger.debug("adversarial_label_prediction: {}".format(adversarial_label_prediction))

            self.adversarial_entropy = tf.reduce_mean(
                input_tensor=tf.reduce_sum(
                    input_tensor=-adversarial_label_prediction *
                                 tf.log(adversarial_label_prediction + model_config.epsilon), axis=1))
            logger.debug("adversarial_entropy: {}".format(self.adversarial_entropy))

            self.adversarial_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=adversarial_label_prediction, label_smoothing=0.1)
            logger.debug("adversarial_loss: {}".format(self.adversarial_loss))

        # style prediction loss
        with tf.name_scope('style_prediction_loss'):
            style_label_prediction = \
                self.get_style_label_prediction(self.style_embedding)
            logger.debug("style_label_prediction: {}".format(style_label_prediction))

            self.style_prediction_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=style_label_prediction, label_smoothing=0.1)
            logger.debug("style_prediction_loss: {}".format(self.style_prediction_loss))

        # reconstruction loss
        with tf.name_scope('reconstruction_loss'):
            batch_maxlen = tf.reduce_max(self.sequence_lengths)
            logger.debug("batch_maxlen: {}".format(batch_maxlen))

            # the training decoder only emits outputs equal in time-steps to the
            # max time in the current batch
            target_sequence = tf.slice(
                input_=self.input_sequence,
                begin=[0, 0],
                size=[model_config.batch_size, batch_maxlen],
                name="target_sequence")
            logger.debug("target_sequence: {}".format(target_sequence))

            output_sequence_mask = tf.sequence_mask(
                lengths=tf.add(x=self.sequence_lengths, y=1),
                maxlen=batch_maxlen,
                dtype=tf.float32)

            self.reconstruction_loss = tf.contrib.seq2seq.sequence_loss(
                logits=training_output, targets=target_sequence,
                weights=output_sequence_mask)
            logger.debug("reconstruction_loss: {}".format(self.reconstruction_loss))

        # tensorboard logging variable summaries
        tf.summary.scalar(tensor=self.reconstruction_loss, name="reconstruction_loss_summary")
        tf.summary.scalar(tensor=self.style_prediction_loss, name="style_prediction_loss_summary")
        tf.summary.scalar(tensor=self.adversarial_loss, name="adversarial_loss_summary")

    def get_batch_indices(self, offset, batch_number, data_limit):

        start_index = offset + (batch_number * model_config.batch_size)
        end_index = offset + ((batch_number + 1) * model_config.batch_size)
        end_index = data_limit if end_index > data_limit else end_index

        return start_index, end_index

    def run_batch(self, sess, start_index, end_index, fetches, padded_sequences,
                  one_hot_labels, text_sequence_lengths,
                  conditioning_embedding, conditioned_generation_mode, current_epoch):

        if not conditioned_generation_mode:
            conditioning_embedding = np.random.uniform(
                size=(model_config.batch_size, model_config.style_embedding_size),
                low=-0.05, high=0.05).astype(dtype=np.float32)

        ops = sess.run(
            fetches=fetches,
            feed_dict={
                self.input_sequence: padded_sequences[start_index: end_index],
                self.input_label: one_hot_labels[start_index: end_index],
                self.sequence_lengths: text_sequence_lengths[start_index: end_index],
                self.conditioned_generation_mode: conditioned_generation_mode,
                self.conditioning_embedding: conditioning_embedding,
                self.epoch: current_epoch
            })

        return ops

    def train(self, sess, data_size, validation_sequences, validation_sequence_lengths,
              validation_labels, inverse_word_index, validation_actual_word_lists, options):

        writer = tf.summary.FileWriter(
            logdir="/tmp/tensorflow_logs/" + dt.now().strftime("%Y%m%d-%H%M%S") + "/",
            graph=sess.graph)

        trainable_variables = tf.trainable_variables()
        logger.debug("trainable_variables: {}".format(trainable_variables))
        self.composite_loss = \
            self.reconstruction_loss \
            - (self.adversarial_entropy * model_config.adversarial_discriminator_loss_weight) \
            + (self.style_prediction_loss * model_config.style_prediction_loss_weight)
        tf.summary.scalar(tensor=self.composite_loss, name="composite_loss")
        self.all_summaries = tf.summary.merge_all()

        adversarial_variable_labels = ["adversarial_label_prediction"]

        # optimize adversarial classification
        adversarial_training_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=model_config.adversarial_discriminator_learning_rate)
        adversarial_training_variables = [
            x for x in trainable_variables if any(
                scope in x.name for scope in adversarial_variable_labels)]
        logger.debug("adversarial_training_optimizer.variables: {}".format(adversarial_training_variables))
        adversarial_training_operation = None
        for i in range(model_config.adversarial_discriminator_iterations):
            adversarial_training_operation = adversarial_training_optimizer.minimize(
                loss=self.adversarial_loss,
                var_list=adversarial_training_variables)

        # optimize reconstruction
        reconstruction_training_optimizer = tf.train.AdamOptimizer(
            learning_rate=model_config.autoencoder_learning_rate)
        reconstruction_training_variables = [
            x for x in trainable_variables if all(
                scope not in x.name for scope in adversarial_variable_labels)]
        logger.debug("reconstruction_training_optimizer.variables: {}".format(reconstruction_training_variables))
        reconstruction_training_operation = None
        for i in range(model_config.autoencoder_iterations):
            reconstruction_training_operation = reconstruction_training_optimizer.minimize(
                loss=self.composite_loss, var_list=reconstruction_training_variables)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        training_examples_size = data_size
        num_batches = training_examples_size // model_config.batch_size
        logger.debug("Training - texts shape: {}; labels shape {}"
                     .format(self.padded_sequences[:training_examples_size].shape,
                             self.one_hot_labels[:training_examples_size].shape))

        for current_epoch in range(1, global_config.training_epochs + 1):

            all_style_embeddings = list()
            all_content_embeddings = list()

            shuffle_indices = np.random.permutation(np.arange(data_size))

            shuffled_padded_sequences = self.padded_sequences[shuffle_indices]
            shuffled_one_hot_labels = self.one_hot_labels[shuffle_indices]
            shuffled_text_sequence_lengths = self.text_sequence_lengths[shuffle_indices]

            for batch_number in range(num_batches):
                (start_index, end_index) = self.get_batch_indices(
                    offset=0, batch_number=batch_number, data_limit=data_size)

                fetches = \
                    [reconstruction_training_operation,
                     adversarial_training_operation,
                     self.reconstruction_loss,
                     self.adversarial_loss,
                     self.adversarial_entropy,
                     self.style_prediction_loss,
                     self.composite_loss,
                     self.style_embedding,
                     self.content_embedding,
                     self.all_summaries]

                [_, _, reconstruction_loss, adversarial_loss, adversarial_entropy,
                 style_loss, composite_loss,
                 style_embeddings, content_embedding, all_summaries] = \
                    self.run_batch(
                        sess, start_index, end_index, fetches,
                        shuffled_padded_sequences, shuffled_one_hot_labels,
                        shuffled_text_sequence_lengths, None, False, current_epoch)

                log_msg = "[R: {:.2f}, ACE: {:.2f}, AE: {:.2f}, S: {:.2f}], " \
                          "Epoch {}-{}: {:.4f} "
                logger.info(log_msg.format(
                    reconstruction_loss, adversarial_loss, adversarial_entropy, style_loss,
                    current_epoch, batch_number, composite_loss))

                all_style_embeddings.extend(style_embeddings)
                all_content_embeddings.extend(content_embedding)

                writer.add_summary(all_summaries)
                writer.flush()

            saver.save(sess=sess, save_path=global_config.model_save_path)

            with open(global_config.all_style_embeddings_path, 'wb') as pickle_file:
                pickle.dump(all_style_embeddings, pickle_file)
            with open(global_config.all_content_embeddings_path, 'wb') as pickle_file:
                pickle.dump(all_content_embeddings, pickle_file)
            with open(global_config.all_shuffled_labels_path, 'wb') as pickle_file:
                pickle.dump(shuffled_one_hot_labels, pickle_file)

            if not current_epoch % global_config.validation_interval:

                logger.info("Running Validation {}:".format(current_epoch // global_config.validation_interval))

                glove_model = content_preservation.load_glove_model(options.validation_embeddings_file_path)

                validation_style_transfer_scores = list()
                validation_content_preservation_scores = list()
                for i in range(self.num_labels):

                    label_embeddings = list()
                    validation_sequences_to_transfer = list()
                    validation_labels_to_transfer = list()
                    validation_sequence_lengths_to_transfer = list()

                    for k in range(len(all_style_embeddings)):
                        if shuffled_one_hot_labels[k].tolist().index(1) == i:
                            label_embeddings.append(all_style_embeddings[k])

                    for k in range(len(validation_sequences)):
                        if validation_labels[k].tolist().index(1) != i:
                            validation_sequences_to_transfer.append(validation_sequences[k])
                            validation_labels_to_transfer.append(validation_labels[k])
                            validation_sequence_lengths_to_transfer.append(validation_sequence_lengths[k])

                    style_embedding = np.mean(np.asarray(label_embeddings), axis=0)

                    conditioning_embedding = np.tile(A=style_embedding, reps=(model_config.batch_size, 1))

                    validation_batches = len(validation_sequences_to_transfer) // model_config.batch_size
                    validation_generated_sequences = list()
                    validation_generated_sequence_lengths = list()
                    for val_batch_number in range(validation_batches):
                        (start_index, end_index) = self.get_batch_indices(
                            offset=0, batch_number=val_batch_number,
                            data_limit=len(validation_sequences_to_transfer))

                        [validation_generated_sequences_batch, validation_sequence_lengths_batch] = \
                            self.run_batch(
                                sess, start_index, end_index,
                                [self.inference_output, self.final_sequence_lengths],
                                validation_sequences_to_transfer, validation_labels_to_transfer,
                                validation_sequence_lengths_to_transfer,
                                conditioning_embedding, True, current_epoch)
                        validation_generated_sequences.extend(validation_generated_sequences_batch)
                        validation_generated_sequence_lengths.extend(validation_sequence_lengths_batch)

                    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                        validation_generated_sequences, maxlen=global_config.max_sequence_length, padding='post',
                        truncating='post', value=self.word_index[global_config.eos_token])

                    trimmed_generated_sequences = \
                        [[index for index in sequence
                          if index != global_config.predefined_word_index[global_config.eos_token]]
                         for sequence in [x[:(y - 1)] for (x, y) in zip(
                            validation_generated_sequences, validation_generated_sequence_lengths)]]

                    generated_word_lists = \
                        [data_processor.generate_words_from_indices(x, inverse_word_index)
                         for x in trimmed_generated_sequences]

                    generated_sentences = [" ".join(x) for x in generated_word_lists]

                    output_file_path = "output/{}/validation_sentences_{}.txt".format(
                        global_config.experiment_timestamp, i)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, 'w') as output_file:
                        for sentence in generated_sentences:
                            output_file.write(sentence + "\n")

                    [style_transfer_score, confusion_matrix] = style_transfer.get_style_transfer_score(
                        options.classifier_checkpoint_dir, output_file_path, i)
                    logger.debug("style_transfer_score: {}".format(style_transfer_score))
                    logger.debug("confusion_matrix: {}".format(confusion_matrix))

                    content_preservation_score = content_preservation.get_content_preservation_score(
                        validation_actual_word_lists, generated_word_lists, glove_model)
                    logger.debug("content_preservation_score: {}".format(content_preservation_score))

                    validation_style_transfer_scores.append(style_transfer_score)
                    validation_content_preservation_scores.append(content_preservation_score)

                logger.info("Total Style Transfer: {}".format(
                    np.mean(np.asarray(validation_style_transfer_scores))))
                logger.info("Total Content Preservation: {}".format(
                    np.mean(np.asarray(validation_content_preservation_scores))))

        writer.close()

    def infer(self, sess, offset, samples_size):

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=global_config.model_save_path)

        generated_sequences = list()
        final_sequence_lengths = list()
        num_batches = samples_size // model_config.batch_size

        end_index = None
        for batch_number in range(num_batches):
            (start_index, end_index) = self.get_batch_indices(
                offset=offset, batch_number=batch_number, data_limit=(offset + samples_size))

            generated_sequences_batch, final_sequence_lengths_batch = self.run_batch(
                sess, start_index, end_index, [self.inference_output, self.final_sequence_lengths],
                self.padded_sequences, self.one_hot_labels, self.text_sequence_lengths,
                None, False, 0)

            generated_sequences.extend(generated_sequences_batch)
            final_sequence_lengths.extend(final_sequence_lengths_batch)

        return generated_sequences, final_sequence_lengths

    def generate_novel_sentences(self, sess, padded_sequences, text_sequence_lengths, style_embedding,
                                 num_labels):

        conditioning_embedding = np.tile(A=style_embedding, reps=(model_config.batch_size, 1))

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=global_config.model_save_path)

        generated_sequences = list()
        final_sequence_lengths = list()
        num_batches = len(padded_sequences) // model_config.batch_size

        # these won't be needed to generate new sentences, so just use random numbers
        one_hot_labels_placeholder = np.random.randint(
            low=0, high=1, size=(len(padded_sequences), num_labels)).astype(dtype=np.int32)

        end_index = None
        for batch_number in range(num_batches):
            (start_index, end_index) = self.get_batch_indices(
                offset=0, batch_number=batch_number, data_limit=len(padded_sequences))

            generated_sequences_batch, final_sequence_lengths_batch = self.run_batch(
                sess, start_index, end_index, [self.inference_output, self.final_sequence_lengths],
                padded_sequences, one_hot_labels_placeholder, text_sequence_lengths,
                conditioning_embedding, True, 0)

            generated_sequences.extend(generated_sequences_batch)
            final_sequence_lengths.extend(final_sequence_lengths_batch)

        return generated_sequences, final_sequence_lengths
