import json
import logging
import numpy as np
import os
import pickle
import random
import tensorflow as tf

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.config.model_config import mconf
from linguistic_style_transfer_model.evaluators import content_preservation, style_transfer
from linguistic_style_transfer_model.utils import data_processor, custom_decoder

logger = logging.getLogger(global_config.logger_name)


class AdversarialAutoencoder:

    def get_sentence_embedding(self, encoder_embedded_sequence):

        scope_name = "sentence_embedding"
        with tf.name_scope(scope_name):
            encoder_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=mconf.encoder_rnn_size),
                input_keep_prob=self.recurrent_state_keep_prob,
                output_keep_prob=self.recurrent_state_keep_prob,
                state_keep_prob=self.recurrent_state_keep_prob)
            encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=mconf.encoder_rnn_size),
                input_keep_prob=self.recurrent_state_keep_prob,
                output_keep_prob=self.recurrent_state_keep_prob,
                state_keep_prob=self.recurrent_state_keep_prob)

            _, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw, cell_bw=encoder_cell_bw,
                inputs=encoder_embedded_sequence, scope=scope_name,
                sequence_length=self.sequence_lengths, dtype=tf.float32)

            return tf.concat(values=encoder_states, axis=1, name="sentence_embedding")

    def get_style_embedding(self, sentence_embedding, num_labels):

        with tf.name_scope("style_embedding"):
            style_embedding = tf.nn.dropout(
                x=tf.layers.dense(
                    inputs=sentence_embedding,
                    units=mconf.style_embedding_size_per_label * num_labels,
                    activation=tf.nn.leaky_relu, name="style_embedding"),
                keep_prob=self.fully_connected_keep_prob)

            return style_embedding

    def get_content_embedding(self, sentence_embedding):

        with tf.name_scope("content_embedding"):
            content_embedding = tf.nn.dropout(
                x=tf.layers.dense(
                    inputs=sentence_embedding,
                    units=mconf.content_embedding_size,
                    activation=tf.nn.leaky_relu, name="content_embedding"),
                keep_prob=self.fully_connected_keep_prob)

            return content_embedding

    def get_content_adversary_prediction(self, style_embedding):

        content_adversary_mlp = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=style_embedding, units=global_config.bow_size,
                activation=tf.nn.leaky_relu, name="content_adversary_mlp"),
            keep_prob=self.fully_connected_keep_prob)

        content_adversary_prediction = tf.layers.dense(
            inputs=content_adversary_mlp, units=global_config.bow_size,
            activation=tf.nn.softmax, name="content_adversary_prediction")

        return content_adversary_prediction

    def get_style_adversary_prediction(self, content_embedding, num_labels):

        style_adversary_mlp = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=content_embedding, units=mconf.content_embedding_size,
                activation=tf.nn.leaky_relu, name="style_adversary_mlp"),
            keep_prob=self.fully_connected_keep_prob)

        style_adversary_prediction = tf.layers.dense(
            inputs=style_adversary_mlp, units=num_labels,
            activation=tf.nn.softmax, name="style_adversary_prediction")

        return style_adversary_prediction

    def generate_output_sequence(self, embedded_sequence, generative_embedding,
                                 decoder_embeddings, word_index, batch_size):

        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.contrib.rnn.GRUCell(num_units=mconf.decoder_rnn_size),
            input_keep_prob=self.recurrent_state_keep_prob,
            output_keep_prob=self.recurrent_state_keep_prob,
            state_keep_prob=self.recurrent_state_keep_prob)

        projection_layer = tf.layers.Dense(units=global_config.vocab_size, use_bias=False)

        init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        training_decoder_scope_name = "training_decoder"
        with tf.name_scope(training_decoder_scope_name):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_sequence,
                sequence_length=self.sequence_lengths)

            training_decoder = custom_decoder.CustomBasicDecoder(
                cell=decoder_cell, helper=training_helper,
                initial_state=init_state,
                latent_vector=generative_embedding,
                output_layer=projection_layer)
            training_decoder.initialize(training_decoder_scope_name)

            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, impute_finished=True,
                maximum_iterations=global_config.max_sequence_length,
                scope=training_decoder_scope_name)

        inference_decoder_scope_name = "inference_decoder"
        with tf.name_scope(inference_decoder_scope_name):
            greedy_embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=decoder_embeddings,
                start_tokens=tf.fill(dims=[batch_size],
                                     value=word_index[global_config.sos_token]),
                end_token=word_index[global_config.eos_token])

            inference_decoder = custom_decoder.CustomBasicDecoder(
                cell=decoder_cell, helper=greedy_embedding_helper,
                initial_state=init_state,
                latent_vector=generative_embedding,
                output_layer=projection_layer)
            inference_decoder.initialize(inference_decoder_scope_name)

            inference_decoder_output, _, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(
                    decoder=inference_decoder, impute_finished=True,
                    maximum_iterations=global_config.max_sequence_length,
                    scope=inference_decoder_scope_name)

        return [training_decoder_output.rnn_output, inference_decoder_output.sample_id, final_sequence_lengths]

    def mmd_penalty(self, sample_pz, sample_qz, batch_size, latent_vector_size):
        n = batch_size
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2
        half_size = tf.cast(half_size, tf.int32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2 * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2 * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2 * dotprods

        if mconf.mmd_kernel == global_config.MMDKernel.RBF:
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

            res1 = tf.exp(- distances_qz / 2 / sigma2_k)
            res1 += tf.exp(- distances_pz / 2 / sigma2_k)
            res1 = tf.multiply(res1, 1 - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2 / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2 / (nf * nf)
            stat = res1 - res2
        elif mconf.mmd_kernel == global_config.MMDKernel.IMQ:
            # sigma2_p # for normal sigma2_p = 1
            Cbase = 2 * latent_vector_size * 2 * 1
            stat = 0
            for scale in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1 - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2 / (nf * nf)
                stat += res1 - res2
        return stat

    def get_style_prior(self, batch_label, num_labels, batch_size):
        separate_arrays = list()
        for i in range(num_labels):
            npfunc = np.ones if i == batch_label else np.zeros
            separate_arrays.append(npfunc((batch_size, mconf.style_embedding_size_per_label)))

        combined = np.concatenate(separate_arrays, axis=1) * mconf.prior_mean_multiplier

        return combined

    def compute_batch_entropy(self, x):
        return tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=-x * tf.log(x + mconf.epsilon), axis=1))

    def build_model(self, word_index, encoder_embedding_matrix, decoder_embedding_matrix, num_labels):

        # model inputs
        self.input_sequence = tf.placeholder(
            dtype=tf.int32, shape=[None, global_config.max_sequence_length],
            name="input_sequence")
        logger.debug("input_sequence: {}".format(self.input_sequence))

        batch_size = tf.shape(self.input_sequence)[0]
        logger.debug("batch_size: {}".format(batch_size))

        self.input_label = tf.placeholder(
            dtype=tf.float32, shape=[None, num_labels], name="input_label")
        logger.debug("input_label: {}".format(self.input_label))

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[None], name="sequence_lengths")
        logger.debug("sequence_lengths: {}".format(self.sequence_lengths))

        self.input_bow_representations = tf.placeholder(
            dtype=tf.float32, shape=[None, global_config.bow_size],
            name="input_bow_representations")
        logger.debug("input_bow_representations: {}".format(self.input_bow_representations))

        self.inference_mode = tf.placeholder(dtype=tf.bool, name="inference_mode")
        logger.debug("inference_mode: {}".format(self.inference_mode))

        self.recurrent_state_keep_prob = tf.cond(
            pred=self.inference_mode,
            true_fn=lambda: 1.0,
            false_fn=lambda: mconf.recurrent_state_keep_prob)

        self.fully_connected_keep_prob = tf.cond(
            pred=self.inference_mode,
            true_fn=lambda: 1.0,
            false_fn=lambda: mconf.fully_connected_keep_prob)

        self.sequence_word_keep_prob = tf.cond(
            pred=self.inference_mode,
            true_fn=lambda: 1.0,
            false_fn=lambda: mconf.sequence_word_keep_prob)

        self.style_prior = tf.placeholder(
            dtype=tf.float32, shape=[None, mconf.style_embedding_size_per_label * num_labels],
            name="style_prior")
        logger.debug("style_prior: {}".format(self.style_prior))

        self.epoch = tf.placeholder(dtype=tf.float32, shape=(), name="epoch")
        logger.debug("epoch: {}".format(self.epoch))

        decoder_input = tf.concat(
            values=[tf.fill(dims=[batch_size, 1], value=word_index[global_config.sos_token]),
                    self.input_sequence], axis=1, name="decoder_input")

        with tf.device('/cpu:0'):
            with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
                # word embeddings matrices
                encoder_embeddings = tf.get_variable(
                    initializer=encoder_embedding_matrix, dtype=tf.float32,
                    trainable=True, name="encoder_embeddings")
                logger.debug("encoder_embeddings: {}".format(encoder_embeddings))

                decoder_embeddings = tf.get_variable(
                    initializer=decoder_embedding_matrix, dtype=tf.float32,
                    trainable=True, name="decoder_embeddings")
                logger.debug("decoder_embeddings: {}".format(decoder_embeddings))

                # embedded sequences
                encoder_embedded_sequence = tf.nn.dropout(
                    x=tf.nn.embedding_lookup(params=encoder_embeddings, ids=self.input_sequence),
                    keep_prob=self.sequence_word_keep_prob,
                    name="encoder_embedded_sequence")
                logger.debug("encoder_embedded_sequence: {}".format(encoder_embedded_sequence))

                decoder_embedded_sequence = tf.nn.dropout(
                    x=tf.nn.embedding_lookup(params=decoder_embeddings, ids=decoder_input),
                    keep_prob=self.sequence_word_keep_prob,
                    name="decoder_embedded_sequence")
                logger.debug("decoder_embedded_sequence: {}".format(decoder_embedded_sequence))

        sentence_embedding = self.get_sentence_embedding(encoder_embedded_sequence)

        # style embedding
        style_embedding = self.get_style_embedding(sentence_embedding, num_labels)
        sampled_style_embedding = \
            self.style_prior + \
            tf.random_normal(
                shape=(batch_size, mconf.style_embedding_size_per_label * num_labels),
                dtype=tf.float32)
        self.style_wasserstein_loss = self.mmd_penalty(
            sampled_style_embedding, style_embedding, batch_size,
            mconf.style_embedding_size_per_label * num_labels)

        self.style_embedding = tf.cond(
            pred=self.inference_mode,
            true_fn=lambda: self.style_prior,
            false_fn=lambda: style_embedding)
        logger.debug("style_embedding: {}".format(self.style_embedding))

        # content embedding
        content_embedding = self.get_content_embedding(sentence_embedding)
        sampled_content_embedding = tf.random_normal(
            shape=(batch_size, mconf.content_embedding_size), dtype=tf.float32)
        self.content_wasserstein_loss = self.mmd_penalty(
            sampled_content_embedding, content_embedding, batch_size, mconf.content_embedding_size)

        self.content_embedding = content_embedding
        logger.debug("content_embedding: {}".format(self.content_embedding))

        # concatenated generative embedding
        generative_embedding = tf.layers.dense(
            inputs=tf.concat(values=[self.style_embedding, self.content_embedding], axis=1),
            units=mconf.decoder_rnn_size, activation=tf.nn.leaky_relu,
            name="generative_embedding")
        logger.debug("generative_embedding: {}".format(generative_embedding))

        # sequence predictions
        with tf.name_scope('sequence_prediction'):
            training_output, self.inference_output, self.final_sequence_lengths = \
                self.generate_output_sequence(
                    decoder_embedded_sequence, generative_embedding, decoder_embeddings,
                    word_index, batch_size)
            logger.debug("training_output: {}".format(training_output))
            logger.debug("inference_output: {}".format(self.inference_output))

        # adversarial loss
        with tf.name_scope('adversarial_objectives'):
            # style adversary
            style_adversary_prediction = self.get_style_adversary_prediction(self.content_embedding, num_labels)
            logger.debug("style_adversary_prediction: {}".format(style_adversary_prediction))

            self.quantized_style_adversary_prediction = tf.contrib.seq2seq.hardmax(
                logits=style_adversary_prediction, name="quantized_style_adversary_prediction")

            self.style_adversary_entropy = self.compute_batch_entropy(style_adversary_prediction)
            logger.debug("style_adversary_entropy: {}".format(self.style_adversary_entropy))

            self.style_adversary_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=style_adversary_prediction,
                label_smoothing=0.1)
            logger.debug("style_adversary_loss: {}".format(self.style_adversary_loss))

            # content adversary
            content_adversary_prediction = self.get_content_adversary_prediction(self.style_embedding)
            logger.debug("content_adversary_prediction: {}".format(content_adversary_prediction))

            self.content_adversary_entropy = self.compute_batch_entropy(content_adversary_prediction)
            logger.debug("content_adversary_entropy: {}".format(self.content_adversary_entropy))

            self.content_adversary_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_bow_representations, logits=content_adversary_prediction,
                label_smoothing=0.1)
            logger.debug("content_adversary_loss: {}".format(self.content_adversary_loss))

        # multi-task objectives
        with tf.name_scope('multitask_objectives'):
            # style multitask
            style_multitask_prediction = tf.nn.dropout(
                x=tf.layers.dense(
                    inputs=self.style_embedding, units=num_labels,
                    activation=tf.nn.softmax, name="style_multitask_prediction"),
                keep_prob=self.fully_connected_keep_prob)
            logger.debug("style_multitask_prediction: {}".format(style_multitask_prediction))

            self.quantized_style_multitask_prediction = tf.contrib.seq2seq.hardmax(
                logits=style_multitask_prediction, name="quantized_style_multitask_prediction")

            self.style_multitask_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=style_multitask_prediction, label_smoothing=0.1)
            logger.debug("style_multitask_loss: {}".format(self.style_multitask_loss))

            # bow multitask
            content_multitask_prediction = tf.nn.dropout(
                x=tf.layers.dense(
                    inputs=self.content_embedding, units=global_config.bow_size,
                    activation=tf.nn.leaky_relu, name="content_multitask_prediction"),
                keep_prob=self.fully_connected_keep_prob)
            logger.debug("content_multitask_prediction: {}".format(content_multitask_prediction))

            self.content_multitask_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_bow_representations, logits=content_multitask_prediction,
                label_smoothing=0.1)
            logger.debug("content_multitask_loss: {}".format(self.content_multitask_loss))

        # overall latent space classifier
        # not required for style transfer
        # used to prove disentanglement
        style_overall_prediction = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=tf.concat(values=[self.style_embedding, self.content_embedding], axis=1),
                units=num_labels, activation=tf.nn.softmax,
                name="style_overall_prediction"),
            keep_prob=self.fully_connected_keep_prob)
        logger.debug("style_overall_prediction: {}".format(style_overall_prediction))

        self.quantized_style_overall_prediction = tf.contrib.seq2seq.hardmax(
            logits=style_overall_prediction, name="quantized_style_overall_prediction")

        self.style_overall_prediction_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.input_label, logits=style_overall_prediction, label_smoothing=0.1)
        logger.debug("style_overall_prediction_loss: {}".format(self.style_overall_prediction_loss))

        # reconstruction loss
        with tf.name_scope('reconstruction_loss'):
            batch_maxlen = tf.reduce_max(self.sequence_lengths)
            logger.debug("batch_maxlen: {}".format(batch_maxlen))

            # the training decoder only emits outputs equal in time-steps to the
            # max time-steps in the current batch
            target_sequence = tf.slice(
                input_=self.input_sequence,
                begin=[0, 0],
                size=[batch_size, batch_maxlen],
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
        tf.summary.scalar(tensor=self.style_multitask_loss, name="style_multitask_loss_summary")
        tf.summary.scalar(tensor=self.style_adversary_loss, name="style_adversary_loss_summary")
        tf.summary.scalar(tensor=self.content_adversary_loss, name="content_adversary_loss_summary")
        tf.summary.scalar(tensor=self.content_multitask_loss, name="content_multitask_loss_summary")
        tf.summary.scalar(tensor=self.style_wasserstein_loss, name="style_wasserstein_loss")
        tf.summary.scalar(tensor=self.content_wasserstein_loss, name="content_wasserstein_loss")

    def run_batch(self, sess, fetches, style_prior, padded_sequences, one_hot_labels,
                  text_sequence_lengths, inference_mode, current_epoch):

        bow_representations = data_processor.get_bow_representations(padded_sequences)

        ops = sess.run(
            fetches=fetches,
            feed_dict={
                self.style_prior: style_prior,
                self.input_sequence: padded_sequences,
                self.input_label: one_hot_labels,
                self.sequence_lengths: text_sequence_lengths,
                self.input_bow_representations: bow_representations,
                self.inference_mode: inference_mode,
                self.epoch: current_epoch
            })

        return ops

    def train(self, sess, data_size, padded_sequences, text_sequence_lengths, one_hot_labels, num_labels,
              word_index, encoder_embedding_matrix, decoder_embedding_matrix, validation_sequences,
              validation_sequence_lengths, validation_labels, inverse_word_index, validation_actual_word_lists,
              options):

        writer = tf.summary.FileWriter(logdir=global_config.log_directory, graph=sess.graph)

        trainable_variables = tf.trainable_variables()
        logger.debug("trainable_variables: {}".format(trainable_variables))

        self.composite_loss = 0.0
        self.composite_loss += self.reconstruction_loss
        self.composite_loss += self.style_multitask_loss * mconf.style_multitask_loss_weight
        self.composite_loss += self.content_multitask_loss * mconf.content_multitask_loss_weight
        self.composite_loss -= self.style_adversary_entropy * mconf.style_adversary_loss_weight
        self.composite_loss -= self.content_adversary_entropy * mconf.content_adversary_loss_weight
        self.composite_loss += self.style_wasserstein_loss * mconf.style_wasserstein_weight
        self.composite_loss += self.content_wasserstein_loss * mconf.content_wasserstein_weight
        tf.summary.scalar(tensor=self.composite_loss, name="composite_loss_summary")
        self.all_summaries = tf.summary.merge_all()

        # optimize adversarial classification
        style_adversary_variable_labels = ["style_adversary"]
        content_adversary_variable_labels = ["content_adversary"]
        # style
        style_adversary_training_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=mconf.style_adversary_learning_rate)
        style_adversary_training_variables = [
            x for x in trainable_variables if any(
                scope in x.name for scope in style_adversary_variable_labels)]
        logger.debug("style_adversary_training_optimizer.variables: {}".format(
            style_adversary_training_variables))
        style_adversary_training_operation = style_adversary_training_optimizer.minimize(
            loss=self.style_adversary_loss,
            var_list=style_adversary_training_variables)
        # content
        content_adversary_training_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=mconf.content_adversary_learning_rate)
        content_adversary_training_variables = [
            x for x in trainable_variables if any(
                scope in x.name for scope in content_adversary_variable_labels)]
        logger.debug("content_adversary_training_optimizer.variables: {}".format(
            content_adversary_training_variables))
        content_adversary_training_operation = content_adversary_training_optimizer.minimize(
            loss=self.content_adversary_loss,
            var_list=content_adversary_training_variables)

        # optimize overall latent space classification
        style_overall_variable_labels = ["style_overall"]
        style_overall_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=mconf.autoencoder_learning_rate)
        style_overall_training_variables = [
            x for x in trainable_variables if any(scope in x.name for scope in style_overall_variable_labels)]
        logger.debug("style_overall_training_variables: {}".format(style_overall_training_variables))
        style_overall_training_operation = style_overall_optimizer.minimize(
            loss=self.style_overall_prediction_loss,
            var_list=style_overall_training_variables)

        # optimize reconstruction
        reconstruction_training_optimizer = tf.train.AdamOptimizer(
            learning_rate=mconf.autoencoder_learning_rate)
        reconstruction_training_variables = [
            x for x in trainable_variables if all(
                scope not in x.name for scope in
                (style_adversary_variable_labels +
                 content_adversary_variable_labels +
                 style_overall_variable_labels))]
        logger.debug("reconstruction_training_optimizer.variables: {}".format(reconstruction_training_variables))
        reconstruction_training_operation = reconstruction_training_optimizer.minimize(
            loss=self.composite_loss, var_list=reconstruction_training_variables)

        # Defining model variables is complete

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        logger.debug("Training - texts shape: {}; labels shape {}"
                     .format(padded_sequences.shape, one_hot_labels.shape))

        homogenous_batches = data_processor.get_label_homogenous_batches(
            mconf.batch_size, zip(one_hot_labels, padded_sequences, text_sequence_lengths))
        logger.info("Total Homogenous Batches: {}".format(len(homogenous_batches)))

        iteration = 0
        for current_epoch in range(1, options.training_epochs + 1):

            random.shuffle(homogenous_batches)

            all_style_embeddings = list()
            all_content_embeddings = list()
            all_shuffled_one_hot_labels = list()

            for batch_number, (batch_label, training_batch) in enumerate(homogenous_batches):
                one_hot_labels, sequences, sequence_lengths = zip(*training_batch)
                all_shuffled_one_hot_labels.extend(one_hot_labels)
                style_prior = self.get_style_prior(batch_label, num_labels, len(one_hot_labels))

                fetches = \
                    [reconstruction_training_operation,
                     style_adversary_training_operation,
                     content_adversary_training_operation,
                     style_overall_training_operation,
                     self.reconstruction_loss,
                     self.style_multitask_loss,
                     self.content_multitask_loss,
                     self.style_adversary_loss,
                     self.style_adversary_entropy,
                     self.content_adversary_loss,
                     self.content_adversary_entropy,
                     self.style_wasserstein_loss,
                     self.content_wasserstein_loss,
                     self.composite_loss,
                     self.style_embedding,
                     self.content_embedding,
                     self.all_summaries]

                [_, _, _, _,
                 reconstruction_loss,
                 style_multitask_loss, content_multitask_loss,
                 style_adversary_crossentropy, style_adversary_entropy,
                 content_adversary_crossentropy, content_adversary_entropy,
                 style_wasserstein_loss, content_wasserstein_loss,
                 composite_loss,
                 style_embeddings, content_embedding,
                 all_summaries] = \
                    self.run_batch(
                        sess, fetches, style_prior, sequences, one_hot_labels,
                        sequence_lengths, False, current_epoch)

                log_msg = "[R: {:.2f}, " \
                          "SMT: {:.2f}, CMT: {:.2f}, " \
                          "SCE: {:.2f}, SE: {:.2f}, " \
                          "CCE: {:.2f}, CE: {:.2f}, " \
                          "SWL: {:.2f}, CWL: {:.2f}] " \
                          "Epoch {}-{}: {:.4f}"
                logger.info(log_msg.format(
                    reconstruction_loss,
                    style_multitask_loss, content_multitask_loss,
                    style_adversary_crossentropy, style_adversary_entropy,
                    content_adversary_crossentropy, content_adversary_entropy,
                    style_wasserstein_loss, content_wasserstein_loss,
                    current_epoch, batch_number, composite_loss))

                all_style_embeddings.extend(style_embeddings)
                all_content_embeddings.extend(content_embedding)

                iteration += 1

                writer.add_summary(all_summaries, iteration)
                writer.flush()

            saver.save(sess=sess, save_path=global_config.model_save_path)

            np.save(file=global_config.all_style_embeddings_path, arr=np.asarray(all_style_embeddings))
            np.save(file=global_config.all_content_embeddings_path, arr=all_content_embeddings)
            with open(global_config.all_shuffled_labels_path, 'wb') as pickle_file:
                pickle.dump(all_shuffled_one_hot_labels, pickle_file)

            average_label_embeddings = data_processor.get_average_label_embeddings(
                data_size, options.dump_embeddings, current_epoch)
            with open(global_config.average_label_embeddings_path, 'wb') as pickle_file:
                pickle.dump(average_label_embeddings, pickle_file)

            # Code for validation run begins
            if not current_epoch % global_config.validation_interval:
                self.run_validation(options, num_labels, validation_sequences, validation_sequence_lengths,
                                    validation_labels, validation_actual_word_lists, all_style_embeddings,
                                    all_shuffled_one_hot_labels, inverse_word_index, current_epoch, sess)

        writer.close()

    def run_validation(self, options, num_labels, validation_sequences, validation_sequence_lengths,
                       validation_labels, validation_actual_word_lists, all_style_embeddings,
                       shuffled_one_hot_labels, inverse_word_index, current_epoch, sess):
        logger.info("Running Validation {}:".format(current_epoch // global_config.validation_interval))

        glove_model = content_preservation.load_glove_model(options.validation_embeddings_file_path)

        validation_style_transfer_scores = list()
        validation_content_preservation_scores = list()
        validation_word_overlap_scores = list()

        for i in range(num_labels):

            logger.info("validating label {}".format(i))

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

            validation_batches = len(validation_sequences_to_transfer) // mconf.batch_size
            if len(validation_sequences_to_transfer) % mconf.batch_size:
                validation_batches += 1

            validation_generated_sequences = list()
            validation_generated_sequence_lengths = list()
            for val_batch_number in range(validation_batches):
                (start_index, end_index) = data_processor.get_batch_indices(
                    batch_number=val_batch_number,
                    data_limit=len(validation_sequences_to_transfer),
                    batch_size=mconf.batch_size)

                style_prior = self.get_style_prior(i, num_labels, end_index - start_index)

                [validation_generated_sequences_batch, validation_sequence_lengths_batch] = \
                    self.run_batch(
                        sess, [self.inference_output, self.final_sequence_lengths], style_prior,
                        validation_sequences_to_transfer[start_index:end_index],
                        validation_labels_to_transfer[start_index:end_index],
                        validation_sequence_lengths_to_transfer[start_index:end_index],
                        True, current_epoch)
                validation_generated_sequences.extend(validation_generated_sequences_batch)
                validation_generated_sequence_lengths.extend(validation_sequence_lengths_batch)

            trimmed_generated_sequences = \
                [[index for index in sequence
                  if index != global_config.predefined_word_index[global_config.eos_token]]
                 for sequence in [x[:(y - 1)] for (x, y) in zip(
                    validation_generated_sequences, validation_generated_sequence_lengths)]]

            generated_word_lists = \
                [data_processor.generate_words_from_indices(x, inverse_word_index)
                 for x in trimmed_generated_sequences]

            generated_sentences = [" ".join(x) for x in generated_word_lists]

            output_file_path = "output/{}-training/validation_sentences_{}.txt".format(
                global_config.experiment_timestamp, i)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as output_file:
                for sentence in generated_sentences:
                    output_file.write(sentence + "\n")

            [style_transfer_score, confusion_matrix] = style_transfer.get_style_transfer_score(
                options.classifier_saved_model_path, output_file_path, i)
            logger.debug("style_transfer_score: {}".format(style_transfer_score))
            logger.debug("confusion_matrix:\n{}".format(confusion_matrix))

            content_preservation_score = content_preservation.get_content_preservation_score(
                validation_actual_word_lists, generated_word_lists, glove_model)
            logger.debug("content_preservation_score: {}".format(content_preservation_score))

            word_overlap_score = content_preservation.get_word_overlap_score(
                validation_actual_word_lists, generated_word_lists)
            logger.debug("word_overlap_score: {}".format(word_overlap_score))

            validation_style_transfer_scores.append(style_transfer_score)
            validation_content_preservation_scores.append(content_preservation_score)
            validation_word_overlap_scores.append(word_overlap_score)

        aggregate_style_transfer = np.mean(np.asarray(validation_style_transfer_scores))
        logger.info("Aggregate Style Transfer: {}".format(aggregate_style_transfer))

        aggregate_content_preservation = np.mean(np.asarray(validation_content_preservation_scores))
        logger.info("Aggregate Content Preservation: {}".format(aggregate_content_preservation))

        aggregate_word_overlap = np.mean(np.asarray(validation_word_overlap_scores))
        logger.info("Aggregate Word Overlap: {}".format(aggregate_word_overlap))

        with open(global_config.validation_scores_path, 'a+') as validation_scores_file:
            validation_record = {
                "epoch": current_epoch,
                "style-transfer": aggregate_style_transfer,
                "content-preservation": aggregate_content_preservation,
                "word-overlap": aggregate_word_overlap
            }
            validation_scores_file.write(json.dumps(validation_record) + "\n")

    def generate_novel_sentences(self, sess, padded_sequences, text_sequence_lengths, style_embedding,
                                 num_labels, model_save_path):

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=model_save_path)

        data_size = len(padded_sequences)
        generated_sequences = list()
        final_sequence_lengths = list()
        overall_label_predictions = list()
        style_label_predictions = list()
        adversarial_label_predictions = list()
        cross_entropy_scores = list()
        num_batches = data_size // mconf.batch_size
        if data_size % mconf.batch_size:
            num_batches += 1

        # these won't be needed to generate new sentences, so just use random numbers
        one_hot_labels_placeholder = np.random.randint(
            low=0, high=1, size=(data_size, num_labels)).astype(dtype=np.int32)

        end_index = None
        current_epoch = 0
        for batch_number in range(num_batches):
            (start_index, end_index) = data_processor.get_batch_indices(
                batch_number=batch_number, data_limit=data_size, batch_size=mconf.batch_size)

            style_prior = self.get_style_prior(label, num_labels, end_index - start_index)

            generated_sequences_batch, final_sequence_lengths_batch, \
            overall_label_predictions_batch, style_label_predictions_batch, \
            adversarial_label_predictions_batch, cross_entropy_score = \
                self.run_batch(
                    sess,
                    [self.inference_output, self.final_sequence_lengths,
                     self.quantized_style_overall_prediction,
                     self.quantized_style_multitask_prediction,
                     self.quantized_style_adversary_prediction,
                     self.reconstruction_loss], style_prior,
                    padded_sequences[start_index:end_index],
                    one_hot_labels_placeholder[start_index:end_index],
                    text_sequence_lengths[start_index:end_index],
                    True, current_epoch)

            generated_sequences.extend(generated_sequences_batch)
            final_sequence_lengths.extend(final_sequence_lengths_batch)
            overall_label_predictions.extend(overall_label_predictions_batch)
            style_label_predictions.extend(style_label_predictions_batch)
            adversarial_label_predictions.extend(adversarial_label_predictions_batch)
            cross_entropy_scores.append(cross_entropy_score)

        return generated_sequences, final_sequence_lengths, overall_label_predictions, \
               style_label_predictions, adversarial_label_predictions, cross_entropy_scores
