import logging
from datetime import datetime as dt

import tensorflow as tf

logger = logging.getLogger('root')


class AdversarialAutoencoder:

    def __init__(self, num_labels, max_sequence_length, vocab_size, sos_index, eos_index,
                 encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences, one_hot_labels,
                 text_sequence_lengths, label_sequences):
        self.batch_size = 32
        self.encoder_rnn_size = 512
        self.recurrent_state_keep_prob = 0.8
        self.fully_connected_keep_prob = 0.5
        self.gradient_clipping_value = 1.0
        self.optimizer_learning_rate = 0.0001
        self.beam_search_width = 5
        self.num_labels = num_labels
        self.label_sequences = label_sequences
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.encoder_embedding_matrix = encoder_embedding_matrix
        self.decoder_embedding_matrix = decoder_embedding_matrix
        self.padded_sequences = padded_sequences
        self.one_hot_labels = one_hot_labels
        self.text_sequence_lengths = text_sequence_lengths
        self.model_save_path = "./saved-models/model.ckpt"

        # declare model fetches and placeholders
        self.input_sequence, self.input_label, self.sequence_lengths, \
            self.reconstruction_loss, self.adversarial_loss, self.inference_output, \
            self.all_summaries, self.final_sequence_lengths \
            = None, None, None, None, None, None, None, None

    def get_style_embedding(self, embedded_sequence):

        scope_name = "style_embedding"

        with tf.name_scope(scope_name):
            encoder_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=self.encoder_rnn_size),
                input_keep_prob=self.recurrent_state_keep_prob,
                output_keep_prob=self.recurrent_state_keep_prob,
                state_keep_prob=self.recurrent_state_keep_prob)
            encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=self.encoder_rnn_size),
                input_keep_prob=self.recurrent_state_keep_prob,
                output_keep_prob=self.recurrent_state_keep_prob,
                state_keep_prob=self.recurrent_state_keep_prob)

            _, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw,
                cell_bw=encoder_cell_bw,
                inputs=embedded_sequence,
                scope=scope_name,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            return tf.concat(values=encoder_states, axis=1)

    def get_content_embedding(self, embedded_sequence):

        scope_name = "content_embedding"

        with tf.name_scope(scope_name):
            encoder_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=self.encoder_rnn_size),
                input_keep_prob=self.recurrent_state_keep_prob,
                output_keep_prob=self.recurrent_state_keep_prob,
                state_keep_prob=self.recurrent_state_keep_prob)
            encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.GRUCell(num_units=self.encoder_rnn_size),
                input_keep_prob=self.recurrent_state_keep_prob,
                output_keep_prob=self.recurrent_state_keep_prob,
                state_keep_prob=self.recurrent_state_keep_prob)

            _, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw,
                cell_bw=encoder_cell_bw,
                inputs=embedded_sequence,
                scope=scope_name,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            return tf.concat(values=encoder_states, axis=1)

    def get_label_prediction(self, content_embedding):

        with tf.name_scope('label_prediction'):
            label_projection = tf.layers.dense(
                inputs=content_embedding, units=self.num_labels,
                name="label_prediction")
            label_prediction = tf.nn.softmax(label_projection)
            return label_prediction

    def generate_output_sequence(self, embedded_sequence, generative_embedding, decoder_embeddings):

        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.contrib.rnn.GRUCell(num_units=self.encoder_rnn_size * 4),
            input_keep_prob=self.recurrent_state_keep_prob,
            output_keep_prob=self.recurrent_state_keep_prob,
            state_keep_prob=self.recurrent_state_keep_prob)

        projection_layer = tf.layers.Dense(units=self.vocab_size, use_bias=False)

        with tf.name_scope('training_decoder'):
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
                maximum_iterations=self.max_sequence_length,
                scope="training_decoder")

        with tf.name_scope('inference_decoder'):
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, embedding=decoder_embeddings,
                start_tokens=tf.fill([self.batch_size], self.sos_index),
                end_token=self.eos_index,
                initial_state=tf.contrib.seq2seq.tile_batch(
                    t=generative_embedding, multiplier=self.beam_search_width),
                beam_width=self.beam_search_width, output_layer=projection_layer,
                length_penalty_weight=0.0
            )
            inference_decoder.initialize("inference_decoder")

            inference_decoder_output, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, impute_finished=False,
                maximum_iterations=self.max_sequence_length,
                scope="inference_decoder")

        return training_decoder_output.rnn_output, \
            inference_decoder_output.predicted_ids[:, :, 0], \
            final_sequence_lengths[:, 0]  # index 0 gets the best beam search outcome

    def build_model(self):

        # model inputs
        self.input_sequence = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size, self.max_sequence_length],
            name="input_sequence")
        logger.debug("input_sequence: {}".format(self.input_sequence))

        self.input_label = tf.placeholder(
            dtype=tf.float32, shape=[self.batch_size, self.num_labels],
            name="input_label")
        logger.debug("input_label: {}".format(self.input_label))

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size],
            name="sequence_lengths")
        logger.debug("sequence_lengths: {}".format(self.sequence_lengths))

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
            keep_prob=self.fully_connected_keep_prob,
            name="encoder_embedded_sequence")
        logger.debug("encoder_embedded_sequence: {}".format(encoder_embedded_sequence))

        decoder_input = tf.concat(
            values=[tf.fill([self.batch_size, 1], self.sos_index), self.input_sequence], axis=1)
        decoder_embedded_sequence = tf.nn.dropout(
            x=tf.nn.embedding_lookup(params=decoder_embeddings, ids=decoder_input),
            keep_prob=self.fully_connected_keep_prob,
            name="decoder_embedded_sequence")
        logger.debug("decoder_embedded_sequence: {}".format(decoder_embedded_sequence))

        # style embedding
        style_embedding = self.get_style_embedding(encoder_embedded_sequence)
        logger.debug("style_embedding: {}".format(style_embedding))

        # content embedding
        content_embedding = self.get_content_embedding(encoder_embedded_sequence)
        logger.debug("content_embedding: {}".format(content_embedding))

        # concatenated generative embedding
        generative_embedding = tf.concat(values=[style_embedding, content_embedding], axis=1)
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
            label_prediction = self.get_label_prediction(content_embedding)
            logger.debug("label_prediction: {}".format(label_prediction))

            self.adversarial_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=label_prediction)
            logger.debug("adversarial_loss: {}".format(self.adversarial_loss))

        # reconstruction loss
        with tf.name_scope('reconstruction_loss'):
            batch_maxlen = tf.reduce_max(self.sequence_lengths)
            logger.debug("batch_maxlen: {}".format(batch_maxlen))

            # the training decoder only emits outputs equal in time-steps to the
            # max time in the current batch
            target_sequence = tf.slice(
                input_=self.input_sequence,
                begin=[0, 0],
                size=[self.batch_size, batch_maxlen],
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
        tf.summary.scalar(tensor=self.adversarial_loss, name="adversarial_loss_summary")
        self.all_summaries = tf.summary.merge_all()

    def get_batch_indices(self, offset, batch_number, data_limit):

        start_index = offset + (batch_number * self.batch_size)
        end_index = offset + ((batch_number + 1) * self.batch_size)
        end_index = data_limit if end_index > data_limit else end_index

        return start_index, end_index

    def run_batch(self, sess, start_index, end_index, fetches):

        ops = sess.run(
            fetches=fetches,
            feed_dict={
                self.input_sequence: self.padded_sequences[start_index: end_index],
                self.input_label: self.one_hot_labels[start_index: end_index],
                self.sequence_lengths: self.text_sequence_lengths[start_index: end_index]
            })

        return ops

    def train(self, sess, data_size, training_epochs):

        writer = tf.summary.FileWriter(
            logdir="/tmp/tensorflow_logs/" + dt.now().strftime("%Y%m%d-%H%M%S") + "/",
            graph=sess.graph)

        trainable_variables = tf.trainable_variables()
        logger.debug("trainable_variables: {}".format(trainable_variables))

        reconstruction_training_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.optimizer_learning_rate)
        reconstruction_gradients_and_variables = reconstruction_training_optimizer.compute_gradients(
            loss=self.reconstruction_loss, var_list=trainable_variables)
        gradients, variables = zip(*reconstruction_gradients_and_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(
            t_list=gradients, clip_norm=self.gradient_clipping_value)
        reconstruction_training_operation = reconstruction_training_optimizer.apply_gradients(
            grads_and_vars=zip(clipped_gradients, variables))

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        training_examples_size = data_size
        num_batches = training_examples_size // self.batch_size
        logger.debug("Training - texts shape: {}; labels shape {}"
                     .format(self.padded_sequences[:training_examples_size].shape,
                             self.one_hot_labels[:training_examples_size].shape))

        adv_loss, rec_loss, all_summaries = (None, None, None)
        for current_epoch in range(1, training_epochs + 1):
            for batch_number in range(num_batches):
                (start_index, end_index) = self.get_batch_indices(
                    offset=0, batch_number=batch_number, data_limit=data_size)

                fetches = \
                    [reconstruction_training_operation,
                     self.reconstruction_loss,
                     self.all_summaries]

                _, rec_loss, all_summaries = self.run_batch(
                    sess, start_index, end_index, fetches)

            saver.save(sess=sess, save_path=self.model_save_path)
            writer.add_summary(all_summaries, current_epoch)
            writer.flush()

            logger.info("Reconstruction loss: {:.9f}; Training epoch: {}"
                        .format(rec_loss, current_epoch))
        writer.close()

    def infer(self, sess, offset, samples_size):

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=self.model_save_path)

        generated_sequences = list()
        final_sequence_lengths = list()
        num_batches = samples_size // self.batch_size

        end_index = None
        for batch_number in range(num_batches):

            (start_index, end_index) = self.get_batch_indices(
                offset=offset, batch_number=batch_number, data_limit=(offset + samples_size))

            if start_index == end_index:
                break

            generated_sequences_batch, final_sequence_lengths_batch = self.run_batch(
                sess, start_index, end_index, [self.inference_output, self.final_sequence_lengths])

            generated_sequences.extend(generated_sequences_batch)
            final_sequence_lengths.extend(final_sequence_lengths_batch)

        return generated_sequences, end_index, final_sequence_lengths
