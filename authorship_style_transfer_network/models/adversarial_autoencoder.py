from datetime import datetime as dt

import tensorflow as tf


class AdversarialAutoencoder:

    def __init__(self, num_labels, max_sequence_length, vocab_size, sos_index, eos_index,
                 encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences, one_hot_labels,
                 text_sequence_lengths, label_sequences):
        self.batch_size = 128
        self.encoder_rnn_size = 256
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
        self.input_sequence, self.input_label, self.sequence_lengths, \
            self.reconstruction_loss, self.inference_output, \
            self.all_summaries = None, None, None, None, None, None

    def get_sentence_representation(self, embedded_sequence):

        with tf.name_scope('sentence_representation'):

            encoder_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.encoder_rnn_size),
                input_keep_prob=self.recurrent_state_keep_prob,
                output_keep_prob=self.recurrent_state_keep_prob,
                state_keep_prob=self.recurrent_state_keep_prob)

            _, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_lstm_cell,
                inputs=embedded_sequence,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            return encoder_state

    def get_label_prediction(self, content_representation):

        with tf.name_scope('label_prediction'):
            label_projection = tf.layers.dense(
                inputs=content_representation, units=self.num_labels,
                name="label_prediction")
            label_prediction = tf.nn.softmax(label_projection)
            return label_prediction

    def generate_output_sequence(self, embedded_sequence, encoder_state, decoder_embeddings):

        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.encoder_rnn_size),
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
                initial_state=encoder_state,
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
                initial_state=tf.contrib.rnn.LSTMStateTuple(
                    c=tf.contrib.seq2seq.tile_batch(
                        t=encoder_state.c, multiplier=self.beam_search_width),
                    h=tf.contrib.seq2seq.tile_batch(
                        t=encoder_state.h, multiplier=self.beam_search_width)),
                beam_width=self.beam_search_width, output_layer=projection_layer,
                length_penalty_weight=0.0
            )
            inference_decoder.initialize("inference_decoder")

            inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, impute_finished=False,
                maximum_iterations=self.max_sequence_length,
                scope="inference_decoder")

        return training_decoder_output.rnn_output, inference_decoder_output.predicted_ids

    def build_model(self):

        self.input_sequence = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size, self.max_sequence_length],
            name="input_sequence")
        print("input_sequence: {}".format(self.input_sequence))

        self.input_label = tf.placeholder(
            dtype=tf.float32, shape=[self.batch_size, self.num_labels],
            name="input_label")
        print("input_label: {}".format(self.input_label))

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size],
            name="sequence_lengths")
        print("sequence_lengths: {}".format(self.sequence_lengths))

        # word embeddings matrices
        encoder_embeddings = tf.get_variable(
            initializer=self.encoder_embedding_matrix, dtype=tf.float32,
            trainable=True, name="encoder_embeddings")
        print("encoder_embeddings: {}".format(encoder_embeddings))

        decoder_embeddings = tf.get_variable(
            initializer=self.decoder_embedding_matrix, dtype=tf.float32,
            trainable=True, name="decoder_embeddings")
        print("decoder_embeddings: {}".format(decoder_embeddings))

        encoder_embedded_sequence = tf.nn.dropout(
            x=tf.nn.embedding_lookup(
                params=encoder_embeddings, ids=self.input_sequence),
            keep_prob=self.fully_connected_keep_prob,
            name="encoder_embedded_sequence")
        print("encoder_embedded_sequence: {}".format(encoder_embedded_sequence))

        # get sentence representation
        encoder_state = self.get_sentence_representation(
            encoder_embedded_sequence)
        print("encoder_state: {}".format(encoder_state))

        left_shifted_decoder_input = tf.strided_slice(
            input_=self.input_sequence, begin=[0, 0], end=[self.batch_size, -1], strides=[1, 1],)
        decoder_input = tf.concat(
            values=[tf.fill([self.batch_size, 1], self.sos_index), left_shifted_decoder_input], axis=1)

        decoder_embedded_sequence = tf.nn.dropout(
            x=tf.nn.embedding_lookup(params=decoder_embeddings, ids=decoder_input),
            keep_prob=self.fully_connected_keep_prob,
            name="decoder_embedded_sequence")
        print("decoder_embedded_sequence: {}".format(decoder_embedded_sequence))

        with tf.name_scope('sequence_prediction'):
            training_output, self.inference_output = \
                self.generate_output_sequence(
                    decoder_embedded_sequence, encoder_state, decoder_embeddings)
            print("training_output: {}".format(training_output))
            print("inference_output: {}".format(self.inference_output))

        with tf.name_scope('reconstruction_loss'):
            output_sequence_mask = tf.sequence_mask(
                lengths=self.sequence_lengths, maxlen=self.max_sequence_length,
                dtype=tf.float32)

            self.reconstruction_loss = tf.contrib.seq2seq.sequence_loss(
                logits=training_output, targets=self.input_sequence,
                weights=output_sequence_mask)
            print("reconstruction_loss: {}".format(self.reconstruction_loss))

        tf.summary.scalar(tensor=self.reconstruction_loss, name="reconstruction_loss_summary")
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
        print("trainable_variables: {}".format(trainable_variables))

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
        print("Training - texts shape: {}; labels shape {}"
              .format(self.padded_sequences[:training_examples_size].shape,
                      self.one_hot_labels[:training_examples_size].shape))

        adv_loss, rec_loss, all_summaries = (None, None, None)
        for current_epoch in range(1, training_epochs + 1):
            for batch_number in range(num_batches):
                (start_index, end_index) = self.get_batch_indices(
                    offset=0, batch_number=batch_number, data_limit=data_size)

                fetches = [
                    reconstruction_training_operation,
                    self.reconstruction_loss,
                    self.all_summaries]

                _, rec_loss, all_summaries = self.run_batch(
                    sess, start_index, end_index, fetches)

            saver.save(sess=sess, save_path=self.model_save_path)
            writer.add_summary(all_summaries, current_epoch)
            writer.flush()

            print("Reconstruction loss: {:.9f}; Training epoch: {}"
                  .format(rec_loss, current_epoch))
        writer.close()

    def infer(self, sess, offset, samples_size):

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=self.model_save_path)

        generated_sequences = list()
        num_batches = samples_size // self.batch_size

        for batch_number in range(num_batches + 1):

            (start_index, end_index) = self.get_batch_indices(
                offset=offset, batch_number=batch_number, data_limit=(offset + samples_size))

            if start_index == end_index:
                break

            generated_sequences_batch = self.run_batch(
                sess, start_index, end_index, self.inference_output)

            generated_sequences.extend(generated_sequences_batch)

        return generated_sequences
