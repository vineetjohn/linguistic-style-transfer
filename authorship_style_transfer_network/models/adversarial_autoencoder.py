import tensorflow as tf

from datetime import datetime as dt


class AdversarialAutoencoder():

    def __init__(self, num_labels, max_sequence_length, vocab_size, sos_index, eos_index,
                 encoder_embedding_matrix, decoder_embedding_matrix, padded_sequences, one_hot_labels,
                 text_sequence_lengths):
        self.batch_size = 100
        self.style_embedding_size = 512
        self.content_embedding_size = 512
        self.num_labels = num_labels
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.encoder_embedding_matrix = encoder_embedding_matrix
        self.decoder_embedding_matrix = decoder_embedding_matrix
        self.padded_sequences = padded_sequences
        self.one_hot_labels = one_hot_labels
        self.text_sequence_lengths = text_sequence_lengths

    def get_sentence_representation(self, embedded_sequence):

        with tf.name_scope('sentence_representation'):
            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell=tf.contrib.rnn.BasicLSTMCell(num_units=256),
                input_keep_prob=0.75,
                output_keep_prob=0.75,
                state_keep_prob=0.75)

            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell=tf.contrib.rnn.BasicLSTMCell(num_units=256),
                input_keep_prob=0.75,
                output_keep_prob=0.75,
                state_keep_prob=0.75)

            _, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw,
                inputs=embedded_sequence,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            sentence_representation_dense = tf.concat(
                values=[encoder_states[0].h, encoder_states[1].h], axis=1)

            sentence_representation = tf.nn.dropout(
                x=sentence_representation_dense, keep_prob=0.75)

            return sentence_representation

    def get_content_representation(self, sentence_representation):

        with tf.name_scope('content_representation'):
            content_representation_dense = tf.layers.dense(
                inputs=sentence_representation, units=self.content_embedding_size,
                activation=tf.nn.relu, name="content_representation")

            content_representation = tf.nn.dropout(
                x=content_representation_dense, keep_prob=0.75)

            return content_representation

    def get_style_representation(self, sentence_representation):

        with tf.name_scope('style_representation'):
            style_representation_dense = tf.layers.dense(
                inputs=sentence_representation, units=self.style_embedding_size,
                activation=tf.nn.relu, name="style_representation")

            style_representation = tf.nn.dropout(
                x=style_representation_dense, keep_prob=0.75)

            return style_representation

    def get_label_prediction(self, content_representation):

        with tf.name_scope('label_prediction'):
            label_projection = tf.layers.dense(
                inputs=content_representation, units=self.num_labels,
                activation=tf.nn.relu, name="label_prediction")

            label_prediction = tf.nn.softmax(label_projection)

            return label_prediction

    def generate_output_sequence(self, embedded_sequence, generative_embedding,
                                 decoder_embeddings):

        def get_training_decoder_output():
            decoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=256),
                input_keep_prob=0.75,
                output_keep_prob=0.75,
                state_keep_prob=0.75)

            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_sequence,
                sequence_length=tf.fill([self.batch_size], self.max_sequence_length))

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=training_helper,
                initial_state=decoder_cell.zero_state(self.batch_size, tf.float32),
                output_layer=tf.layers.Dense(
                    units=self.vocab_size, activation=tf.nn.relu))

            # Dynamic decoding
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, impute_finished=True,
                maximum_iterations=self.max_sequence_length)

            return training_decoder_output

        def get_inference_decoder_output():
            decoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=256),
                input_keep_prob=0.75,
                output_keep_prob=0.75,
                state_keep_prob=0.75)

            greedy_embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=decoder_embeddings,
                start_tokens=tf.fill([self.batch_size], self.sos_index),
                end_token=self.eos_index)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=greedy_embedding_helper,
                initial_state=decoder_cell.zero_state(self.batch_size, tf.float32),
                output_layer=tf.layers.Dense(
                    units=self.vocab_size, activation=tf.nn.relu))

            # Dynamic decoding
            inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, impute_finished=True,
                maximum_iterations=self.max_sequence_length)

            return inference_decoder_output

        decoder_output = tf.cond(
            pred=self.training_phase,
            true_fn=get_training_decoder_output,
            false_fn=get_inference_decoder_output,
            name="training_inference_conditional")

        return decoder_output.rnn_output

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

        self.training_phase = tf.placeholder(
            dtype=tf.bool, name="training_phase")
        print("training_phase: {}".format(self.training_phase))

        # word embeddings matrix
        encoder_embeddings = tf.get_variable(
            initializer=self.encoder_embedding_matrix, dtype=tf.float32,
            name="encoder_embeddings")
        print("encoder_embeddings: {}".format(encoder_embeddings))

        encoder_embedded_sequence = tf.nn.embedding_lookup(
            params=encoder_embeddings, ids=self.input_sequence,
            name="encoder_embedded_sequence")
        print("encoder_embedded_sequence: {}".format(encoder_embedded_sequence))

        # get sentence representation
        sentence_representation = self.get_sentence_representation(
            encoder_embedded_sequence)
        print("sentence_representation: {}".format(sentence_representation))

        # get content representation
        content_representation = self.get_content_representation(
            sentence_representation)
        print("content_representation: {}".format(content_representation))

        # get style representation
        self.style_representation = self.get_style_representation(
            sentence_representation)
        print("style_representation: {}".format(self.style_representation))

        # use content representation to predict a label
        self.label_prediction = self.get_label_prediction(
            content_representation)
        print("label_prediction: {}".format(self.label_prediction))

        self.adversarial_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.input_label, logits=self.label_prediction)
        print("adversarial_loss: {}".format(self.adversarial_loss))

        # generate new sentence
        with tf.name_scope('generative_embedding'):
            generative_embedding_concatenated = tf.concat(
                values=[self.style_representation, content_representation], axis=1)

            generative_embedding_dense = tf.layers.dense(
                inputs=generative_embedding_concatenated,
                units=512,
                activation=tf.nn.relu)

            generative_embedding = tf.nn.dropout(
                x=generative_embedding_dense, keep_prob=0.75)
            print("generative_embedding: {}".format(generative_embedding))

        decoder_embeddings = tf.get_variable(
            initializer=self.decoder_embedding_matrix, dtype=tf.float32,
            name="decoder_embeddings")
        print("decoder_embeddings: {}".format(decoder_embeddings))

        decoder_embedded_sequence = tf.nn.embedding_lookup(
            params=decoder_embeddings, ids=self.input_sequence,
            name="decoder_embedded_sequence")
        print("decoder_embedded_sequence: {}".format(decoder_embedded_sequence))

        with tf.name_scope('sequence_prediction'):
            self.sequence_prediction = \
                self.generate_output_sequence(
                    decoder_embedded_sequence, generative_embedding, decoder_embeddings)

        with tf.name_scope('reconstruction_loss'):
            output_sequence_mask = tf.sequence_mask(
                lengths=self.sequence_lengths, maxlen=self.max_sequence_length,
                dtype=tf.float32)

            self.reconstruction_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.sequence_prediction, targets=self.input_sequence,
                weights=output_sequence_mask)

            print("reconstruction_loss: {}".format(self.reconstruction_loss))

        # loss summaries for tensorboard logging
        self.adversarial_loss_summary = tf.summary.scalar(
            tensor=self.adversarial_loss, name="adversarial_loss_summary")

        self.reconstruction_loss_summary = tf.summary.scalar(
            tensor=self.reconstruction_loss, name="reconstruction_loss_summary")

    def get_batch_indices(self, offset, batch_size, batch_number, data_limit):

        start_index = offset + (batch_number * batch_size)
        end_index = offset + ((batch_number + 1) * batch_size)

        end_index = data_limit if end_index > data_limit else end_index

        return (start_index, end_index)

    def run_batch(self, sess, start_index, end_index, training_phase, fetches):

        ops = sess.run(
            fetches=fetches,
            feed_dict={
                self.input_sequence: self.padded_sequences[start_index: end_index],
                self.input_label: self.one_hot_labels[start_index: end_index],
                self.sequence_lengths: self.text_sequence_lengths[start_index: end_index],
                self.training_phase: training_phase
            })

        return ops

    def train(self, sess, data_size, training_epochs):

        writer = tf.summary.FileWriter(
            logdir="/tmp/tensorflow_logs/" + dt.now().strftime("%Y%m%d-%H%M%S") + "/",
            graph=sess.graph)

        adversarial_training_optimizer = tf.train.AdamOptimizer()
        adversarial_training_operation = adversarial_training_optimizer.minimize(
            self.adversarial_loss)

        reconstruction_training_optimizer = tf.train.AdamOptimizer()
        reconstruction_training_operation = reconstruction_training_optimizer.minimize(
            self.reconstruction_loss - self.adversarial_loss)

        sess.run(tf.global_variables_initializer())

        epoch_reporting_interval = 1
        training_examples_size = data_size
        num_batches = training_examples_size // self.batch_size
        print("Training - texts shape: {}; labels shape {}"
              .format(self.padded_sequences[:training_examples_size].shape,
                      self.one_hot_labels[:training_examples_size].shape))

        for current_epoch in range(1, training_epochs + 1):
            self.all_style_representations = list()
            for batch_number in range(num_batches):
                (start_index, end_index) = self.get_batch_indices(
                    offset=0, batch_size=self.batch_size,
                    batch_number=batch_number, data_limit=data_size)

                fetches = [adversarial_training_operation,
                           self.adversarial_loss,
                           self.adversarial_loss_summary,
                           reconstruction_training_operation,
                           self.reconstruction_loss,
                           self.reconstruction_loss_summary,
                           self.style_representation]

                _, adv_loss, adv_loss_sum, _, rec_loss, \
                rec_loss_sum, style_embeddings = self.run_batch(
                    sess, start_index, end_index, True, fetches)

                self.all_style_representations.extend(style_embeddings)

            writer.add_summary(adv_loss_sum, current_epoch)
            writer.add_summary(rec_loss_sum, current_epoch)
            writer.flush()

            if (current_epoch % epoch_reporting_interval == 0):
                print("Training epoch: {}; Reconstruction loss: {}; Adversarial loss {}" \
                      .format(current_epoch, rec_loss, adv_loss))

        writer.close()

    def infer(self, sess, offset, samples_size):

        generated_sequences = list()
        num_batches = samples_size // self.batch_size

        for batch_number in range(num_batches + 1):

            (start_index, end_index) = self.get_batch_indices(
                offset=offset, batch_size=self.batch_size,
                batch_number=batch_number, data_limit=(offset + samples_size))

            if start_index == end_index:
                break

            generated_sequences_batch = self.run_batch(
                sess, start_index, end_index, False, self.sequence_prediction)

            generated_sequences.extend(generated_sequences_batch)

        return generated_sequences
