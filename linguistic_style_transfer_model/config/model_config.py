class ModelConfig():

    def __init__(self):
        # batch settings
        self.batch_size = 128

        # layer sizes
        self.encoder_rnn_size = 256
        self.decoder_rnn_size = 256
        self.style_embedding_size = 8
        self.content_embedding_size = 128

        # dropout
        self.sequence_word_keep_prob = 0.8
        self.recurrent_state_keep_prob = 0.8
        self.fully_connected_keep_prob = 0.8

        # learning rates
        self.autoencoder_learning_rate = 0.001
        self.adversarial_discriminator_learning_rate = 0.001

        # loss weights
        self.adversarial_discriminator_loss_weight = 0.3
        self.style_prediction_loss_weight = 1
        # self.adversarial_bow_loss_weight = 0.0001

        # training iterations
        self.kl_anneal_iterations = 20000
        self.style_kl_lambda = 0.03
        self.content_kl_lambda = 0.03

        # noise
        self.epsilon = 1e-8
        self.beam_search_width = 10

    def init_from_dict(self, previous_config):
        for key in previous_config:
            setattr(self, key, previous_config[key])


mconf = ModelConfig()
