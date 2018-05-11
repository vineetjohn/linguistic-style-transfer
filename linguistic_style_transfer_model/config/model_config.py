# batch settings
batch_size = 128

# layer sizes
encoder_rnn_size = 256
decoder_rnn_size = 256
style_embedding_size = 8
content_embedding_size = 128

# dropout
sequence_word_keep_prob = 0.8
recurrent_state_keep_prob = 0.8
fully_connected_keep_prob = 0.8

# learning rates
autoencoder_learning_rate = 0.001
adversarial_discriminator_learning_rate = 0.001

# loss weights
adversarial_discriminator_loss_weight = 0.3
style_prediction_loss_weight = 1

# training iterations
adversarial_discriminator_iterations = 1
autoencoder_iterations = 1

# noise
epsilon = 1e-8
