# batch settings
batch_size = 32

# layer sizes
encoder_rnn_size = 1024
decoder_rnn_size = 1024
style_embedding_size = 64
content_embedding_size = 512

# dropout
sequence_word_keep_prob = 0.6
recurrent_state_keep_prob = 0.6
fully_connected_keep_prob = 0.6
style_embedding_keep_prob = 0.6

# learning rates
autoencoder_learning_rate = 0.0001
adversarial_discriminator_learning_rate = 0.0001

# loss weights
adversarial_discriminator_loss_weight = 1
style_prediction_loss_weight = 1
bow_prediction_loss_weight = 0.1

# training iterations
adversarial_discriminator_iterations = 3
autoencoder_iterations = 1

# noise settings
adversarial_discriminator_noise_stddev = 0.1

# annealing
autoencoder_annealment_min_epoch = 50

# noise
epsilon = 1e-8
