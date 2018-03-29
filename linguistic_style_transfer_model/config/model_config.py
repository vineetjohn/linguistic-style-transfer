# batch settings
batch_size = 32

# layer sizes
encoder_rnn_size = 1024
decoder_rnn_size = 1024
style_embedding_size = 512
content_embedding_size = 512

# dropout
sequence_word_keep_prob = 0.6
recurrent_state_keep_prob = 0.6
fully_connected_keep_prob = 0.5

# learning rates
autoencoder_learning_rate = 0.00001
adversarial_discriminator_learning_rate = 0.00001

# gradient clipping values
autoencoder_gradient_clip_value = 1.0
adversarial_discriminator_gradient_clip_value = 0.01

# loss weights
adversarial_discriminator_loss_weight = 10
style_prediction_loss_weight = 1

# training iterations
adversarial_discriminator_iterations = 1
autoencoder_iterations = 1

# decoding settings
beam_search_width = 3

# noise settings
adversarial_discriminator_noise_stddev = 0.2
