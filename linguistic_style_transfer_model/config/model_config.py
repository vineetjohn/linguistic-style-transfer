# batch settings
batch_size = 32

# layer sizes
encoder_rnn_size = 1024
decoder_rnn_size = 1024
style_embedding_size = 8
content_embedding_size = 1024

# dropout
sequence_word_keep_prob = 0.8
recurrent_state_keep_prob = 0.8
fully_connected_keep_prob = 0.6

# learning rates
autoencoder_learning_rate = 0.0005
adversarial_discriminator_learning_rate = 0.0005

# gradient clipping values
autoencoder_gradient_clip_value = 5.0
adversarial_discriminator_gradient_clip_value = 0.01

# loss weights
adversarial_discriminator_loss_weight = 1
style_prediction_loss_weight = 1

# training iterations
adversarial_discriminator_iterations = 5

# decoding settings
beam_search_width = 5
