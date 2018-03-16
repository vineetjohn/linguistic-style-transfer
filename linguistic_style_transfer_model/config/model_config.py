from linguistic_style_transfer_model.config import global_config

batch_size = 32
encoder_rnn_size = 256
decoder_rnn_size = 256
style_embedding_size = 8
content_embedding_size = 512
sequence_word_keep_prob = 0.8
recurrent_state_keep_prob = 0.8
fully_connected_keep_prob = 0.75
gradient_clipping_value = 5.0
generator_learning_rate = 0.0005
adversarial_discriminator_learning_rate = 0.001
beam_search_width = 10
model_save_path = global_config.save_directory + "/linguistic_style_transfer_model.ckpt"
