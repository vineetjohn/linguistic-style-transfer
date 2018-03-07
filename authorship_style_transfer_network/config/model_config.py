from authorship_style_transfer_network.config import global_config

batch_size = 32
encoder_rnn_size = 128
recurrent_state_keep_prob = 0.8
fully_connected_keep_prob = 0.5
gradient_clipping_value = 1.0
optimizer_learning_rate = 0.0001
beam_search_width = 5
model_save_path = global_config.save_directory + "/text_style_transfer_model.ckpt"
