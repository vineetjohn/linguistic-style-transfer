logger_name = "style_transfer"
embedding_size = 300
word_vector_path = "./word-embeddings/"

bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}

model_config = {
    "batch_size": 32,
    "encoder_rnn_size": 512,
    "recurrent_state_keep_prob": 0.8,
    "fully_connected_keep_prob": 0.5,
    "gradient_clipping_value": 1.0,
    "optimizer_learning_rate": 0.0001,
    "beam_search_width": 5,
    "model_save_path": "./saved-models/model.ckpt"
}
