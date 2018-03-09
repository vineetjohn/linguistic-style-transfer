logger_name = "style_transfer"
embedding_size = 300
max_sequence_length = 20
word_vector_path = "./word-embeddings/"
save_directory = "./saved-models"
author_embedding_path = save_directory + "/style_embeddings.pkl"
bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}
