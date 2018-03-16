logger_name = "linguistic_style_transfer"
vocab_size = None
training_epochs = None
embedding_size = 300
max_sequence_length = 20
word_vector_path = "./word-embeddings/"
save_directory = "./saved-models"
model_save_path = save_directory + "/linguistic_style_transfer_model.ckpt"
all_style_embeddings_path = save_directory + "/all_style_embeddings.pkl"
label_mapped_style_embeddings_path = save_directory + "/label_mapped_style_embeddings.pkl"
unk_token = "<unk>"
sos_token = "<sos>"
eos_token = "<eos>"
bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}
