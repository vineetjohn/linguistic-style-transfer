logger_name = "linguistic_style_transfer"

vocab_size = None
training_epochs = None

embedding_size = 300
max_sequence_length = 20

word_vector_path = "./word-embeddings/"

save_directory = "./saved-models"
model_save_path = save_directory + "/linguistic_style_transfer_model.ckpt"
all_style_embeddings_path = save_directory + "/all_style_embeddings.pkl"
all_content_embeddings_path = save_directory + "/all_content_embeddings.pkl"
label_mapped_style_embeddings_path = save_directory + "/label_mapped_style_embeddings.pkl"
label_names_path = save_directory + "/label_names.pkl"
style_coordinates_path = save_directory + "/style_coordinates.pkl"
content_coordinates_path = save_directory + "/content_coordinates.pkl"
style_embedding_plot_path = save_directory + "/tsne_embeddings_plot_style.svg"
content_embedding_plot_path = save_directory + "/tsne_embeddings_plot_content.svg"
style_embedding_custom_plot_path = save_directory + "/tsne_embeddings_custom_plot_style.svg"
content_embedding_custom_plot_path = save_directory + "/tsne_embeddings_custom_plot_content.svg"

unk_token = "<unk>"
sos_token = "<sos>"
eos_token = "<eos>"
predefined_word_index = {
    unk_token: 0,
    sos_token: 1,
    eos_token: 2,
}

bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}
