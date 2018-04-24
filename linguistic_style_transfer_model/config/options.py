from types import SimpleNamespace


class Options(SimpleNamespace):
    logging_level = None
    train_model = None
    infer_sequences = None
    generate_novel_text = None
    vocab_size = None
    training_epochs = None
    text_file_path = None
    label_file_path = None
    use_pretrained_embeddings = None
