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
    validation_text_file_path = None
    validation_label_file_path = None
    validation_embeddings_file_path = None
    classifier_checkpoint_dir = None
    dump_embeddings = None
    evaluation_text_file_path = None
    use_pretrained_embeddings = None
