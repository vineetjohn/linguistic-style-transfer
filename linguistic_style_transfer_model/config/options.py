from types import SimpleNamespace


class Options(SimpleNamespace):

    def __init__(self):
        self.logging_level = None
        self.train_model = None
        self.generate_novel_text = None
        self.vocab_size = None
        self.training_epochs = None
        self.text_file_path = None
        self.label_file_path = None
        self.validation_text_file_path = None
        self.validation_label_file_path = None
        self.training_embeddings_file_path = None
        self.validation_embeddings_file_path = None
        self.saved_model_path = None
        self.classifier_saved_model_path = None
        self.dump_embeddings = None
        self.evaluation_text_file_path = None
