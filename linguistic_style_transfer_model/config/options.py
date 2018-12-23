import argparse
from typing import Any


class Options(argparse.Namespace):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.logging_level = None
        self.train_model = None
        self.transform_text = None
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
        self.evaluation_label_file_path = None
        self.num_sentences_to_generate = None
        self.label_index = None
